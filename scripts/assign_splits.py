#!/usr/bin/env python3
"""
assign_splits.py

Assigns each labeled image_id to exactly one of calibration/validation/
holdout and records it in evaluation_dataset/split.csv (EVALUATION_PROTOCOL.md
section 6 "leakage prevention" / collection-workflow sections 18-19).

SAFETY RULES (both enforced, not just documented):
  1. An image_id already present in split.csv is NEVER reassigned by a
     normal run -- only --reassign explicitly allows changing an existing
     assignment, and even then only for image_ids you name, never a blanket
     reshuffle. "Do not randomly reshuffle after calibration has started"
     is enforced structurally, not just requested.
  2. Once evaluation_dataset/dataset_status.json says FROZEN, this script
     refuses to add or change ANY split assignment at all -- a frozen
     ground-truth snapshot's splits are part of what's frozen.

The split assignment itself is a fixed-seed deterministic shuffle (Python's
random.Random(seed), NOT the bare random module / no Math.random-equivalent)
over the sorted list of not-yet-assigned image_ids, so re-running with the
same seed and the same input set reproduces the same result -- required for
"use a fixed random seed if random splitting is used."

Usage:
    python scripts/assign_splits.py \\
        --labels evaluation_dataset/labels.csv \\
        --split-file evaluation_dataset/split.csv \\
        --calibration 0.4 --validation 0.3 --holdout 0.3 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.eval.dataset_status import is_frozen

SPLITS = ("calibration", "validation", "holdout")


def load_split_csv(path: Path) -> dict:
    assignments = {}
    if not path.exists():
        return assignments
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            image_id = (row.get("image_id") or "").strip()
            split = (row.get("split") or "").strip()
            if image_id:
                assignments[image_id] = split
    return assignments


def load_label_ids(labels_path: Path) -> list:
    ids = []
    with open(labels_path, newline="") as f:
        for row in csv.DictReader(f):
            image_id = (row.get("image_id") or "").strip()
            if image_id:
                ids.append(image_id)
    return ids


def write_split_csv(path: Path, assignments: dict) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "split"])
        writer.writeheader()
        for image_id in sorted(assignments):
            writer.writerow({"image_id": image_id, "split": assignments[image_id]})


def allocate_splits(image_ids: list, ratios: dict, seed: int) -> dict:
    """Deterministic: same ids + same ratios + same seed -> same result,
    every time, on any machine (Python's Mersenne Twister is
    deterministic given a seed).
    """
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total} ({ratios})")

    ordered = sorted(image_ids)  # sort first so shuffle input order is itself deterministic
    rng = random.Random(seed)
    shuffled = ordered[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_cal = round(n * ratios["calibration"])
    n_val = round(n * ratios["validation"])
    # holdout takes the remainder so rounding never drops or duplicates an id
    assignments = {}
    for image_id in shuffled[:n_cal]:
        assignments[image_id] = "calibration"
    for image_id in shuffled[n_cal:n_cal + n_val]:
        assignments[image_id] = "validation"
    for image_id in shuffled[n_cal + n_val:]:
        assignments[image_id] = "holdout"
    return assignments


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=Path(__file__).resolve().parent.parent / "evaluation_dataset" / "labels.csv")
    ap.add_argument("--split-file", type=Path, default=Path(__file__).resolve().parent.parent / "evaluation_dataset" / "split.csv")
    ap.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parent.parent / "evaluation_dataset")
    ap.add_argument("--calibration", type=float, default=0.4)
    ap.add_argument("--validation", type=float, default=0.3)
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reassign", nargs="*", default=[], help="image_ids to explicitly allow re-splitting (never a blanket reshuffle)")
    args = ap.parse_args()

    if is_frozen(args.dataset):
        print("ERROR: dataset_status.json says FROZEN -- split assignments for a frozen "
              "dataset must not change. Unfreeze deliberately first if this is truly intended.", file=sys.stderr)
        sys.exit(1)

    existing = load_split_csv(args.split_file)
    all_ids = load_label_ids(args.labels)

    reassign_set = set(args.reassign)
    unassigned = [i for i in all_ids if i not in existing or i in reassign_set]

    if not unassigned:
        print(f"All {len(all_ids)} labeled image_id(s) already have a split assignment. Nothing to do.")
        print("(Pass --reassign <id> [<id> ...] to explicitly change specific existing assignments.)")
        return

    new_assignments = allocate_splits(
        unassigned, {"calibration": args.calibration, "validation": args.validation, "holdout": args.holdout}, args.seed,
    )

    merged = {**existing, **new_assignments}
    write_split_csv(args.split_file, merged)

    counts = {s: sum(1 for v in merged.values() if v == s) for s in SPLITS}
    print(f"Assigned {len(new_assignments)} new split(s) (seed={args.seed}).")
    print(f"Total split distribution: {counts}")
    print(f"Wrote {args.split_file}")


if __name__ == "__main__":
    main()
