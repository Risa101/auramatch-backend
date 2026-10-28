#!/usr/bin/env python3
"""
dataset_status.py

THE single authoritative freeze mechanism (EVALUATION_PROTOCOL.md section
27) -- read, freeze, unfreeze, or pre-freeze-check
evaluation_dataset/dataset_status.json. DRAFT = still being collected/
labeled. FROZEN = a fixed ground-truth/holdout snapshot from here on --
scripts/evaluate_personal_color.py's --require-frozen flag checks this
before running a real evaluation.

Usage:
    python scripts/dataset_status.py --show
    python scripts/dataset_status.py --preflight-freeze
    python scripts/dataset_status.py --freeze --by "Jane Doe" --notes "60/80 images labeled, both reviewers done"
    python scripts/dataset_status.py --unfreeze --reason "found 3 mislabeled rows, correcting before re-freezing"

--freeze runs the SAME checks --preflight-freeze does, first. A structural
error (invalid labels.csv, split leakage, a labeled image_id with no
landmarks export at all) blocks the freeze outright. A soft, human-judgment
item (duplicate images found, dataset below the protocol's size floor)
prints "MANUAL REVIEW REQUIRED" and blocks the freeze UNLESS
--acknowledge-warnings is also passed -- so a freeze can never happen by
someone skimming past a warning without an explicit, separate flag saying
they saw it.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.eval.dataset_stats import SEASONS, label_distribution, read_csv_rows
from services.eval.dataset_status import freeze, read_status, unfreeze
from services.eval.integrity import (
    check_all_landmarks, check_split_leakage, find_exact_duplicates,
    find_near_duplicates,
)
from services.eval.validate_labels import has_blocking_errors, validate_labels_csv

MIN_TOTAL = 60


def run_preflight_freeze(dataset_dir: Path) -> tuple:
    """Returns (hard_errors: list[str], soft_warnings: list[str]).
    Hard errors always block a freeze. Soft warnings block it too, unless
    the caller passes --acknowledge-warnings -- see module docstring.
    """
    hard, soft = [], []

    if not (dataset_dir / "EVALUATION_PROTOCOL.md").exists():
        hard.append("EVALUATION_PROTOCOL.md not found -- this dataset has no governing protocol")

    images_dir = dataset_dir / "images"
    landmarks_dir = dataset_dir / "landmarks"
    labels_path = dataset_dir / "labels.csv"
    split_path = dataset_dir / "split.csv"

    issues = validate_labels_csv(labels_path, images_dir=images_dir if images_dir.exists() else None)
    if has_blocking_errors(issues):
        hard.append(f"labels.csv has {sum(1 for i in issues if i.severity == 'error')} blocking validation error(s) -- run scripts/check_labels.py")

    labels = read_csv_rows(labels_path)
    image_ids = [r["image_id"] for r in labels if r.get("image_id")]

    if not split_path.exists() or not read_csv_rows(split_path):
        hard.append("split.csv is missing or empty -- run scripts/assign_splits.py before freezing")
    else:
        split_rows = read_csv_rows(split_path)
        leakage = check_split_leakage(split_rows)
        if leakage:
            hard.append(f"split.csv has {len(leakage)} leakage issue(s) -- {leakage[:3]}")
        split_ids = {r["image_id"] for r in split_rows}
        unassigned = [i for i in image_ids if i not in split_ids]
        if unassigned:
            hard.append(f"{len(unassigned)} labeled image_id(s) have no split assignment: {unassigned[:5]}...")

    if image_ids:
        landmark_states = check_all_landmarks(image_ids, landmarks_dir)
        missing = [r["image_id"] for r in landmark_states if r["state"] == "MISSING"]
        invalid = [r["image_id"] for r in landmark_states if r["state"] == "INVALID"]
        if missing:
            hard.append(f"{len(missing)} labeled image(s) have no landmarks exported yet: {missing[:5]}...")
        if invalid:
            hard.append(f"{len(invalid)} labeled image(s) have invalid/malformed landmark exports: {invalid[:5]}...")

    if images_dir.exists():
        exact = find_exact_duplicates(images_dir)
        near = find_near_duplicates(images_dir)
        if exact:
            soft.append(f"{len(exact)} exact-duplicate image group(s) found -- confirm reviewed (scripts/check_dataset_integrity.py)")
        if near:
            soft.append(f"{len(near)} near-duplicate image pair(s) found -- confirm reviewed (scripts/check_dataset_integrity.py)")

    season_known, _ = label_distribution(labels, "season", SEASONS)
    total_season_labeled = sum(season_known.values())
    if total_season_labeled < MIN_TOTAL:
        soft.append(
            f"only {total_season_labeled}/{MIN_TOTAL} season-labeled images -- below EVALUATION_PROTOCOL.md "
            f"section 8's floor. Freezing below this floor is allowed (the protocol calls it a practical "
            f"target, not a hard rule) but must be a deliberate choice, not an oversight."
        )

    current = read_status(dataset_dir)
    if current.get("status") != "FROZEN" and current.get("version") is None:
        soft.append("no dataset version recorded yet -- one will be auto-assigned on freeze (v1.0-frozen) unless --version is given")

    return hard, soft


def print_preflight_result(hard: list, soft: list) -> None:
    if not hard and not soft:
        print("All pre-freeze checks passed. No manual review items outstanding.")
        return
    if hard:
        print(f"BLOCKING ({len(hard)}):")
        for h in hard:
            print(f"  ✗ {h}")
    if soft:
        print(f"\nMANUAL REVIEW REQUIRED ({len(soft)}):")
        for s in soft:
            print(f"  ! {s}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parent.parent / "evaluation_dataset")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--show", action="store_true")
    group.add_argument("--preflight-freeze", action="store_true")
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--unfreeze", action="store_true")
    ap.add_argument("--by", help="required with --freeze: who is freezing this dataset")
    ap.add_argument("--notes", default="", help="optional with --freeze")
    ap.add_argument("--version", default=None, help="optional with --freeze: explicit version string, e.g. v2.0-frozen (auto-bumped if omitted)")
    ap.add_argument("--acknowledge-warnings", action="store_true",
                     help="required with --freeze if any MANUAL REVIEW REQUIRED item is outstanding")
    ap.add_argument("--reason", help="required with --unfreeze: why")
    args = ap.parse_args()

    if args.show:
        status = read_status(args.dataset)
        print(f"Status:    {status['status']}")
        print(f"Version:   {status.get('version') or '-'}")
        print(f"Frozen at: {status.get('frozen_at') or '-'}")
        print(f"Frozen by: {status.get('frozen_by') or '-'}")
        print(f"Notes:     {status.get('notes') or '-'}")
        return

    if args.preflight_freeze:
        hard, soft = run_preflight_freeze(args.dataset)
        print_preflight_result(hard, soft)
        sys.exit(1 if hard else 0)

    timestamp = datetime.now(timezone.utc).isoformat()

    if args.freeze:
        if not args.by:
            print("ERROR: --freeze requires --by \"Name\"", file=sys.stderr)
            sys.exit(1)

        hard, soft = run_preflight_freeze(args.dataset)
        if hard or soft:
            print_preflight_result(hard, soft)
        if hard:
            print("\nRefusing to freeze: fix the BLOCKING item(s) above first.", file=sys.stderr)
            sys.exit(1)
        if soft and not args.acknowledge_warnings:
            print("\nRefusing to freeze: MANUAL REVIEW REQUIRED item(s) above are outstanding.", file=sys.stderr)
            print("Re-run with --acknowledge-warnings once you have actually reviewed them.", file=sys.stderr)
            sys.exit(1)

        status = freeze(args.dataset, frozen_by=args.by, timestamp=timestamp, notes=args.notes, version=args.version)
        print(f"\nDataset FROZEN at {timestamp} by {args.by} (version {status['version']}).")
        print("Split assignments and existing labels should not change from here on --")
        print("see EVALUATION_PROTOCOL.md section 27 and 'holdout labels must remain protected from tuning'.")
        return

    if args.unfreeze:
        if not args.reason:
            print("ERROR: --unfreeze requires --reason \"...\"", file=sys.stderr)
            sys.exit(1)
        unfreeze(args.dataset, reason=args.reason, timestamp=timestamp)
        print(f"Dataset returned to DRAFT at {timestamp}. Reason recorded in dataset_status.json.")


if __name__ == "__main__":
    main()
