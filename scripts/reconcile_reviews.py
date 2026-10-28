#!/usr/bin/env python3
"""
reconcile_reviews.py

Merges evaluation_dataset/reviewer_labels.csv (one row per reviewer per
image, filled in independently -- see LABELING_GUIDE.md) into the
authoritative evaluation_dataset/labels.csv, following
EVALUATION_PROTOCOL.md section 2's exact rule set:

  - One reviewer only for an image_id: their answer is accepted as-is
    (the protocol explicitly allows a single careful labeler at minimum
    scale -- this is not treated as "unresolved").
  - Two+ reviewers AGREE on a field: that value is accepted.
  - Two+ reviewers DISAGREE on a field: labels.csv's cell is left BLANK,
    and every reviewer's answer is recorded in `notes` verbatim -- never
    averaged, never majority-voted, never resolved by silently picking one.
  - A reviewer leaving a field blank (their own uncertainty) does not
    count as a vote either way.

This script never invents a label and never consults AuraMatch's output.
It only merges what humans already wrote in reviewer_labels.csv.

Usage:
    python scripts/reconcile_reviews.py
    python scripts/reconcile_reviews.py --dry-run   # preview without writing labels.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.eval.dataset_stats import read_csv_rows

FIELDS = ["season", "undertone", "face_shape", "lighting", "makeup"]


def group_by_image(reviewer_rows: list) -> dict:
    groups = defaultdict(list)
    for row in reviewer_rows:
        image_id = (row.get("image_id") or "").strip()
        if image_id:
            groups[image_id].append(row)
    return groups


def reconcile_one(image_id: str, reviews: list) -> dict:
    """Returns {field: value_or_None} plus a 'notes' entry describing any
    disagreement, for exactly one image_id.
    """
    result = {f: None for f in FIELDS}
    notes_parts = []

    for field in FIELDS:
        answers = {}  # reviewer -> value, only non-blank answers
        for r in reviews:
            reviewer = r.get("reviewer") or "?"
            value = (r.get(field) or "").strip()
            if value:
                answers[reviewer] = value

        distinct_values = set(answers.values())
        if len(distinct_values) == 0:
            continue  # nobody answered -- stays blank, not a disagreement
        if len(distinct_values) == 1:
            result[field] = next(iter(distinct_values))
        else:
            pairs = ", ".join(f"{rev}={val}" for rev, val in answers.items())
            notes_parts.append(f"{field} disagreement: {pairs}")

    if notes_parts:
        result["notes"] = "; ".join(notes_parts)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parent.parent / "evaluation_dataset")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    reviewer_path = args.dataset / "reviewer_labels.csv"
    labels_path = args.dataset / "labels.csv"

    reviewer_rows = read_csv_rows(reviewer_path)
    if not reviewer_rows:
        print(f"No rows in {reviewer_path} -- nothing to reconcile.")
        return

    existing_labels = {r["image_id"]: r for r in read_csv_rows(labels_path) if r.get("image_id")}
    groups = group_by_image(reviewer_rows)

    updated = 0
    for image_id, reviews in groups.items():
        reconciled = reconcile_one(image_id, reviews)
        row = existing_labels.get(image_id)
        if row is None:
            print(f"  skip {image_id}: not present in labels.csv (assign it an ID first via assign_dataset_ids.py)")
            continue

        changed = False
        for field in FIELDS:
            if reconciled[field] is not None and (row.get(field) or "").strip() != reconciled[field]:
                row[field] = reconciled[field]
                changed = True
        if "notes" in reconciled:
            existing_notes = (row.get("notes") or "").strip()
            if reconciled["notes"] not in existing_notes:
                row["notes"] = (existing_notes + "; " if existing_notes else "") + reconciled["notes"]
                changed = True

        n_reviewers = len({r.get("reviewer") for r in reviews})
        status = "disagreement recorded" if "notes" in reconciled else (
            "single reviewer, accepted" if n_reviewers == 1 else "reviewers agreed"
        )
        print(f"  {image_id}: {status}" + (" (updated)" if changed else " (no change)"))
        updated += 1 if changed else 0

    if args.dry_run:
        print(f"\n--dry-run: would update {updated} row(s) in {labels_path}. Nothing written.")
        return

    fieldnames = ["image_id", "season", "undertone", "face_shape", "lighting", "makeup", "notes"]
    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for image_id in sorted(existing_labels):
            writer.writerow({k: existing_labels[image_id].get(k, "") for k in fieldnames})

    print(f"\nUpdated {updated} row(s) in {labels_path}.")


if __name__ == "__main__":
    main()
