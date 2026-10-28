#!/usr/bin/env python3
"""
check_labels.py

Label-quality checker for evaluation_dataset/labels.csv. Reuses
services/eval/validate_labels.py for everything it already covers
(invalid enum values, missing image_id, duplicate image_id, malformed
rows) and adds what that module doesn't: duplicate ROW *content* (two
different image_ids with identical labels across every field -- a
plausible copy-paste error, not necessarily a real problem, so this is a
warning, not a rejection), and a summary of disagreement/uncertain/
unresolved rows using services/eval/dataset_stats.py's shared definitions.

Never rejects a legitimate blank field -- EVALUATION_PROTOCOL.md section 2
explicitly allows (and expects) blank cells for uncertain or unresolved
labels; this script surfaces them as counts, not errors.

Usage:
    python scripts/check_labels.py
    python scripts/check_labels.py --labels evaluation_dataset/labels.csv --images evaluation_dataset/images
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.eval.dataset_stats import (
    disagreement_rows, read_csv_rows, uncertain_rows, unresolved_rows,
)
from services.eval.validate_labels import (
    format_issues, has_blocking_errors, validate_labels_csv,
)

CONTENT_FIELDS = ("season", "undertone", "face_shape", "lighting", "makeup")


def find_duplicate_row_content(rows: list) -> list:
    """Groups rows sharing identical values across every content field
    (ignoring image_id and notes). Only flags a group if at least one
    field in the shared tuple is non-blank -- an all-blank match is just
    two rows nobody has labeled yet, not a meaningful "duplicate."
    """
    groups = defaultdict(list)
    for r in rows:
        key = tuple((r.get(f) or "").strip() for f in CONTENT_FIELDS)
        if any(key):
            groups[key].append(r.get("image_id"))
    return [(key, ids) for key, ids in groups.items() if len(ids) > 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=Path(__file__).resolve().parent.parent / "evaluation_dataset" / "labels.csv")
    ap.add_argument("--images", type=Path, default=Path(__file__).resolve().parent.parent / "evaluation_dataset" / "images")
    args = ap.parse_args()

    issues = validate_labels_csv(args.labels, images_dir=args.images if args.images.exists() else None)
    print(format_issues(issues))

    rows = read_csv_rows(args.labels)
    print(f"\nRows read: {len(rows)}")

    dup_content = find_duplicate_row_content(rows)
    print(f"\nDuplicate row content (same labels across different image_ids -- check for copy-paste, not necessarily wrong): {len(dup_content)}")
    for key, ids in dup_content[:10]:
        labeled = dict(zip(CONTENT_FIELDS, key))
        print(f"  {ids}: {labeled}")

    disagreements = disagreement_rows(rows)
    uncertain = uncertain_rows(rows)
    unresolved = unresolved_rows(rows)
    print(f"\nDisagreement records: {len(disagreements)}")
    for r in disagreements[:10]:
        print(f"  {r.get('image_id')}: {r.get('notes')}")
    print(f"\nUncertain (blank label, reason recorded): {len(uncertain)}")
    print(f"Unresolved (blank label, no reason recorded yet): {len(unresolved)}")

    print(f"\nStatus: {'FAIL' if has_blocking_errors(issues) else 'PASS'}")
    if has_blocking_errors(issues):
        sys.exit(1)


if __name__ == "__main__":
    main()
