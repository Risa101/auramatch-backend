#!/usr/bin/env python3
"""
privacy_check.py

Heuristic scan of image filenames and labels.csv/metadata.csv text fields
for obvious identifying information (name-like patterns, email-like
strings, phone-number-like strings, social-media handles). See
services/eval/privacy.py's module docstring: THIS IS A HEURISTIC WARNING
SYSTEM ONLY. A clean result means "nothing matched these patterns," not
"this dataset is guaranteed anonymous." Filenames assigned by
scripts/assign_dataset_ids.py (AM-EVAL-NNNN) will always pass cleanly by
design -- this check exists to catch a photo that bypassed that tool, or a
`notes` field where a curator/reviewer typed a real name by habit.

Usage:
    python scripts/privacy_check.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.eval.dataset_io import IMAGE_EXTS
from services.eval.dataset_stats import read_csv_rows
from services.eval.privacy import Path_stem, scan_csv_rows, scan_text

TEXT_FIELDS = ["notes"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parent.parent / "evaluation_dataset")
    args = ap.parse_args()

    images_dir = args.dataset / "images"
    filenames = [p.name for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS] if images_dir.exists() else []

    filename_findings = {f: scan_text(Path_stem(f)) for f in filenames if scan_text(Path_stem(f))}

    labels = read_csv_rows(args.dataset / "labels.csv")
    metadata = read_csv_rows(args.dataset / "metadata.csv")
    labels_findings = scan_csv_rows(labels, TEXT_FIELDS)
    metadata_findings = scan_csv_rows(metadata, TEXT_FIELDS + ["occlusion"])

    total_metadata_issues = len(labels_findings) + len(metadata_findings)

    print("Privacy Check\n")
    print(f"Filenames scanned: {len(filenames)}")
    print(f"Potential identifying filenames: {len(filename_findings)}")
    for f, findings in filename_findings.items():
        print(f"  {f}: {findings}")
    print(f"\nPotential metadata issues: {total_metadata_issues}")
    for (image_id, field), findings in labels_findings.items():
        print(f"  labels.csv [{image_id}].{field}: {findings}")
    for (image_id, field), findings in metadata_findings.items():
        print(f"  metadata.csv [{image_id}].{field}: {findings}")

    status = "PASS" if not filename_findings and not total_metadata_issues else "REVIEW RECOMMENDED"
    print(f"\nStatus: {status}")
    print("\nReminder: this is a heuristic scan, not a guarantee of anonymity -- "
          "it cannot inspect image EXIF metadata or verify anything beyond the patterns it looks for.")


if __name__ == "__main__":
    main()
