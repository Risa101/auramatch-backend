#!/usr/bin/env python3
"""
check_dataset_integrity.py

Lightweight dataset curation check (collection-workflow brief, section 16).
Prints exactly what is automated today vs. what still needs a human, using
the SAME criteria and thresholds already defined in
src/utils/imageQuality.js / services/eval/quality_port.py and
EVALUATION_PROTOCOL.md section 5 -- no new thresholds invented here.

Also runs the technical checks from sections 17-18: exact and near-
duplicate image detection (file hash + average-hash, NOT face/identity
recognition -- see services/eval/integrity.py's docstring), and
split-assignment leakage detection.

Usage:
    python scripts/check_dataset_integrity.py \\
        --images evaluation_dataset/images \\
        --split-file evaluation_dataset/split.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.eval.integrity import (
    check_split_leakage, find_exact_duplicates, find_filename_collisions,
    find_near_duplicates,
)

AUTOMATED_CRITERIA = [
    "Resolution (reject < 180px shorter side, warn < 420px)",
    "Blur / Laplacian-variance proxy (reject < 6, warn < 25)",
    "Exposure (reject mean luma < 25 or > 245, warn < 55 or > 220)",
    "Pixel clipping (warn if > 35% of pixels near-black/near-white)",
    "Face size relative to frame (reject < 8% width, warn < 18%)",
    "Head roll / in-plane rotation (reject > 40 deg, warn > 15 deg)",
    "Exact duplicate images (file-content hash)",
    "Near-duplicate images (average-hash, NOT identity matching)",
    "Filename collisions (case/extension ambiguity)",
    "Split-assignment leakage (same image_id in two different splits)",
]

MANUAL_CRITERIA = [
    "Synthetic / AI-generated image (no detector exists)",
    "Identity duplicate -- same real person, different photo (would need face recognition; deliberately not implemented, see section 17)",
    "Occlusion (glasses/hands/hair/mask covering the face)",
    "Multiple faces in frame -- semantic review of WHICH face is the subject (numFaces=1 detection catches this technically, see below, but choosing the right subject is a human call)",
    "Extreme pose beyond roll -- yaw/pitch turning/nodding (only roll is measured today)",
    "Label uncertainty / disagreement resolution (EVALUATION_PROTOCOL.md section 2)",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=Path(__file__).resolve().parent.parent / "evaluation_dataset" / "images")
    ap.add_argument("--split-file", type=Path, default=Path(__file__).resolve().parent.parent / "evaluation_dataset" / "split.csv")
    ap.add_argument("--near-dup-threshold", type=int, default=5, help="max Hamming distance (of 64 bits) to flag as a near-duplicate")
    args = ap.parse_args()

    print("AuraMatch Evaluation Dataset -- Curation Check\n")
    print("Automated:")
    for c in AUTOMATED_CRITERIA:
        print(f"  - {c}")
    print("\nManual review required (no automated check exists for these):")
    for c in MANUAL_CRITERIA:
        print(f"  ! {c}")

    if not args.images.exists() or not any(args.images.iterdir()):
        print(f"\n{args.images} is empty -- nothing to check yet.")
        return

    print("\n--- Results ---\n")

    exact = find_exact_duplicates(args.images)
    if exact:
        print(f"Exact duplicates found ({len(exact)} group(s)):")
        for h, names in exact.items():
            print(f"  {h[:12]}...: {names}")
    else:
        print("Exact duplicates: none found")

    near = find_near_duplicates(args.images, threshold=args.near_dup_threshold)
    if near:
        print(f"\nNear-duplicates found ({len(near)} pair(s), Hamming distance <= {args.near_dup_threshold}/64) -- flagged for manual review, NOT auto-removed:")
        for a, b, d in near:
            print(f"  {a} ~ {b} (distance={d})")
    else:
        print("Near-duplicates: none found")

    collisions = find_filename_collisions(args.images)
    if collisions:
        print(f"\nFilename collisions found ({len(collisions)} group(s)) -- ambiguous for image_id matching:")
        for group in collisions:
            print(f"  {group}")
    else:
        print("Filename collisions: none found")

    if args.split_file.exists():
        with open(args.split_file, newline="") as f:
            split_rows = list(csv.DictReader(f))
        leakage = check_split_leakage(split_rows)
        if leakage:
            print(f"\nSplit leakage issues ({len(leakage)}):")
            for issue in leakage:
                print(f"  {issue}")
        else:
            print("Split leakage: none found")
    else:
        print(f"\nSplit leakage: {args.split_file} does not exist yet -- nothing to check")


if __name__ == "__main__":
    main()
