#!/usr/bin/env python3
"""
dataset_balance.py

Console-focused class-balance view over evaluation_dataset/labels.csv --
a narrower, faster-to-scan companion to scripts/generate_dataset_report.py
(which produces the full markdown health report). Both share the same
underlying distribution logic (services/eval/dataset_stats.py) so they can
never silently disagree with each other.

Purpose: surface collection imbalance BEFORE freezing, per
EVALUATION_PROTOCOL.md section 8's floors -- 15-20 images per season,
60-80 total. Warnings only, never automatic rejection or rebalancing (never
duplicate an image to "fix" a shortfall -- collect a real one instead).

Usage:
    python scripts/dataset_balance.py
    python scripts/dataset_balance.py --labels evaluation_dataset/labels.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.eval.dataset_stats import (
    FACE_SHAPES, SEASONS, UNDERTONES, disagreement_rows, label_distribution,
    read_csv_rows, uncertain_rows, unresolved_rows,
)

MIN_PER_SEASON = 15  # EVALUATION_PROTOCOL.md section 8 -- practical floor, not a statistical guarantee
MIN_TOTAL = 60


def _print_block(title, known: dict, unknown: dict):
    print(f"{title}")
    for k, v in known.items():
        print(f"  {k}: {v}")
    if unknown:
        print(f"  ⚠ unrecognized values found (should have been caught by --validate-only): {unknown}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=Path(__file__).resolve().parent.parent / "evaluation_dataset" / "labels.csv")
    args = ap.parse_args()

    rows = read_csv_rows(args.labels)
    n = len(rows)

    print("AuraMatch Dataset Balance Report\n")
    print(f"Dataset size: {n}\n")

    if n == 0:
        print("No rows in labels.csv -- nothing to report yet.")
        return

    season_known, season_unknown = label_distribution(rows, "season", SEASONS)
    _print_block("Personal Color", season_known, season_unknown)

    undertone_known, undertone_unknown = label_distribution(rows, "undertone", UNDERTONES)
    _print_block("Undertone", undertone_known, undertone_unknown)

    shape_known, shape_unknown = label_distribution(rows, "face_shape", FACE_SHAPES)
    _print_block("Face Shape", shape_known, shape_unknown)

    lighting_known, _ = label_distribution(rows, "lighting", sorted({(r.get("lighting") or "").strip() for r in rows} - {""}))
    print("Lighting")
    for k, v in lighting_known.items():
        print(f"  {k}: {v}")
    print()

    makeup_known, _ = label_distribution(rows, "makeup", sorted({(r.get("makeup") or "").strip() for r in rows} - {""}))
    print("Makeup")
    for k, v in makeup_known.items():
        print(f"  {k}: {v}")
    print()

    unresolved = unresolved_rows(rows)
    uncertain = uncertain_rows(rows)
    disagreements = disagreement_rows(rows)
    print("Unlabeled")
    print(f"  No labels at all, no note explaining why: {len(unresolved)}")
    print(f"  Uncertain (blank label, reason recorded in notes): {len(uncertain)}")
    print()

    print("Disagreements")
    print(f"  {len(disagreements)}")
    for r in disagreements[:10]:
        print(f"    - {r.get('image_id')}: {r.get('notes')}")
    print()

    warnings = []
    labeled_seasons_total = sum(season_known.values())
    if labeled_seasons_total < MIN_TOTAL:
        warnings.append(
            f"Dataset has not reached the protocol's minimum collection floor "
            f"({labeled_seasons_total}/{MIN_TOTAL} season-labeled images -- EVALUATION_PROTOCOL.md section 8)."
        )
    underrepresented = [s for s, c in season_known.items() if 0 < c < MIN_PER_SEASON]
    missing_entirely = [s for s, c in season_known.items() if c == 0]
    if underrepresented or missing_entirely:
        detail = []
        if missing_entirely:
            detail.append(f"no samples at all for {missing_entirely}")
        if underrepresented:
            detail.append(f"below the {MIN_PER_SEASON}/season floor: {underrepresented}")
        warnings.append(
            "Season distribution is imbalanced (" + "; ".join(detail) + "). "
            "Consider collecting additional real samples before freezing -- do not duplicate existing images to compensate."
        )

    if warnings:
        print("WARNING:")
        for w in warnings:
            print(f"  {w}")
    else:
        print("No collection-balance warnings at this dataset size.")


if __name__ == "__main__":
    main()
