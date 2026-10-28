#!/usr/bin/env python3
"""
generate_dataset_report.py

Generates evaluation_dataset/dataset_report.md from the ACTUAL current
contents of labels.csv, metadata.csv, landmarks/, split.csv, images/, and
dataset_status.json -- every number in the output is computed at run time.
Nothing is hard-coded; an empty dataset produces a report that says so.

Usage:
    python scripts/generate_dataset_report.py
    python scripts/generate_dataset_report.py --output evaluation_dataset/dataset_report.md
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.eval.dataset_io import IMAGE_EXTS
from services.eval.dataset_stats import (
    FACE_SHAPES, SEASONS, UNDERTONES, disagreement_rows, label_distribution,
    read_csv_rows,
)
from services.eval.dataset_status import read_status
from services.eval.integrity import (
    check_all_landmarks, check_split_leakage, find_exact_duplicates,
    find_filename_collisions, find_near_duplicates,
)


def _md_table(counts: dict, header=("Value", "Count")) -> str:
    lines = [f"| {header[0]} | {header[1]} |", "| --- | ---: |"]
    for k, v in counts.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parent.parent / "evaluation_dataset")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    dataset_dir = args.dataset
    images_dir = dataset_dir / "images"
    landmarks_dir = dataset_dir / "landmarks"
    labels_path = dataset_dir / "labels.csv"
    metadata_path = dataset_dir / "metadata.csv"
    split_path = dataset_dir / "split.csv"
    output_path = args.output or (dataset_dir / "dataset_report.md")

    labels = read_csv_rows(labels_path)
    metadata = read_csv_rows(metadata_path)
    split_rows = read_csv_rows(split_path)
    status = read_status(dataset_dir)

    n_images = len([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]) if images_dir.exists() else 0
    n_labeled_rows = len(labels)

    lines = ["# AuraMatch Evaluation Dataset -- Health Report\n"]
    lines.append(f"Generated from the current on-disk contents of `{dataset_dir}` -- every number below is computed, none are hard-coded.\n")
    lines.append(f"**Dataset status:** {status['status']}" + (f" (frozen {status.get('frozen_at')} by {status.get('frozen_by')})" if status["status"] == "FROZEN" else "") + "\n")

    lines.append("## Dataset Size\n")
    lines.append(f"* Images in `images/`: {n_images}")
    lines.append(f"* Rows in `labels.csv`: {n_labeled_rows}")
    lines.append(f"* Rows in `metadata.csv`: {len(metadata)}")
    lines.append(f"* Rows in `split.csv`: {len(split_rows)}\n")

    if n_labeled_rows == 0:
        lines.append("**No real ground-truth dataset exists yet.** Every distribution below is "
                      "necessarily empty. This is the expected, honest state until real photos are "
                      "collected and labeled per README.md and EVALUATION_PROTOCOL.md.\n")

    season_known, season_unknown = label_distribution(labels, "season", SEASONS)
    lines.append("## Season Distribution\n")
    lines.append(_md_table(season_known, ("Season", "Count")) if n_labeled_rows else "no data yet")
    if season_unknown:
        lines.append(f"\n⚠ Unrecognized season values found (should have been caught by --validate-only): {season_unknown}")
    lines.append("")

    undertone_known, undertone_unknown = label_distribution(labels, "undertone", UNDERTONES)
    lines.append("## Undertone Distribution\n")
    lines.append(_md_table(undertone_known, ("Undertone", "Count")) if n_labeled_rows else "no data yet")
    if undertone_unknown:
        lines.append(f"\n⚠ Unrecognized undertone values found: {undertone_unknown}")
    lines.append("")

    shape_known, shape_unknown = label_distribution(labels, "face_shape", FACE_SHAPES)
    lines.append("## Face Shape Distribution\n")
    lines.append(_md_table(shape_known, ("Face Shape", "Count")) if n_labeled_rows else "no data yet")
    if shape_unknown:
        lines.append(f"\n⚠ Unrecognized face_shape values found: {shape_unknown}")
    lines.append("")

    lighting_counts = Counter((r.get("lighting") or "unknown").strip() or "unknown" for r in labels)
    lines.append("## Lighting Distribution\n")
    lines.append(_md_table(dict(lighting_counts), ("Lighting", "Count")) if n_labeled_rows else "no data yet")
    lines.append("")

    makeup_counts = Counter((r.get("makeup") or "unknown").strip() or "unknown" for r in labels)
    lines.append("## Makeup Distribution\n")
    lines.append(_md_table(dict(makeup_counts), ("Makeup", "Count")) if n_labeled_rows else "no data yet")
    lines.append("")

    quality_counts = Counter((r.get("quality_status") or "unrecorded").strip() or "unrecorded" for r in metadata)
    lines.append("## Quality Distribution\n")
    lines.append(f"(from `metadata.csv`'s `quality_status` column, curator-recorded per EVALUATION_PROTOCOL.md section 5)\n")
    lines.append(_md_table(dict(quality_counts), ("Quality status", "Count")) if metadata else "no data yet")
    lines.append("")

    missing_season = sum(1 for r in labels if not (r.get("season") or "").strip())
    missing_undertone = sum(1 for r in labels if not (r.get("undertone") or "").strip())
    missing_shape = sum(1 for r in labels if not (r.get("face_shape") or "").strip())
    lines.append("## Missing Labels\n")
    lines.append(f"* Rows missing `season`: {missing_season}/{n_labeled_rows}" if n_labeled_rows else "no data yet")
    if n_labeled_rows:
        lines.append(f"* Rows missing `undertone`: {missing_undertone}/{n_labeled_rows}")
        lines.append(f"* Rows missing `face_shape`: {missing_shape}/{n_labeled_rows}")
    lines.append("")

    disagreements = disagreement_rows(labels)
    lines.append("## Disagreements\n")
    lines.append(f"* Rows whose `notes` mention a disagreement (per EVALUATION_PROTOCOL.md section 2's recording convention): {len(disagreements)}" if n_labeled_rows else "no data yet")
    for r in disagreements[:20]:
        lines.append(f"  - {r.get('image_id')}: {r.get('notes')}")
    lines.append("")

    lines.append("## Missing Landmarks\n")
    if n_labeled_rows and landmarks_dir.exists():
        image_ids = [r["image_id"] for r in labels if r.get("image_id")]
        landmark_states = check_all_landmarks(image_ids, landmarks_dir)
        state_counts = Counter(r["state"] for r in landmark_states)
        lines.append(_md_table(dict(state_counts), ("State", "Count")))
    else:
        lines.append("no data yet")
    lines.append("")

    lines.append("## Duplicate Warnings\n")
    if n_images:
        exact = find_exact_duplicates(images_dir)
        near = find_near_duplicates(images_dir)
        collisions = find_filename_collisions(images_dir)
        lines.append(f"* Exact duplicate groups: {len(exact)}")
        lines.append(f"* Near-duplicate pairs (average-hash, manual review only, never auto-removed): {len(near)}")
        lines.append(f"* Filename collisions: {len(collisions)}")
    else:
        lines.append("no data yet")
    lines.append("")

    lines.append("## Split Distribution\n")
    if split_rows:
        split_counts = Counter((r.get("split") or "").strip() for r in split_rows)
        lines.append(_md_table(dict(split_counts), ("Split", "Count")))
        leakage = check_split_leakage(split_rows)
        if leakage:
            lines.append(f"\n⚠ {len(leakage)} split leakage issue(s) found:")
            for issue in leakage:
                lines.append(f"  - {issue}")
    else:
        lines.append("no data yet -- run scripts/assign_splits.py once enough images are labeled")
    lines.append("")

    if n_labeled_rows and season_known:
        min_per_season = min(season_known.values())
        if min_per_season < 15:
            underrepresented = [s for s, c in season_known.items() if c < 15]
            lines.append(f"⚠ **Class balance warning:** {underrepresented} below the 15-20/season floor "
                         f"EVALUATION_PROTOCOL.md section 8 recommends. Do not duplicate images to compensate "
                         f"-- collect more real photos for the underrepresented class(es) instead.\n")

    report = "\n".join(lines)
    output_path.write_text(report)
    print(report)
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
