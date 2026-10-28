#!/usr/bin/env python3
"""
assign_dataset_ids.py

Assigns anonymous, sequential AM-EVAL-NNNN IDs to newly-collected photos and
copies them into evaluation_dataset/images/ under that ID -- stripping
whatever identifying filename they arrived with (a person's name, a phone's
auto-generated "IMG_1234 (Jane's iPhone)" pattern, etc.) before the photo
ever enters the tracked evaluation workflow.

Also appends a blank-labeled row to labels.csv and a metadata.csv row with
whatever is mechanically measurable (width/height) filled in and everything
else (occlusion, multi_face, quality_status) left blank for a human curator
to fill in per evaluation_dataset/README.md.

THIS SCRIPT NEVER ASSIGNS A LABEL. It only assigns an ID and copies a file.

Usage:
    python scripts/assign_dataset_ids.py --input /path/to/new/photos

Existing evaluation_dataset/images/ files are never touched or renumbered
-- new IDs always continue from the highest existing AM-EVAL-NNNN found, so
re-running this script after adding more source photos is always safe.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from services.eval.dataset_io import IMAGE_EXTS

ID_PREFIX = "AM-EVAL-"
ID_PATTERN = re.compile(rf"^{re.escape(ID_PREFIX)}(\d+)$")
ID_WIDTH = 4  # AM-EVAL-0001


def _existing_max_id(images_dir: Path) -> int:
    highest = 0
    if images_dir.exists():
        for p in images_dir.iterdir():
            m = ID_PATTERN.match(p.stem)
            if m:
                highest = max(highest, int(m.group(1)))
    return highest


def _next_id(n: int) -> str:
    return f"{ID_PREFIX}{n:0{ID_WIDTH}d}"


def _append_csv_row(path: Path, header: list[str], row: dict) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="folder of newly-collected photos (any filenames)")
    ap.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parent.parent / "evaluation_dataset")
    ap.add_argument("--dry-run", action="store_true", help="show what would happen without copying or writing anything")
    args = ap.parse_args()

    images_dir = args.dataset / "images"
    labels_path = args.dataset / "labels.csv"
    metadata_path = args.dataset / "metadata.csv"

    if not args.input.exists():
        print(f"ERROR: --input folder not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    sources = sorted(p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not sources:
        print(f"No images found in {args.input} (looked for {sorted(IMAGE_EXTS)})")
        return

    next_n = _existing_max_id(images_dir) + 1
    labels_header = ["image_id", "season", "undertone", "face_shape", "lighting", "makeup", "notes"]
    metadata_header = ["image_id", "filename", "width", "height", "occlusion", "multi_face", "quality_status", "notes"]

    assigned = []
    for src in sources:
        image_id = _next_id(next_n)
        dest_name = f"{image_id}{src.suffix.lower()}"
        dest = images_dir / dest_name

        img = cv2.imread(str(src))
        width, height = (img.shape[1], img.shape[0]) if img is not None else (None, None)

        print(f"  {src.name} -> {dest_name}" + ("" if img is not None else "  (WARNING: could not decode -- copied anyway, but check this file)"))

        if not args.dry_run:
            images_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            _append_csv_row(labels_path, labels_header, {
                "image_id": image_id, "season": "", "undertone": "", "face_shape": "",
                "lighting": "", "makeup": "", "notes": "",
            })
            _append_csv_row(metadata_path, metadata_header, {
                "image_id": image_id, "filename": dest_name,
                "width": width if width is not None else "", "height": height if height is not None else "",
                "occlusion": "", "multi_face": "", "quality_status": "", "notes": "",
            })

        assigned.append(image_id)
        next_n += 1

    verb = "Would assign" if args.dry_run else "Assigned"
    print(f"\n{verb} {len(assigned)} ID(s): {assigned[0]}..{assigned[-1]}" if assigned else "\nNothing assigned.")
    if not args.dry_run:
        print(f"Blank rows appended to {labels_path} and {metadata_path} -- fill in labels/metadata per README.md and LABELING_GUIDE.md.")


if __name__ == "__main__":
    main()
