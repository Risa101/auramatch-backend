"""Shared dataset I/O helpers used by both scripts/evaluate_personal_color.py
and services/eval/validate_labels.py -- kept in one place (services/eval,
the lower-level package) so the CLI script and the validator import FROM
here rather than one importing the other's internals.
"""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def find_image(images_dir: Path, image_id: str) -> Path | None:
    direct = images_dir / image_id
    if direct.exists():
        return direct
    for ext in IMAGE_EXTS:
        p = images_dir / f"{image_id}{ext}"
        if p.exists():
            return p
    return None
