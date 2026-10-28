"""Validation for evaluation_dataset/labels.csv.

Distinguishes BLOCKING errors (data that would silently corrupt or mislead
an evaluation run -- an invalid enum value, a duplicate image_id that would
otherwise overwrite an earlier row unnoticed) from WARNINGS (worth a human's
attention but not fatal -- a label referencing an image that doesn't exist
yet, which is normal mid-collection).

This module NEVER invents, fixes, or guesses a value -- it only reports.
Blank cells are always valid (a label not yet known); the only thing that's
invalid is a FILLED-IN cell holding something other than a real answer.

Canonical values (case-insensitive on input, canonicalized on output):
    season:     Spring, Summer, Autumn, Winter
    undertone:  Warm, Cool, Neutral
    face_shape: Oval, Round, Square, Heart, Diamond, Triangle
    lighting:   daylight, indoor warm, indoor cool, mixed, backlit, overcast
    makeup:     yes, no

`lighting`'s controlled vocabulary is new as of this validator (the dataset
README previously only gave lighting as free-text examples). It's
deliberately not open-ended: inconsistent free text ("Daylight" vs
"daylight" vs "day light") silently fragments
services/eval/stratify.py's lighting breakdown into meaningless singleton
groups. If a real photo genuinely doesn't fit one of these, leave `lighting`
blank rather than inventing a new category no other row will ever match.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from services.eval.dataset_io import find_image

VALID_SEASONS = {"spring": "Spring", "summer": "Summer", "autumn": "Autumn", "winter": "Winter"}
VALID_UNDERTONES = {"warm": "Warm", "cool": "Cool", "neutral": "Neutral"}
VALID_FACE_SHAPES = {
    "oval": "Oval", "round": "Round", "square": "Square",
    "heart": "Heart", "diamond": "Diamond", "triangle": "Triangle",
}
VALID_LIGHTING = {"daylight", "indoor warm", "indoor cool", "mixed", "backlit", "overcast"}
VALID_MAKEUP = {"yes": "yes", "no": "no"}

REQUIRED_COLUMNS = {"image_id"}
KNOWN_COLUMNS = {"image_id", "season", "undertone", "face_shape", "lighting", "makeup", "notes"}


@dataclass
class ValidationIssue:
    severity: str  # "error" | "warning"
    image_id: str | None
    field: str | None
    message: str

    def __str__(self) -> str:
        tag = "ERROR" if self.severity == "error" else "WARNING"
        where = f"[{self.image_id}]" if self.image_id else ""
        field = f"({self.field}) " if self.field else ""
        return f"{tag} {where} {field}{self.message}".strip()


def validate_labels_csv(path: Path, images_dir: Path | None = None) -> list[ValidationIssue]:
    """Reads `path` directly (independent of any dict-keyed-by-image_id
    loader) so duplicate image_id rows are visible before anything dedupes
    them away. Read-only.
    """
    issues: list[ValidationIssue] = []

    if not Path(path).exists():
        return [ValidationIssue("error", None, None, f"labels.csv not found at {path}")]

    with open(path, newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return [ValidationIssue("error", None, None, "labels.csv is empty (no header row)")]

        header = [h.strip() for h in header]
        missing_required = REQUIRED_COLUMNS - set(header)
        if missing_required:
            issues.append(ValidationIssue(
                "error", None, None,
                f"labels.csv is missing required column(s): {sorted(missing_required)}",
            ))
            return issues

        unknown_columns = set(header) - KNOWN_COLUMNS
        if unknown_columns:
            issues.append(ValidationIssue(
                "warning", None, None,
                f"labels.csv has unrecognized column(s) {sorted(unknown_columns)} -- "
                f"ignored, not an error (maybe a typo, or an intentional extra column)",
            ))

        n_cols = len(header)
        seen_at_line: dict[str, int] = {}

        for line_no, raw_row in enumerate(reader, start=2):  # header is line 1
            if not raw_row or (len(raw_row) == 1 and raw_row[0].strip() == ""):
                continue  # a genuinely blank line -- not a malformed row

            if len(raw_row) != n_cols:
                issues.append(ValidationIssue(
                    "error", None, None,
                    f"line {line_no}: malformed row -- expected {n_cols} columns, got {len(raw_row)} "
                    f"(check for an unescaped comma; quote any field containing one, e.g. `notes`)",
                ))
                continue

            row = dict(zip(header, raw_row))
            image_id = (row.get("image_id") or "").strip()

            if not image_id:
                issues.append(ValidationIssue("warning", None, "image_id", f"line {line_no}: blank image_id -- row skipped"))
                continue

            if image_id in seen_at_line:
                issues.append(ValidationIssue(
                    "error", image_id, "image_id",
                    f"line {line_no}: duplicate image_id (first seen at line {seen_at_line[image_id]}) -- "
                    f"a naive {{image_id: row}} loader would silently keep only one of these",
                ))
            else:
                seen_at_line[image_id] = line_no

            season = (row.get("season") or "").strip()
            if season and season.lower() not in VALID_SEASONS:
                issues.append(ValidationIssue(
                    "error", image_id, "season",
                    f"invalid season {season!r} -- must be one of {sorted(VALID_SEASONS.values())} or blank",
                ))

            undertone = (row.get("undertone") or "").strip()
            if undertone and undertone.lower() not in VALID_UNDERTONES:
                issues.append(ValidationIssue(
                    "error", image_id, "undertone",
                    f"invalid undertone {undertone!r} -- must be one of {sorted(VALID_UNDERTONES.values())} or blank",
                ))

            face_shape = (row.get("face_shape") or "").strip()
            if face_shape and face_shape.lower() not in VALID_FACE_SHAPES:
                issues.append(ValidationIssue(
                    "error", image_id, "face_shape",
                    f"invalid face_shape {face_shape!r} -- must be one of {sorted(VALID_FACE_SHAPES.values())} or blank",
                ))

            lighting = (row.get("lighting") or "").strip()
            if lighting and lighting.lower() not in VALID_LIGHTING:
                issues.append(ValidationIssue(
                    "error", image_id, "lighting",
                    f"invalid lighting {lighting!r} -- must be one of {sorted(VALID_LIGHTING)} or blank",
                ))

            makeup = (row.get("makeup") or "").strip()
            if makeup and makeup.lower() not in VALID_MAKEUP:
                issues.append(ValidationIssue(
                    "error", image_id, "makeup",
                    f"invalid makeup {makeup!r} -- must be one of {sorted(VALID_MAKEUP.values())} or blank",
                ))

            if images_dir is not None and find_image(images_dir, image_id) is None:
                issues.append(ValidationIssue(
                    "warning", image_id, "image_id",
                    f"no image file found in {images_dir} for image_id {image_id!r} "
                    f"(fine mid-collection; otherwise check for a typo)",
                ))

    return issues


def has_blocking_errors(issues: list[ValidationIssue]) -> bool:
    return any(i.severity == "error" for i in issues)


def format_issues(issues: list[ValidationIssue]) -> str:
    if not issues:
        return "labels.csv: no validation issues found."
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    lines = [f"labels.csv validation: {len(errors)} error(s), {len(warnings)} warning(s)"]
    lines += [f"  {i}" for i in issues]
    return "\n".join(lines)
