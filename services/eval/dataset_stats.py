"""Shared CSV-reading and label-distribution helpers, factored out so
scripts/generate_dataset_report.py and scripts/dataset_balance.py compute
distributions the SAME way instead of two scripts silently drifting out of
sync with each other over time.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

SEASONS = ["Spring", "Summer", "Autumn", "Winter"]
UNDERTONES = ["Warm", "Cool", "Neutral"]
FACE_SHAPES = ["Oval", "Round", "Square", "Heart", "Diamond", "Triangle"]


def read_csv_rows(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def label_distribution(rows: list, key: str, known_values: list) -> tuple:
    """Returns (known_counts, unknown_counts) -- unknown_counts should
    always be empty for a labels.csv that's passed --validate-only, but a
    non-empty unknown dict here is worth surfacing rather than hiding
    (e.g. someone edited the file by hand after validation).
    """
    counts = Counter((r.get(key) or "").strip() for r in rows)
    counts.pop("", None)
    known = {v: counts.get(v, 0) for v in known_values}
    unknown = {v: n for v, n in counts.items() if v not in known_values}
    return known, unknown


def disagreement_rows(rows: list) -> list:
    return [r for r in rows if "disagreement" in (r.get("notes") or "").lower()]


def uncertain_rows(rows: list, primary_fields=("season", "undertone", "face_shape")) -> list:
    """Rows with a note but at least one primary field still blank --
    the "uncertain, documented why" state EVALUATION_PROTOCOL.md section 2
    asks for, distinct from a disagreement (which has its own keyword)."""
    return [
        r for r in rows
        if (r.get("notes") or "").strip()
        and "disagreement" not in (r.get("notes") or "").lower()
        and not all((r.get(k) or "").strip() for k in primary_fields)
    ]


def unresolved_rows(rows: list, primary_fields=("season", "undertone", "face_shape")) -> list:
    """Every primary field blank AND no note explaining why -- not
    necessarily a problem (could just be freshly ID-assigned, not yet
    reached by a reviewer), but worth surfacing as a distinct count from
    "uncertain" (which at least has a documented reason)."""
    return [
        r for r in rows
        if not any((r.get(k) or "").strip() for k in primary_fields)
        and not (r.get("notes") or "").strip()
    ]
