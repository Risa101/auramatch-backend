"""Heuristic privacy scanning for filenames and CSV text fields.

HONESTY NOTE, stated once here and repeated by every caller: this is a
pattern-matching heuristic, not a guarantee. It cannot know what a "real
name" looks like in general, cannot detect every phone/email format, and
cannot see inside binary image metadata (EXIF) at all -- it only looks at
filenames and the text already sitting in labels.csv/metadata.csv. A clean
scan result means "nothing matched these patterns," not "this data is
definitely anonymous." Treat every finding as a prompt for a human to look,
and treat a clean scan as "no obvious problem found," not proof of safety.
"""

from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(\+?\d[\d\-\s()]{7,}\d)")
_HANDLE_RE = re.compile(r"@[A-Za-z0-9_]{2,}")
# A crude "looks like a personal name" signal: two words joined by an
# UNDERSCORE OR HYPHEN specifically (e.g. "jane_doe", "jane-doe") --
# deliberately NOT space-separated, since two space-separated English words
# describe almost every ordinary sentence ("taken outdoors") and would make
# this heuristic fire on nearly all normal prose. Underscore/hyphen joins
# are a much rarer, more filename-shaped pattern. Still approximate --
# many false positives and false negatives are expected, which is exactly
# why this is a heuristic, not a guarantee.
_NAME_LIKE_RE = re.compile(r"\b([A-Za-z]{2,})[_-]([A-Za-z]{2,})\b")
# Halves that are NOT identifying despite matching the name-like regex --
# the project's own ID scheme and generic camera filename prefixes.
_SAFE_JOIN_HALVES = {"am", "eval", "img", "dsc", "pxl", "photo"}
_SAFE_PATTERNS = (
    re.compile(r"^AM-EVAL-\d+$", re.IGNORECASE),
    re.compile(r"^IMG[-_]?\d+$", re.IGNORECASE),
    re.compile(r"^DSC[-_]?\d+$", re.IGNORECASE),
    re.compile(r"^PXL[-_]?\d+$", re.IGNORECASE),
    re.compile(r"^photo[-_]?\d+$", re.IGNORECASE),
)


def _is_safe_pattern(text: str) -> bool:
    return any(p.match(text.strip()) for p in _SAFE_PATTERNS)


def _has_name_like_pattern(text: str) -> bool:
    for m in _NAME_LIKE_RE.finditer(text):
        left, right = m.group(1).lower(), m.group(2).lower()
        if left in _SAFE_JOIN_HALVES or right in _SAFE_JOIN_HALVES:
            continue
        return True
    return False


def scan_text(text: str) -> list[str]:
    """Returns a list of human-readable finding descriptions, empty if
    nothing matched. Does not identify WHICH substring matched a
    name-like pattern in the returned message for name-like matches
    specifically, to avoid the tool's own output becoming a place that
    echoes identifying text back out -- it says a match was found, not
    what it was, for that one category. Email/phone/handle findings do
    quote the match since redacting them provides little benefit (their
    presence at all is already the finding) while quoting helps a human
    verify quickly.
    """
    if not text or _is_safe_pattern(text):
        return []
    findings = []
    if _EMAIL_RE.search(text):
        findings.append(f"possible email address: {_EMAIL_RE.search(text).group()}")
    if _PHONE_RE.search(text):
        findings.append(f"possible phone number: {_PHONE_RE.search(text).group()}")
    if _HANDLE_RE.search(text):
        findings.append(f"possible social-media handle: {_HANDLE_RE.search(text).group()}")
    if _has_name_like_pattern(text):
        findings.append("possible personal name pattern (two joined words) -- heuristic, verify manually")
    return findings


def scan_filenames(filenames: list) -> dict:
    """Returns {filename: [findings]} for filenames with >= 1 finding."""
    return {f: scan_text(Path_stem(f)) for f in filenames if scan_text(Path_stem(f))}


def Path_stem(filename: str) -> str:
    """Local, dependency-free stem extraction (avoids importing pathlib
    just for this one call site's use across many short strings)."""
    name = filename.rsplit("/", 1)[-1]
    if "." in name:
        name = name.rsplit(".", 1)[0]
    return name


def scan_csv_rows(rows: list, fields: list) -> dict:
    """Returns {(image_id, field): [findings]} for any cell with a finding."""
    results = {}
    for row in rows:
        image_id = row.get("image_id", "?")
        for field in fields:
            value = row.get(field) or ""
            findings = scan_text(value)
            if findings:
                results[(image_id, field)] = findings
    return results
