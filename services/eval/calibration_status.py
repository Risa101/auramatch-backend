"""Calibration freeze/draft status -- the SAME pattern as
services/eval/dataset_status.py (DRAFT/FROZEN manifest, not a filesystem
lock), applied to a SEPARATE concern: whether season.py calibration
PARAMETERS have been reviewed and frozen, distinct from whether the
dataset itself is frozen.

Why a second file instead of reusing dataset_status.py directly:
dataset_status.py hardcodes STATUS_FILENAME = "dataset_status.json" and
answers "is the ground-truth snapshot fixed?" -- a different question from
"have calibration parameters been decided?" A dataset can be FROZEN with
zero calibration having happened yet (the common case today), and
calibration parameters must be FROZEN *before* the holdout split may ever
be read (CALIBRATION_PROTOCOL.md's non-negotiable rule) -- two independent
gates, not one. Rather than parameterizing dataset_status.py's filename for
a feature only this module needs, this file duplicates the small, already-
proven DRAFT/FROZEN pattern verbatim (~90 lines) for calibration_status.json.

DRAFT: calibration parameters are still being searched/reviewed --
holdout access must stay locked. FROZEN: a human has explicitly reviewed
and approved a specific candidate configuration -- holdout may now be
read, exactly once, for the final report (CALIBRATION_PROTOCOL.md section
"Final Holdout Lock").

No Date.now()-equivalent call lives in this module, matching
dataset_status.py's own convention -- freeze()/unfreeze() take an explicit
caller-supplied timestamp.
"""

from __future__ import annotations

import json
from pathlib import Path

STATUS_FILENAME = "calibration_status.json"
VALID_STATUSES = {"DRAFT", "FROZEN"}


def _status_path(dataset_dir) -> Path:
    return Path(dataset_dir) / STATUS_FILENAME


def read_status(dataset_dir) -> dict:
    path = _status_path(dataset_dir)
    if not path.exists():
        return {
            "status": "DRAFT",
            "frozen_at": None,
            "frozen_by": None,
            "approved_candidate": None,
            "notes": "no calibration_status.json found -- treated as DRAFT by default "
                     "(holdout access stays locked until this is explicitly FROZEN)",
        }
    status = json.loads(path.read_text())
    status.setdefault("approved_candidate", None)
    return status


def write_status(dataset_dir, status: dict) -> None:
    if status.get("status") not in VALID_STATUSES:
        raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}, got {status.get('status')!r}")
    _status_path(dataset_dir).write_text(json.dumps(status, indent=2) + "\n")


def freeze(dataset_dir, frozen_by: str, timestamp: str, approved_candidate: dict, notes: str = "") -> dict:
    """Freezes calibration. `approved_candidate` is the EXACT parameter
    override dict a human reviewed and approved (e.g.
    {"value_pivot": 60.0}) -- recorded verbatim so the frozen decision is
    self-documenting, not just a status flag with no record of what was
    actually approved. An empty dict ({}) is valid and means "baseline
    (unchanged DEFAULT_MODEL) was explicitly approved as the calibration
    outcome" -- calibration concluding "don't change anything" is a real,
    legitimate result, not a no-op to skip recording.
    """
    if not frozen_by or not frozen_by.strip():
        raise ValueError("frozen_by is required -- a calibration freeze must record who approved it")
    if not timestamp or not timestamp.strip():
        raise ValueError("timestamp is required (caller-supplied)")
    if approved_candidate is None:
        raise ValueError(
            "approved_candidate is required (pass {} explicitly if the approved outcome is "
            "'no change from baseline' -- that is a valid, real calibration result)"
        )
    status = {
        "status": "FROZEN", "frozen_at": timestamp, "frozen_by": frozen_by,
        "approved_candidate": approved_candidate, "notes": notes,
    }
    write_status(dataset_dir, status)
    return status


def unfreeze(dataset_dir, reason: str, timestamp: str) -> dict:
    if not reason or not reason.strip():
        raise ValueError("reason is required -- un-freezing calibration must be explained")
    current = read_status(dataset_dir)
    note = f"un-frozen at {timestamp}: {reason}"
    if current.get("status") == "FROZEN":
        note += (f" (was approved_candidate={current.get('approved_candidate')!r}, "
                  f"frozen_at={current.get('frozen_at')!r} by {current.get('frozen_by')!r})")
    status = {"status": "DRAFT", "frozen_at": None, "frozen_by": None, "approved_candidate": None, "notes": note}
    write_status(dataset_dir, status)
    return status


def is_frozen(dataset_dir) -> bool:
    return read_status(dataset_dir).get("status") == "FROZEN"
