"""Dataset freeze/draft status.

A lightweight manifest (dataset_status.json), not a filesystem lock --
EVALUATION_PROTOCOL.md section 27 / the collection-workflow brief's section
27 both call for exactly this: "a manifest/version field is sufficient,"
"not irreversible at the filesystem level."

DRAFT: the dataset may still change (images/labels being added, corrected,
or re-reviewed). FROZEN: from this point forward it should be treated as a
fixed ground-truth/holdout snapshot -- evaluate_personal_color.py's
--require-frozen flag refuses to run a real evaluation against a DRAFT
dataset, so a result can't accidentally be reported against labels that
were still being edited.

No Date.now()-equivalent call lives in this module (this repo's own
scripting convention, matching the workflow rule that dynamic scripts must
receive timestamps from their caller, not generate them internally) --
freeze()/unfreeze() take an explicit timestamp so this stays trivially
testable without mocking the clock.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

STATUS_FILENAME = "dataset_status.json"
VALID_STATUSES = {"DRAFT", "FROZEN"}
_VERSION_RE = re.compile(r"^v(\d+)\.(\d+)-frozen$")


def _status_path(dataset_dir) -> Path:
    return Path(dataset_dir) / STATUS_FILENAME


def read_status(dataset_dir) -> dict:
    path = _status_path(dataset_dir)
    if not path.exists():
        return {
            "status": "DRAFT",
            "version": None,
            "frozen_at": None,
            "frozen_by": None,
            "notes": "no dataset_status.json found -- treated as DRAFT by default",
        }
    status = json.loads(path.read_text())
    status.setdefault("version", None)  # tolerate a status file written before versioning existed
    return status


def write_status(dataset_dir, status: dict) -> None:
    if status.get("status") not in VALID_STATUSES:
        raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}, got {status.get('status')!r}")
    _status_path(dataset_dir).write_text(json.dumps(status, indent=2) + "\n")


def _bump_version(current_version) -> str:
    """v1.0-frozen -> v2.0-frozen -> v3.0-frozen ... Falls back to
    v1.0-frozen for anything that isn't already in that exact shape
    (including None, or a hand-edited/unexpected string).
    """
    m = _VERSION_RE.match(current_version or "")
    if m:
        return f"v{int(m.group(1)) + 1}.0-frozen"
    return "v1.0-frozen"


def freeze(dataset_dir, frozen_by: str, timestamp: str, notes: str = "", version: str | None = None) -> dict:
    """Freezes the dataset. `version` is optional -- if omitted, one is
    auto-derived (v1.0-frozen the first time, then bumped each subsequent
    freeze) so a dataset can never be silently re-frozen under the SAME
    version string by accident. Passing the CURRENT version explicitly
    when the dataset is already frozen is rejected outright -- that would
    be exactly the "accidental mutation of a frozen dataset without an
    explicit new version" this exists to prevent; passing any other
    explicit version is accepted as a deliberate choice.
    """
    if not frozen_by or not frozen_by.strip():
        raise ValueError("frozen_by is required -- a freeze must record who froze it")
    if not timestamp or not timestamp.strip():
        raise ValueError("timestamp is required (caller-supplied)")

    current = read_status(dataset_dir)
    if current.get("status") == "FROZEN" and version is not None and version == current.get("version"):
        raise ValueError(
            f"dataset is already frozen at version {version!r} -- re-freezing requires a NEW, "
            f"different version (or omit --version to auto-bump)"
        )
    if version is None:
        version = _bump_version(current.get("version")) if current.get("status") == "FROZEN" else (current.get("version") or "v1.0-frozen")

    status = {"status": "FROZEN", "version": version, "frozen_at": timestamp, "frozen_by": frozen_by, "notes": notes}
    write_status(dataset_dir, status)
    return status


def unfreeze(dataset_dir, reason: str, timestamp: str) -> dict:
    """Reversible by design -- but the previous freeze's details are folded
    into the new notes field so un-freezing leaves a visible trace rather
    than silently erasing the fact that this dataset was ever frozen.
    """
    if not reason or not reason.strip():
        raise ValueError("reason is required -- un-freezing a dataset must be explained")
    current = read_status(dataset_dir)
    note = f"un-frozen at {timestamp}: {reason}"
    if current.get("status") == "FROZEN":
        note += f" (was version={current.get('version')!r}, frozen_at={current.get('frozen_at')!r} by {current.get('frozen_by')!r})"
    status = {"status": "DRAFT", "version": current.get("version"), "frozen_at": None, "frozen_by": None, "notes": note}
    write_status(dataset_dir, status)
    return status


def is_frozen(dataset_dir) -> bool:
    return read_status(dataset_dir).get("status") == "FROZEN"
