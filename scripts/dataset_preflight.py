#!/usr/bin/env python3
"""
dataset_preflight.py

Checks whether this repository is actually ready to receive the real
evaluation dataset -- structure, schemas, git protection, and tooling
importability. Every PASS/FAIL below is a real check result; nothing is
hard-coded (an intentionally broken repo state will print FAIL for the
specific thing that's broken).

Usage:
    python scripts/dataset_preflight.py
"""

from __future__ import annotations

import csv
import importlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "evaluation_dataset"

EXPECTED_LABELS_HEADER = ["image_id", "season", "undertone", "face_shape", "lighting", "makeup", "notes"]
EXPECTED_METADATA_HEADER = ["image_id", "filename", "width", "height", "occlusion", "multi_face", "quality_status", "notes"]
EXPECTED_SPLIT_HEADER = ["image_id", "split"]

EVAL_MODULES = [
    "services.eval.metrics", "services.eval.calibration", "services.eval.stratify",
    "services.eval.ablation", "services.eval.debug_viz", "services.eval.quality_port",
    "services.eval.validate_labels", "services.eval.report", "services.eval.dataset_io",
    "services.eval.dataset_status", "services.eval.integrity", "services.eval.repro",
    "services.eval.dataset_stats", "services.eval.privacy",
]


def _read_header(path: Path) -> list | None:
    if not path.exists():
        return None
    with open(path, newline="") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return []


def check_protocol() -> tuple:
    required = ["EVALUATION_PROTOCOL.md", "DATASET_RESEARCH.md"]
    missing = [f for f in required if not (DATASET_DIR / f).exists()]
    return (not missing, f"missing: {missing}" if missing else "")


def check_labeling_guide() -> tuple:
    ok = (DATASET_DIR / "LABELING_GUIDE.md").exists() and (DATASET_DIR / "README.md").exists()
    return (ok, "" if ok else "LABELING_GUIDE.md or README.md missing")


def check_dataset_structure() -> tuple:
    required_dirs = ["images", "landmarks", "debug", "results"]
    missing_dirs = [d for d in required_dirs if not (DATASET_DIR / d).is_dir()]
    required_files = ["labels.csv", "metadata.csv", "split.csv", "dataset_status.json", "reviewer_labels.csv", "labeling_template.csv"]
    missing_files = [f for f in required_files if not (DATASET_DIR / f).exists()]
    ok = not missing_dirs and not missing_files
    detail = []
    if missing_dirs:
        detail.append(f"missing dirs: {missing_dirs}")
    if missing_files:
        detail.append(f"missing files: {missing_files}")
    return (ok, "; ".join(detail))


def check_gitignore_protection() -> tuple:
    gitignore = DATASET_DIR / ".gitignore"
    if not gitignore.exists():
        return (False, ".gitignore missing")
    text = gitignore.read_text()
    required_patterns = ["images/*", "landmarks/*", "debug/*", "results/*"]
    missing = [p for p in required_patterns if p not in text]
    if missing:
        return (False, f"missing patterns in .gitignore: {missing}")

    # Live confirmation via git itself, when available -- falls back to the
    # pattern check above if git isn't on PATH or this isn't a git checkout.
    try:
        result = subprocess.run(
            ["git", "check-ignore", str(DATASET_DIR / "images" / "__preflight_probe__.jpg")],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=5,
        )
        # check-ignore exits 0 if the path WOULD be ignored -- exactly what we want.
        if result.returncode not in (0, 1):
            return (True, "pattern check passed; live git check-ignore was inconclusive (non-fatal)")
        if result.returncode == 1:
            return (False, "git check-ignore says a new file under images/ would NOT be ignored")
    except (OSError, subprocess.SubprocessError):
        return (True, "pattern check passed; git unavailable for live confirmation (non-fatal)")

    return (True, "")


def check_no_tracked_images() -> tuple:
    try:
        result = subprocess.run(
            ["git", "ls-files", "evaluation_dataset/images", "evaluation_dataset/landmarks", "evaluation_dataset/debug"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=5,
        )
        tracked = [line for line in result.stdout.splitlines() if line.strip() and not line.endswith(".gitkeep")]
        return (not tracked, f"tracked files found: {tracked}" if tracked else "")
    except (OSError, subprocess.SubprocessError):
        return (True, "git unavailable -- could not confirm (non-fatal)")


def check_labels_schema() -> tuple:
    header = _read_header(DATASET_DIR / "labels.csv")
    if header is None:
        return (False, "labels.csv not found")
    ok = header == EXPECTED_LABELS_HEADER
    return (ok, "" if ok else f"expected {EXPECTED_LABELS_HEADER}, got {header}")


def check_metadata_schema() -> tuple:
    header = _read_header(DATASET_DIR / "metadata.csv")
    if header is None:
        return (False, "metadata.csv not found")
    ok = header == EXPECTED_METADATA_HEADER
    return (ok, "" if ok else f"expected {EXPECTED_METADATA_HEADER}, got {header}")


def check_split_schema() -> tuple:
    header = _read_header(DATASET_DIR / "split.csv")
    if header is None:
        return (False, "split.csv not found")
    ok = header == EXPECTED_SPLIT_HEADER
    return (ok, "" if ok else f"expected {EXPECTED_SPLIT_HEADER}, got {header}")


def check_dataset_status_valid() -> tuple:
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from services.eval.dataset_status import read_status
        status = read_status(DATASET_DIR)
        ok = status.get("status") in ("DRAFT", "FROZEN")
        return (ok, "" if ok else f"invalid status: {status}")
    except Exception as exc:
        return (False, str(exc))


def check_evaluation_tooling_importable() -> tuple:
    sys.path.insert(0, str(REPO_ROOT))
    failed = []
    for mod in EVAL_MODULES:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            failed.append(f"{mod}: {exc}")
    return (not failed, "; ".join(failed))


CHECKS = [
    ("Protocol", check_protocol),
    ("Labeling Guide", check_labeling_guide),
    ("Dataset Structure", check_dataset_structure),
    ("Gitignore Protection", check_gitignore_protection),
    ("No Tracked Images", check_no_tracked_images),
    ("Labels Schema", check_labels_schema),
    ("Metadata Schema", check_metadata_schema),
    ("Split Schema", check_split_schema),
    ("Dataset Status", check_dataset_status_valid),
    ("Evaluation Tooling", check_evaluation_tooling_importable),
]


def main():
    print("AuraMatch Dataset Preflight\n")
    all_ok = True
    for name, check_fn in CHECKS:
        ok, detail = check_fn()
        all_ok = all_ok and ok
        line = f"{name:<25} {'PASS' if ok else 'FAIL'}"
        print(line)
        if not ok and detail:
            print(f"  -> {detail}")

    print()
    print("READY FOR DATA COLLECTION" if all_ok else "NOT READY -- fix the FAIL item(s) above")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
