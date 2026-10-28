"""Reproducibility metadata for an evaluation run (EVALUATION_PROTOCOL.md
section 10, item 12 -- "Reproducibility information," previously a
documented gap with no implementation; this closes it).

Pure functions, no clock/subprocess calls hidden inside anything that
claims to be a data transform -- git/subprocess/timestamp inputs are all
passed in or isolated in a single clearly-labeled "gather" function so
tests can substitute fakes without mocking the world.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path


def git_commit(repo_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir,
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except ImportError:
        return None


def file_hash(path: Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dataset_image_count(images_dir: Path) -> int:
    from services.eval.dataset_io import IMAGE_EXTS
    if not Path(images_dir).exists():
        return 0
    return len([p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS])


def collect(*, repo_dir: Path, dataset_dir: Path, timestamp: str, config: dict) -> dict:
    """Assembles everything the reporting template's "Reproducibility
    information" item asks for. `timestamp` and `config` (the CLI flags
    actually used for this run) are caller-supplied so this stays testable
    without touching the real clock or environment.
    """
    labels_path = Path(dataset_dir) / "labels.csv"
    return {
        "timestamp": timestamp,
        "git_commit": git_commit(repo_dir),
        "python_version": sys.version.split()[0],
        "package_versions": {
            "numpy": package_version("numpy"),
            "opencv-python-headless": package_version("opencv-python-headless"),
            "scikit-learn": package_version("scikit-learn"),
        },
        "dataset_labels_sha256": file_hash(labels_path),
        "dataset_image_count": dataset_image_count(Path(dataset_dir) / "images"),
        "random_seed_note": "K-means skin aggregation uses a fixed random_state=0 "
                             "(services/color_engine/extract.py) -- no other randomness "
                             "exists in the personal-color pipeline.",
        "config": config,
    }


def format_markdown(info: dict) -> str:
    lines = [
        f"* Run timestamp: {info['timestamp']}",
        f"* Code commit: {info['git_commit'] or 'unknown (not a git checkout, or git unavailable)'}",
        f"* Python version: {info['python_version']}",
        "* Package versions: " + ", ".join(
            f"{k}={v or 'unknown'}" for k, v in info["package_versions"].items()
        ),
        f"* Dataset labels.csv SHA-256: {info['dataset_labels_sha256'] or 'labels.csv not found'}",
        f"* Images present at run time: {info['dataset_image_count']}",
        f"* Randomness: {info['random_seed_note']}",
        f"* CLI configuration used: {info['config']}",
    ]
    return "\n".join(lines)
