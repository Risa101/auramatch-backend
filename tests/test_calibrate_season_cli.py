"""End-to-end smoke test for scripts/calibrate_season.py on a synthetic
images/landmarks/labels.csv/split.csv fixture -- proves the calibration
infrastructure actually works, and that holdout enforcement is real (not
just documented), without needing real photos or real calibration data.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from services.color_engine.skin_mask import CHEEK_LEFT, FOREHEAD
from services.eval import calibration_status, dataset_status

REPO_ROOT = Path(__file__).resolve().parent.parent
CANVAS = (200, 200)
NUM_LANDMARKS = 468


def _normalized_landmarks():
    lm = np.full((NUM_LANDMARKS, 3), 0.5)
    h, w = CANVAS

    def circle(cx, cy, r, n):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = (cx + r * np.cos(angles)) / w
        ys = (cy + r * np.sin(angles)) / h
        return np.stack([xs, ys, np.zeros(n)], axis=1)

    lm[CHEEK_LEFT] = circle(60, 120, 30, len(CHEEK_LEFT))
    lm[FOREHEAD] = circle(100, 50, 25, len(FOREHEAD))
    return lm.tolist()


def _write_synthetic_dataset(tmp_path: Path, *, n_per_season=4):
    # n_per_season must be >= len(splits_cycle) below for every split
    # (including "holdout") to actually be assigned at least once.
    images_dir = tmp_path / "images"
    landmarks_dir = tmp_path / "landmarks"
    images_dir.mkdir()
    landmarks_dir.mkdir()

    labels_path = tmp_path / "labels.csv"
    split_path = tmp_path / "split.csv"
    label_rows = [["image_id", "season", "undertone", "face_shape", "lighting", "makeup", "notes"]]
    split_rows = [["image_id", "split"]]

    rng = np.random.default_rng(0)
    seasons = ["Spring", "Summer", "Autumn", "Winter"]
    splits_cycle = ["calibration", "calibration", "validation", "holdout"]
    i = 0
    for season in seasons:
        for j in range(n_per_season):
            image_id = f"{season.lower()}_{i:02d}.png"
            base = np.full((*CANVAS, 3), (120, 150, 200), dtype=np.int16)
            noisy = np.clip(base + rng.integers(-8, 9, size=base.shape), 0, 255).astype(np.uint8)
            ok, buf = cv2.imencode(".png", noisy)
            assert ok
            (images_dir / image_id).write_bytes(buf.tobytes())

            record = {
                "width": CANVAS[1], "height": CANVAS[0], "faceDetected": True, "faceCount": 1,
                "landmarks": _normalized_landmarks(),
                "faceBox": {"x": 0.2, "y": 0.15, "width": 0.3, "height": 0.4}, "rollDeg": 0.0,
            }
            (landmarks_dir / f"{Path(image_id).stem}.landmarks.json").write_text(json.dumps(record))

            label_rows.append([image_id, season, "", "", "daylight", "no", ""])
            split_rows.append([image_id, splits_cycle[j % len(splits_cycle)]])
            i += 1

    with open(labels_path, "w", newline="") as f:
        csv.writer(f).writerows(label_rows)
    with open(split_path, "w", newline="") as f:
        csv.writer(f).writerows(split_rows)

    return images_dir, landmarks_dir, labels_path, split_path


def _run(args, tmp_path, dataset_dir):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "calibrate_season.py"), *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )


def test_baseline_mode_end_to_end(tmp_path):
    images_dir, landmarks_dir, labels_path, split_path = _write_synthetic_dataset(tmp_path)
    out = tmp_path / "baseline.json"
    report = tmp_path / "baseline.md"

    result = _run([
        "--baseline-only", "--images", str(images_dir), "--landmarks", str(landmarks_dir),
        "--labels", str(labels_path), "--split", str(split_path), "--dataset", str(tmp_path),
        "--output", str(out), "--report", str(report),
    ], tmp_path, tmp_path)
    assert result.returncode == 0, result.stderr

    data = json.loads(out.read_text())
    assert data["status"] == "RUN"
    # calibration + validation only -- 2 of 4 per-season rows are "calibration", 1 is "validation", 1 is "holdout"
    # per the 4-cycle in _write_synthetic_dataset -> 3 dev rows per season = 12 total
    assert data["n"] == 12
    assert "balanced_accuracy" in data["metrics"]
    assert report.read_text()  # non-empty


def test_candidates_mode_requires_baseline_first(tmp_path):
    images_dir, landmarks_dir, labels_path, split_path = _write_synthetic_dataset(tmp_path)
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(json.dumps([{"value_pivot": 60.0}]))

    result = _run([
        "--candidates", str(candidates_path), "--baseline", str(tmp_path / "does_not_exist.json"),
        "--images", str(images_dir), "--landmarks", str(landmarks_dir),
        "--labels", str(labels_path), "--split", str(split_path), "--dataset", str(tmp_path),
        "--output", str(tmp_path / "out.json"),
    ], tmp_path, tmp_path)
    assert result.returncode != 0
    assert "baseline" in result.stderr.lower()


def test_candidates_mode_end_to_end_never_touches_holdout(tmp_path):
    images_dir, landmarks_dir, labels_path, split_path = _write_synthetic_dataset(tmp_path)
    baseline_out = tmp_path / "baseline.json"
    _run([
        "--baseline-only", "--images", str(images_dir), "--landmarks", str(landmarks_dir),
        "--labels", str(labels_path), "--split", str(split_path), "--dataset", str(tmp_path),
        "--output", str(baseline_out),
    ], tmp_path, tmp_path)

    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(json.dumps([{"value_pivot": 60.0}, {"chroma_pivot": 22.0}]))
    out = tmp_path / "calibration_results.json"

    result = _run([
        "--candidates", str(candidates_path), "--baseline", str(baseline_out),
        "--images", str(images_dir), "--landmarks", str(landmarks_dir),
        "--labels", str(labels_path), "--split", str(split_path), "--dataset", str(tmp_path),
        "--output", str(out), "--k-fold", "3", "--seed", "1",
    ], tmp_path, tmp_path)
    assert result.returncode == 0, result.stderr

    data = json.loads(out.read_text())
    assert data["status"] == "RUN"
    assert data["n_development_rows"] == 12  # never the 4 holdout rows
    assert len(data["candidates"]) == 2
    for c in data["candidates"]:
        assert "comparison" in c
        assert "REVIEW" in c["comparison"]["recommendation"]
        assert "cross_validation" in c

    # Structural leak check: holdout image_ids/truths must never appear on
    # stdout (or anywhere else) during a --candidates run -- labels are
    # split-filtered before any row is built, so there is nothing to leak,
    # but assert on the actual process output so a regression is caught.
    # j=3 (index 3 of the 4-cycle) is "holdout" for every season -> i=3,7,11,15
    holdout_ids = ["spring_03.png", "summer_07.png", "autumn_11.png", "winter_15.png"]
    for image_id in holdout_ids:
        assert image_id not in result.stdout
    assert "holdout" not in result.stdout.lower()


def test_candidates_mode_rejects_invalid_parameter_without_crashing(tmp_path):
    images_dir, landmarks_dir, labels_path, split_path = _write_synthetic_dataset(tmp_path)
    baseline_out = tmp_path / "baseline.json"
    _run([
        "--baseline-only", "--images", str(images_dir), "--landmarks", str(landmarks_dir),
        "--labels", str(labels_path), "--split", str(split_path), "--dataset", str(tmp_path),
        "--output", str(baseline_out),
    ], tmp_path, tmp_path)

    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(json.dumps([{"not_a_real_param": 1.0}, {"value_pivot": 55.0}]))
    out = tmp_path / "calibration_results.json"

    result = _run([
        "--candidates", str(candidates_path), "--baseline", str(baseline_out),
        "--images", str(images_dir), "--landmarks", str(landmarks_dir),
        "--labels", str(labels_path), "--split", str(split_path), "--dataset", str(tmp_path),
        "--output", str(out),
    ], tmp_path, tmp_path)
    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text())
    assert "error" in data["candidates"][0]  # invalid one reported, not crashed
    assert "metrics" in data["candidates"][1]  # valid one still evaluated


def test_evaluate_holdout_refuses_when_calibration_not_frozen(tmp_path):
    images_dir, landmarks_dir, labels_path, split_path = _write_synthetic_dataset(tmp_path)
    result = _run([
        "--evaluate-holdout", "--i-understand-this-is-final",
        "--images", str(images_dir), "--landmarks", str(landmarks_dir),
        "--labels", str(labels_path), "--split", str(split_path), "--dataset", str(tmp_path),
        "--output", str(tmp_path / "holdout.json"),
    ], tmp_path, tmp_path)
    assert result.returncode != 0
    assert "not FROZEN" in result.stderr
    assert not (tmp_path / "holdout.json").exists()


def test_evaluate_holdout_refuses_without_confirmation_flag(tmp_path):
    images_dir, landmarks_dir, labels_path, split_path = _write_synthetic_dataset(tmp_path)
    result = _run([
        "--evaluate-holdout",
        "--images", str(images_dir), "--landmarks", str(landmarks_dir),
        "--labels", str(labels_path), "--split", str(split_path), "--dataset", str(tmp_path),
        "--output", str(tmp_path / "holdout.json"),
    ], tmp_path, tmp_path)
    assert result.returncode != 0
    assert "i-understand-this-is-final" in result.stderr


PROTECTED_FILES = [
    REPO_ROOT / "services" / "color_engine" / "season.py",
    REPO_ROOT / "services" / "color_engine" / "color_math.py",
]


def _hash_protected_files():
    import hashlib
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in PROTECTED_FILES}


def test_calibration_run_never_modifies_production_files(tmp_path):
    """The one guarantee this whole module exists to provide -- verified
    with a byte-level hash, not just a manual `git diff` at review time.
    """
    before = _hash_protected_files()

    images_dir, landmarks_dir, labels_path, split_path = _write_synthetic_dataset(tmp_path)
    baseline_out = tmp_path / "baseline.json"
    r1 = _run([
        "--baseline-only", "--images", str(images_dir), "--landmarks", str(landmarks_dir),
        "--labels", str(labels_path), "--split", str(split_path), "--dataset", str(tmp_path),
        "--output", str(baseline_out),
    ], tmp_path, tmp_path)
    assert r1.returncode == 0, r1.stderr

    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(json.dumps([{"value_pivot": 60.0}, {"chroma_pivot": 22.0}]))
    r2 = _run([
        "--candidates", str(candidates_path), "--baseline", str(baseline_out),
        "--images", str(images_dir), "--landmarks", str(landmarks_dir),
        "--labels", str(labels_path), "--split", str(split_path), "--dataset", str(tmp_path),
        "--output", str(tmp_path / "calibration_results.json"),
    ], tmp_path, tmp_path)
    assert r2.returncode == 0, r2.stderr

    after = _hash_protected_files()
    assert before == after, "a calibration run modified production color_engine files"


def test_evaluate_holdout_works_after_both_freezes(tmp_path):
    images_dir, landmarks_dir, labels_path, split_path = _write_synthetic_dataset(tmp_path)
    dataset_status.freeze(tmp_path, frozen_by="Tester", timestamp="t1")
    calibration_status.freeze(tmp_path, frozen_by="Tester", timestamp="t2", approved_candidate={"value_pivot": 60.0})

    out = tmp_path / "holdout.json"
    result = _run([
        "--evaluate-holdout", "--i-understand-this-is-final",
        "--images", str(images_dir), "--landmarks", str(landmarks_dir),
        "--labels", str(labels_path), "--split", str(split_path), "--dataset", str(tmp_path),
        "--output", str(out),
    ], tmp_path, tmp_path)
    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text())
    assert data["status"] == "RUN"
    assert data["approved_candidate"] == {"value_pivot": 60.0}
    assert data["n"] == 4  # exactly the holdout rows (1 per season x 4 seasons)
