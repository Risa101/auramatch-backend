"""End-to-end smoke test for scripts/raw_pipeline_validation.py -- proves
the tooling built for the ROOT-CAUSE experiment actually works on a real
(synthetic) images/landmarks/labels.csv fixture, not just against the
currently-empty evaluation_dataset/. Synthetic images only, matching the
established pattern in tests/test_personal_color_service.py -- no real
face photos.
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


def _write_synthetic_dataset(tmp_path: Path, *, n_per_season=1):
    images_dir = tmp_path / "images"
    landmarks_dir = tmp_path / "landmarks"
    images_dir.mkdir()
    landmarks_dir.mkdir()

    labels_path = tmp_path / "labels.csv"
    rows = [["image_id", "season", "undertone", "face_shape", "lighting", "makeup", "notes"]]

    rng = np.random.default_rng(0)
    seasons = ["Spring", "Summer", "Autumn", "Winter"]
    i = 0
    for season in seasons:
        for _ in range(n_per_season):
            image_id = f"{season.lower()}_{i:02d}.png"
            base = np.full((*CANVAS, 3), (120, 150, 200), dtype=np.int16)
            noisy = np.clip(base + rng.integers(-8, 9, size=base.shape), 0, 255).astype(np.uint8)
            ok, buf = cv2.imencode(".png", noisy)
            assert ok
            (images_dir / image_id).write_bytes(buf.tobytes())

            landmark_record = {
                "width": CANVAS[1], "height": CANVAS[0],
                "faceDetected": True, "faceCount": 1,
                "landmarks": _normalized_landmarks(),
                "faceBox": {"x": 0.2, "y": 0.15, "width": 0.3, "height": 0.4},
                "rollDeg": 0.0,
            }
            stem = Path(image_id).stem
            (landmarks_dir / f"{stem}.landmarks.json").write_text(json.dumps(landmark_record))

            rows.append([image_id, season, "", "", "daylight", "no", ""])
            i += 1

    with open(labels_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    return images_dir, landmarks_dir, labels_path


def test_cli_runs_end_to_end_on_synthetic_dataset(tmp_path):
    images_dir, landmarks_dir, labels_path = _write_synthetic_dataset(tmp_path, n_per_season=1)
    out_csv = tmp_path / "out.csv"
    out_md = tmp_path / "out.md"
    out_json = tmp_path / "out.json"

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "raw_pipeline_validation.py"),
         "--images", str(images_dir), "--landmarks", str(landmarks_dir), "--labels", str(labels_path),
         "--output", str(out_csv), "--report", str(out_md), "--summary-json", str(out_json)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert out_csv.exists() and out_md.exists() and out_json.exists()

    rows = list(csv.DictReader(open(out_csv)))
    assert len(rows) == 4
    assert all(r["valid"] == "True" for r in rows), rows
    assert all(r["predicted_season"] in {"Spring", "Summer", "Autumn", "Winter"} for r in rows)

    summary = json.loads(out_json.read_text())
    assert summary["raw_image_access"] == "AVAILABLE"
    assert summary["accepted"] == 4
    assert summary["rejected"] == 0
    assert summary["production_ai_changed"] is False
    assert summary["final_accuracy"] is None

    report = out_md.read_text()
    assert "RAW IMAGE ACCESS:\nAVAILABLE" in report
    assert "DIAGNOSTIC ABLATION -- NOT PRODUCTION BEHAVIOR" in report


def test_cli_records_rejections_not_silently(tmp_path):
    images_dir, landmarks_dir, labels_path = _write_synthetic_dataset(tmp_path, n_per_season=1)
    # Corrupt one landmark record to force a quality-gate rejection (face
    # too small) -- must still appear as a ROW, not vanish silently.
    bad_record_path = next(landmarks_dir.glob("spring_*.landmarks.json"))
    record = json.loads(bad_record_path.read_text())
    record["faceBox"]["width"] = 0.01  # far below FACE_FRAC_REJECT=0.08
    bad_record_path.write_text(json.dumps(record))

    out_csv = tmp_path / "out.csv"
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "raw_pipeline_validation.py"),
         "--images", str(images_dir), "--landmarks", str(landmarks_dir), "--labels", str(labels_path),
         "--output", str(out_csv), "--report", str(tmp_path / "out.md"), "--summary-json", str(tmp_path / "out.json")],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(open(out_csv)))
    assert len(rows) == 4  # rejected row still present, not dropped
    spring_row = [r for r in rows if r["truth_season"] == "Spring"][0]
    assert spring_row["valid"] == "False"
    assert spring_row["quality_rejection_category"] == "face_size"


def test_cli_sample_per_season_caps_dataset(tmp_path):
    images_dir, landmarks_dir, labels_path = _write_synthetic_dataset(tmp_path, n_per_season=3)
    out_csv = tmp_path / "out.csv"
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "raw_pipeline_validation.py"),
         "--images", str(images_dir), "--landmarks", str(landmarks_dir), "--labels", str(labels_path),
         "--output", str(out_csv), "--report", str(tmp_path / "out.md"), "--summary-json", str(tmp_path / "out.json"),
         "--sample-per-season", "1", "--seed", "1"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(open(out_csv)))
    assert len(rows) == 4  # 1 per season x 4 seasons, not all 12


def test_cli_blocked_on_empty_dataset(tmp_path):
    images_dir = tmp_path / "images"
    landmarks_dir = tmp_path / "landmarks"
    images_dir.mkdir()
    landmarks_dir.mkdir()
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text("image_id,season,undertone,face_shape,lighting,makeup,notes\n")

    out_json = tmp_path / "out.json"
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "raw_pipeline_validation.py"),
         "--images", str(images_dir), "--landmarks", str(landmarks_dir), "--labels", str(labels_path),
         "--output", str(tmp_path / "out.csv"), "--report", str(tmp_path / "out.md"), "--summary-json", str(out_json)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(out_json.read_text())
    assert summary["raw_image_access"] == "BLOCKED"
    assert summary["images_analyzed"] == 0
    assert summary["final_accuracy"] is None
