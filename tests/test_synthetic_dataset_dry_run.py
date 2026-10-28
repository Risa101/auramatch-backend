"""
SYNTHETIC TOOLING FIXTURE -- NOT REAL ACCURACY DATA.

End-to-end integration test exercising the full dataset-collection
tooling chain (curation -> labeling -> reconciliation -> validation ->
splits -> preflight-freeze -> evaluation) against a fabricated, clearly-
synthetic dataset built entirely inside pytest's tmp_path. Nothing here is
a real photo, nothing here is written into evaluation_dataset/, and no
number this test produces is ever printed as, or could be mistaken for, a
real AuraMatch accuracy figure -- it only proves the PLUMBING works.

Per the collection-workflow brief's own instruction (section 15): this
fixture verifies tooling only. It is not a substitute for real labeled
photos, and it must never be reported as one.
"""

from __future__ import annotations

import csv
import json

import cv2
import numpy as np
import pytest

from scripts.assign_splits import allocate_splits, write_split_csv
from scripts.dataset_status import run_preflight_freeze
from scripts.reconcile_reviews import main as reconcile_main
from services.eval.dataset_stats import (
    SEASONS, disagreement_rows, label_distribution, read_csv_rows,
)
from services.eval.dataset_status import freeze, is_frozen
from services.eval.integrity import (
    check_all_landmarks, check_split_leakage, find_exact_duplicates,
)
from services.eval.validate_labels import has_blocking_errors, validate_labels_csv
from services.personal_color_service import analyze_personal_color
from services.color_engine.skin_mask import CHEEK_LEFT, FOREHEAD

N_PER_SEASON = 6  # 24 total -- "20+ images" per the brief, small enough to run fast
LIGHTINGS = ["daylight", "indoor warm", "indoor cool", "mixed"]


def _fake_landmarks_norm():
    """A minimal but structurally valid 468-point landmark set (normalized
    [0,1]) with real cheek/forehead hulls, matching what
    services/personal_color_service.py actually expects -- same technique
    already used in tests/test_personal_color_service.py.
    """
    lm = [[0.5, 0.5, 0.0] for _ in range(468)]

    def circle(cx, cy, r, n):
        return [[cx + r * np.cos(2 * np.pi * i / n), cy + r * np.sin(2 * np.pi * i / n), 0.0] for i in range(n)]

    for idx, pt in zip(CHEEK_LEFT, circle(0.3, 0.6, 0.1, len(CHEEK_LEFT))):
        lm[idx] = pt
    for idx, pt in zip(FOREHEAD, circle(0.5, 0.25, 0.12, len(FOREHEAD))):
        lm[idx] = pt
    return lm


def _write_synthetic_photo(path, bgr_color):
    cv2.imwrite(str(path), np.full((120, 120, 3), bgr_color, dtype=np.uint8))


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Builds the full fixture described in the brief: 24+ images across 4
    seasons, multiple lighting values, some blank labels, one disagreement
    (via reviewer_labels.csv), one exact duplicate pair, and split
    assignments. Returns the dataset_dir Path.
    """
    dataset_dir = tmp_path / "evaluation_dataset"
    for sub in ("images", "landmarks", "debug", "results"):
        (dataset_dir / sub).mkdir(parents=True)
    (dataset_dir / "EVALUATION_PROTOCOL.md").write_text("SYNTHETIC TOOLING FIXTURE placeholder protocol doc.\n")

    # A distinct BGR color per season, just so images aren't byte-identical
    # by accident (except the one deliberate duplicate below).
    season_colors = {
        "Spring": (140, 180, 230), "Summer": (200, 160, 150),
        "Autumn": (60, 110, 160), "Winter": (120, 90, 90),
    }

    labels_rows = []
    reviewer_rows = []
    n = 0

    for season in SEASONS:
        for i in range(N_PER_SEASON):
            n += 1
            image_id = f"AM-EVAL-{n:04d}"
            color = season_colors[season]
            _write_synthetic_photo(dataset_dir / "images" / f"{image_id}.png", color)
            (dataset_dir / "landmarks" / f"{image_id}.landmarks.json").write_text(json.dumps({
                "faceDetected": True, "faceCount": 1, "width": 120, "height": 120,
                "landmarks": _fake_landmarks_norm(), "faceBox": {"x": 0.2, "y": 0.2, "width": 0.5, "height": 0.5},
                "rollDeg": 1.0,
            }))
            lighting = LIGHTINGS[i % len(LIGHTINGS)]

            if i == 0:
                # Blank label -- genuinely unlabeled yet, allowed by protocol.
                labels_rows.append({"image_id": image_id, "season": "", "undertone": "", "face_shape": "",
                                     "lighting": lighting, "makeup": "no", "notes": ""})
            elif i == 1 and season == "Spring":
                # A deliberate disagreement, resolved via reconcile_reviews.py below.
                labels_rows.append({"image_id": image_id, "season": "", "undertone": "", "face_shape": "",
                                     "lighting": lighting, "makeup": "no", "notes": ""})
                reviewer_rows.append({"image_id": image_id, "reviewer": "R1", "season": "Spring", "undertone": "Warm",
                                       "face_shape": "Oval", "lighting": "", "makeup": "", "notes": ""})
                reviewer_rows.append({"image_id": image_id, "reviewer": "R2", "season": "Autumn", "undertone": "Warm",
                                       "face_shape": "Oval", "lighting": "", "makeup": "", "notes": ""})
            else:
                labels_rows.append({"image_id": image_id, "season": season, "undertone": "Neutral", "face_shape": "Oval",
                                     "lighting": lighting, "makeup": "no" if i % 2 else "yes", "notes": ""})

    # One deliberate exact duplicate (same pixel content as AM-EVAL-0001) for the integrity check to find.
    _write_synthetic_photo(dataset_dir / "images" / "AM-EVAL-DUPLICATE.png", season_colors["Spring"])
    labels_rows.append({"image_id": "AM-EVAL-DUPLICATE", "season": "", "undertone": "", "face_shape": "",
                         "lighting": "", "makeup": "", "notes": ""})
    (dataset_dir / "landmarks" / "AM-EVAL-DUPLICATE.landmarks.json").write_text(json.dumps({
        "faceDetected": False, "faceCount": 0, "width": 120, "height": 120, "landmarks": None, "faceBox": None, "rollDeg": None,
    }))

    fieldnames = ["image_id", "season", "undertone", "face_shape", "lighting", "makeup", "notes"]
    with open(dataset_dir / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(labels_rows)

    reviewer_fieldnames = ["image_id", "reviewer", "season", "undertone", "face_shape", "lighting", "makeup", "notes"]
    with open(dataset_dir / "reviewer_labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=reviewer_fieldnames)
        w.writeheader()
        w.writerows(reviewer_rows)

    return dataset_dir


def test_fixture_is_never_mistaken_for_real_data(synthetic_dataset):
    """Sanity check on the fixture itself, not the tooling: this data must
    be trivially distinguishable from anything real.
    """
    assert "SYNTHETIC TOOLING FIXTURE" in (synthetic_dataset / "EVALUATION_PROTOCOL.md").read_text()


def test_reconcile_reviews_resolves_the_planted_disagreement(synthetic_dataset, monkeypatch):
    monkeypatch.setattr("sys.argv", ["reconcile_reviews.py", "--dataset", str(synthetic_dataset)])
    reconcile_main()

    labels = read_csv_rows(synthetic_dataset / "labels.csv")
    disagreements = disagreement_rows(labels)
    assert len(disagreements) == 1
    assert "R1=Spring" in disagreements[0]["notes"] or "R2=Autumn" in disagreements[0]["notes"]
    # the disagreeing field (season) stayed blank; the agreeing field (undertone) got filled in
    row = disagreements[0]
    assert row["season"] == ""
    assert row["undertone"] == "Warm"


def test_integrity_checks_find_the_planted_duplicate(synthetic_dataset):
    dups = find_exact_duplicates(synthetic_dataset / "images")
    all_names = {n for names in dups.values() for n in names}
    assert "AM-EVAL-0001.png" in all_names
    assert "AM-EVAL-DUPLICATE.png" in all_names


def test_validate_labels_finds_no_blocking_errors_on_well_formed_fixture(synthetic_dataset):
    issues = validate_labels_csv(synthetic_dataset / "labels.csv", images_dir=synthetic_dataset / "images")
    assert not has_blocking_errors(issues)


def test_landmark_states_reflect_the_planted_no_face_case(synthetic_dataset):
    labels = read_csv_rows(synthetic_dataset / "labels.csv")
    ids = [r["image_id"] for r in labels]
    states = {r["image_id"]: r["state"] for r in check_all_landmarks(ids, synthetic_dataset / "landmarks")}
    assert states["AM-EVAL-DUPLICATE"] == "NO_FACE"
    assert states["AM-EVAL-0002"] == "SUCCESS"  # a normally-labeled row


def test_season_distribution_matches_what_was_planted(synthetic_dataset):
    labels = read_csv_rows(synthetic_dataset / "labels.csv")
    known, unknown = label_distribution(labels, "season", SEASONS)
    assert unknown == {}
    # N_PER_SEASON=6 per season, minus 2 blanked out (index 0 and the Spring disagreement)
    assert known["Spring"] == N_PER_SEASON - 2
    assert known["Summer"] == N_PER_SEASON - 1
    assert sum(known.values()) > 0


def test_split_assignment_is_deterministic_and_leak_free(synthetic_dataset):
    labels = read_csv_rows(synthetic_dataset / "labels.csv")
    ids = [r["image_id"] for r in labels]
    assignment = allocate_splits(ids, {"calibration": 0.5, "validation": 0.25, "holdout": 0.25}, seed=42)
    write_split_csv(synthetic_dataset / "split.csv", assignment)

    split_rows = read_csv_rows(synthetic_dataset / "split.csv")
    assert check_split_leakage(split_rows) == []
    assert len(split_rows) == len(ids)

    # determinism: re-running with the same seed reproduces the same assignment
    again = allocate_splits(ids, {"calibration": 0.5, "validation": 0.25, "holdout": 0.25}, seed=42)
    assert again == assignment


def test_preflight_freeze_reports_missing_split_before_splits_assigned(synthetic_dataset):
    hard, soft = run_preflight_freeze(synthetic_dataset)
    assert any("split.csv" in h for h in hard)


def test_preflight_freeze_passes_hard_checks_once_everything_is_in_place(synthetic_dataset, monkeypatch):
    monkeypatch.setattr("sys.argv", ["reconcile_reviews.py", "--dataset", str(synthetic_dataset)])
    reconcile_main()

    labels = read_csv_rows(synthetic_dataset / "labels.csv")
    ids = [r["image_id"] for r in labels]
    assignment = allocate_splits(ids, {"calibration": 0.5, "validation": 0.25, "holdout": 0.25}, seed=42)
    write_split_csv(synthetic_dataset / "split.csv", assignment)

    hard, soft = run_preflight_freeze(synthetic_dataset)
    assert hard == []  # no structural blockers left
    # soft warnings (dataset below the 60-image floor, no version yet) are
    # expected and fine -- this fixture is deliberately small.
    assert any("season-labeled" in s for s in soft)


def test_freeze_then_require_frozen_gate_behaves_correctly(synthetic_dataset):
    assert not is_frozen(synthetic_dataset)
    freeze(synthetic_dataset, frozen_by="Synthetic Test Reviewer", timestamp="2026-08-30T00:00:00Z",
           notes="SYNTHETIC TOOLING FIXTURE -- not a real freeze")
    assert is_frozen(synthetic_dataset)


def test_full_personal_color_pipeline_runs_against_every_synthetic_image(synthetic_dataset):
    """The actual analysis pipeline (services/personal_color_service.py,
    UNMODIFIED by this task) runs successfully against every fake image in
    the fixture -- proving the tooling's plumbing works end to end. This is
    explicitly NOT an accuracy check: these are solid-color synthetic
    images with fabricated landmarks, not real faces, so agreement or
    disagreement with the planted "season" label means nothing about
    real-world accuracy.
    """
    labels = read_csv_rows(synthetic_dataset / "labels.csv")
    lm = _fake_landmarks_norm()
    n_valid = 0
    for row in labels:
        image_path = synthetic_dataset / "images" / f"{row['image_id']}.png"
        landmark_path = synthetic_dataset / "landmarks" / f"{row['image_id']}.landmarks.json"
        landmark_record = json.loads(landmark_path.read_text())
        if not landmark_record["faceDetected"]:
            continue
        result = analyze_personal_color(image_path.read_bytes(), lm)
        assert "valid" in result  # pipeline ran to completion without raising
        if result["valid"]:
            n_valid += 1
    assert n_valid > 0  # at least the well-formed synthetic images produced a real (if meaningless) result
