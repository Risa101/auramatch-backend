"""Tests for the extras added to scripts/evaluate_personal_color.py in this
pass: the human-readable validation summary and the --require-frozen gate.
Synthetic fixtures only.
"""

import csv

import pytest

from scripts.evaluate_personal_color import main, print_validation_summary
from services.eval.dataset_status import freeze
from services.eval.validate_labels import ValidationIssue


def _write_labels(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "season", "undertone", "face_shape", "lighting", "makeup", "notes"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_print_validation_summary_reports_uncertain_and_disagreement(tmp_path, capsys):
    labels = {
        "AM-EVAL-0001": {"season": "Spring", "undertone": "", "face_shape": "", "notes": ""},
        "AM-EVAL-0002": {"season": "", "undertone": "", "face_shape": "", "notes": "borderline, unsure"},
        "AM-EVAL-0003": {"season": "", "undertone": "", "face_shape": "", "notes": "season disagreement: R1=Spring, R2=Autumn"},
    }
    print_validation_summary(labels, issues=[], images_dir=tmp_path, landmarks_dir=tmp_path / "landmarks")
    out = capsys.readouterr().out
    assert "Uncertain (notes present, label incomplete): 1" in out
    assert "Disagreements: 1" in out
    assert "Status: PASS" in out


def test_print_validation_summary_status_fail_on_blocking_error(tmp_path, capsys):
    issues = [ValidationIssue("error", "AM-EVAL-0001", "season", "invalid season")]
    print_validation_summary({}, issues=issues, images_dir=tmp_path, landmarks_dir=tmp_path / "landmarks")
    out = capsys.readouterr().out
    assert "Status: FAIL" in out


def test_print_validation_summary_landmarks_not_exported_yet_message(tmp_path, capsys):
    print_validation_summary({}, issues=[], images_dir=tmp_path, landmarks_dir=tmp_path / "does_not_exist")
    out = capsys.readouterr().out
    assert "does not exist yet" in out
    assert "validating BEFORE landmark export" in out


def test_require_frozen_blocks_full_run_on_draft_dataset(tmp_path, monkeypatch):
    dataset = tmp_path / "evaluation_dataset"
    dataset.mkdir()
    labels_path = dataset / "labels.csv"
    images_dir = dataset / "images"
    landmarks_dir = dataset / "landmarks"
    images_dir.mkdir()
    landmarks_dir.mkdir()
    _write_labels(labels_path, [{"image_id": "AM-EVAL-0001", "season": "Spring", "undertone": "", "face_shape": "", "lighting": "", "makeup": "", "notes": ""}])
    # DRAFT by default (no dataset_status.json written)

    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_personal_color.py", "--images", str(images_dir), "--landmarks", str(landmarks_dir),
         "--labels", str(labels_path), "--require-frozen"],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_require_frozen_allows_full_run_when_frozen(tmp_path, monkeypatch):
    dataset = tmp_path / "evaluation_dataset"
    dataset.mkdir()
    labels_path = dataset / "labels.csv"
    images_dir = dataset / "images"
    landmarks_dir = dataset / "landmarks"
    images_dir.mkdir()
    landmarks_dir.mkdir()
    _write_labels(labels_path, [])  # no rows -> the "no labels" early-return path, but proves --require-frozen didn't block it
    freeze(dataset, frozen_by="Reviewer A", timestamp="2026-08-30T00:00:00Z")

    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_personal_color.py", "--images", str(images_dir), "--landmarks", str(landmarks_dir),
         "--labels", str(labels_path), "--require-frozen"],
    )
    main()  # must not raise SystemExit(1) from the freeze gate
