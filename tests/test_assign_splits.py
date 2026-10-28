import csv

import pytest

from scripts.assign_splits import allocate_splits, load_split_csv, main, write_split_csv
from services.eval.dataset_status import freeze


def test_allocate_splits_is_deterministic_given_same_seed():
    ids = [f"AM-EVAL-{i:04d}" for i in range(1, 21)]
    ratios = {"calibration": 0.4, "validation": 0.3, "holdout": 0.3}
    a = allocate_splits(ids, ratios, seed=42)
    b = allocate_splits(ids, ratios, seed=42)
    assert a == b


def test_allocate_splits_different_seeds_can_differ():
    ids = [f"AM-EVAL-{i:04d}" for i in range(1, 21)]
    ratios = {"calibration": 0.4, "validation": 0.3, "holdout": 0.3}
    a = allocate_splits(ids, ratios, seed=1)
    b = allocate_splits(ids, ratios, seed=2)
    assert a != b  # not a hard guarantee in general, but true for this fixture size/ratio


def test_allocate_splits_covers_every_id_exactly_once():
    ids = [f"AM-EVAL-{i:04d}" for i in range(1, 11)]
    result = allocate_splits(ids, {"calibration": 0.4, "validation": 0.3, "holdout": 0.3}, seed=42)
    assert set(result.keys()) == set(ids)
    assert set(result.values()) <= {"calibration", "validation", "holdout"}


def test_allocate_splits_rejects_ratios_not_summing_to_one():
    with pytest.raises(ValueError):
        allocate_splits(["a"], {"calibration": 0.5, "validation": 0.3, "holdout": 0.3}, seed=42)


def test_write_and_load_split_csv_round_trip(tmp_path):
    path = tmp_path / "split.csv"
    write_split_csv(path, {"AM-EVAL-0001": "calibration", "AM-EVAL-0002": "holdout"})
    loaded = load_split_csv(path)
    assert loaded == {"AM-EVAL-0001": "calibration", "AM-EVAL-0002": "holdout"}


def test_load_split_csv_missing_file_returns_empty(tmp_path):
    assert load_split_csv(tmp_path / "nope.csv") == {}


def _write_labels(path, image_ids):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "season", "undertone", "face_shape", "lighting", "makeup", "notes"])
        w.writeheader()
        for i in image_ids:
            w.writerow({"image_id": i, "season": "Spring", "undertone": "", "face_shape": "", "lighting": "", "makeup": "", "notes": ""})


def test_main_never_reassigns_an_existing_split_without_explicit_flag(tmp_path, monkeypatch, capsys):
    labels_path = tmp_path / "labels.csv"
    split_path = tmp_path / "split.csv"
    ids = [f"AM-EVAL-{i:04d}" for i in range(1, 11)]
    _write_labels(labels_path, ids)
    write_split_csv(split_path, {"AM-EVAL-0001": "calibration"})

    monkeypatch.setattr(
        "sys.argv",
        ["assign_splits.py", "--labels", str(labels_path), "--split-file", str(split_path),
         "--dataset", str(tmp_path), "--seed", "42"],
    )
    main()

    result = load_split_csv(split_path)
    assert result["AM-EVAL-0001"] == "calibration"  # untouched
    assert len(result) == len(ids)  # every other id got assigned


def test_main_refuses_to_run_when_dataset_is_frozen(tmp_path, monkeypatch, capsys):
    labels_path = tmp_path / "labels.csv"
    split_path = tmp_path / "split.csv"
    _write_labels(labels_path, ["AM-EVAL-0001"])
    freeze(tmp_path, frozen_by="Reviewer A", timestamp="2026-08-30T00:00:00Z")

    monkeypatch.setattr(
        "sys.argv",
        ["assign_splits.py", "--labels", str(labels_path), "--split-file", str(split_path), "--dataset", str(tmp_path)],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1
    assert not split_path.exists()  # refused before writing anything
