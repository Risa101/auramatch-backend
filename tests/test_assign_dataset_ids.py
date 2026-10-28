import csv

import cv2
import numpy as np

from scripts.assign_dataset_ids import _existing_max_id, _next_id, main


def _write_synthetic_photo(path, color=(120, 130, 140), size=(30, 30)):
    cv2.imwrite(str(path), np.full((*size, 3), color, dtype=np.uint8))


def test_next_id_format():
    assert _next_id(1) == "AM-EVAL-0001"
    assert _next_id(42) == "AM-EVAL-0042"
    assert _next_id(10000) == "AM-EVAL-10000"  # width grows past 4 digits rather than truncating


def test_existing_max_id_zero_when_dir_empty(tmp_path):
    assert _existing_max_id(tmp_path) == 0


def test_existing_max_id_finds_highest_existing(tmp_path):
    (tmp_path / "AM-EVAL-0001.jpg").touch()
    (tmp_path / "AM-EVAL-0007.png").touch()
    (tmp_path / "AM-EVAL-0003.jpg").touch()
    assert _existing_max_id(tmp_path) == 7


def test_existing_max_id_ignores_unrelated_filenames(tmp_path):
    (tmp_path / "random_photo.jpg").touch()
    (tmp_path / "AM-EVAL-0002.jpg").touch()
    assert _existing_max_id(tmp_path) == 2


def test_main_assigns_ids_never_leaks_original_filename(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    _write_synthetic_photo(source / "janes_real_name_selfie.jpg")
    _write_synthetic_photo(source / "another_identifying_filename.png")

    dataset = tmp_path / "evaluation_dataset"
    dataset.mkdir()

    monkeypatch.setattr(
        "sys.argv",
        ["assign_dataset_ids.py", "--input", str(source), "--dataset", str(dataset)],
    )
    main()

    # IDs are assigned in sorted-source-filename order (deterministic, not
    # insertion order): "another_..." < "janes_..." alphabetically.
    images = sorted(p.name for p in (dataset / "images").iterdir())
    assert images == ["AM-EVAL-0001.png", "AM-EVAL-0002.jpg"]
    for name in images:
        assert "janes" not in name and "another" not in name

    with open(dataset / "labels.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["image_id"] for r in rows] == ["AM-EVAL-0001", "AM-EVAL-0002"]
    assert all(r["season"] == "" for r in rows)  # never fabricates a label

    with open(dataset / "metadata.csv", newline="") as f:
        meta_rows = list(csv.DictReader(f))
    assert meta_rows[0]["width"] == "30"
    assert meta_rows[0]["height"] == "30"
    assert meta_rows[0]["occlusion"] == ""  # left blank for human curation


def test_main_continues_numbering_from_existing_images(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    _write_synthetic_photo(source / "new_photo.jpg")

    dataset = tmp_path / "evaluation_dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "images" / "AM-EVAL-0005.jpg").touch()

    monkeypatch.setattr(
        "sys.argv",
        ["assign_dataset_ids.py", "--input", str(source), "--dataset", str(dataset)],
    )
    main()

    images = sorted(p.name for p in (dataset / "images").iterdir())
    assert "AM-EVAL-0006.jpg" in images


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    _write_synthetic_photo(source / "photo.jpg")
    dataset = tmp_path / "evaluation_dataset"

    monkeypatch.setattr(
        "sys.argv",
        ["assign_dataset_ids.py", "--input", str(source), "--dataset", str(dataset), "--dry-run"],
    )
    main()

    assert not (dataset / "images").exists()
    assert not (dataset / "labels.csv").exists()
