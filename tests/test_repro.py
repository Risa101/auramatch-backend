from pathlib import Path

from services.eval.repro import collect, dataset_image_count, file_hash, format_markdown


def test_file_hash_none_for_missing_file(tmp_path):
    assert file_hash(tmp_path / "does_not_exist.csv") is None


def test_file_hash_stable_for_existing_file(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("image_id,season\n")
    assert file_hash(p) == file_hash(p)


def test_dataset_image_count_zero_when_dir_missing(tmp_path):
    assert dataset_image_count(tmp_path / "images") == 0


def test_dataset_image_count_counts_only_image_extensions(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "a.jpg").touch()
    (images_dir / "b.png").touch()
    (images_dir / "notes.txt").touch()
    assert dataset_image_count(images_dir) == 2


def test_collect_never_crashes_on_an_empty_dataset(tmp_path):
    info = collect(repo_dir=Path.cwd(), dataset_dir=tmp_path, timestamp="2026-08-30T00:00:00Z", config={"a": 1})
    assert info["timestamp"] == "2026-08-30T00:00:00Z"
    assert info["dataset_labels_sha256"] is None
    assert info["dataset_image_count"] == 0
    assert info["config"] == {"a": 1}


def test_format_markdown_handles_missing_git_commit_gracefully():
    info = {
        "timestamp": "t", "git_commit": None, "python_version": "3.11.0",
        "package_versions": {"numpy": "2.0"}, "dataset_labels_sha256": None,
        "dataset_image_count": 0, "random_seed_note": "note", "config": {},
    }
    md = format_markdown(info)
    assert "unknown" in md
    assert "3.11.0" in md
