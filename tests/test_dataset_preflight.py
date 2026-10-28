import json

from scripts.dataset_preflight import (
    check_dataset_status_valid, check_dataset_structure,
    check_evaluation_tooling_importable, check_labels_schema,
    check_metadata_schema, check_protocol, check_split_schema,
)


def _make_valid_dataset(tmp_path):
    d = tmp_path / "evaluation_dataset"
    for sub in ("images", "landmarks", "debug", "results"):
        (d / sub).mkdir(parents=True)
    (d / "EVALUATION_PROTOCOL.md").write_text("x")
    (d / "DATASET_RESEARCH.md").write_text("x")
    (d / "labels.csv").write_text("image_id,season,undertone,face_shape,lighting,makeup,notes\n")
    (d / "metadata.csv").write_text("image_id,filename,width,height,occlusion,multi_face,quality_status,notes\n")
    (d / "split.csv").write_text("image_id,split\n")
    (d / "dataset_status.json").write_text(json.dumps({"status": "DRAFT", "version": None, "frozen_at": None, "frozen_by": None, "notes": ""}))
    (d / "reviewer_labels.csv").write_text("image_id,reviewer,season,undertone,face_shape,lighting,makeup,notes\n")
    (d / "labeling_template.csv").write_text("image_id,season,undertone,face_shape,lighting,makeup,notes\n")
    return d


def test_check_protocol_passes_when_both_docs_exist(tmp_path):
    d = _make_valid_dataset(tmp_path)
    ok, detail = _run_with_dataset_dir(check_protocol, d)
    assert ok


def _run_with_dataset_dir(fn, dataset_dir):
    """dataset_preflight.py's check functions close over the module-level
    DATASET_DIR constant rather than taking it as a parameter (a CLI-tool
    simplification) -- patch it for the duration of one call so tests can
    point at a synthetic tmp_path instead of the real repo.
    """
    import scripts.dataset_preflight as mod
    original = mod.DATASET_DIR
    mod.DATASET_DIR = dataset_dir
    try:
        return fn()
    finally:
        mod.DATASET_DIR = original


def test_check_protocol_fails_when_missing(tmp_path):
    d = tmp_path / "evaluation_dataset"
    d.mkdir()
    ok, detail = _run_with_dataset_dir(check_protocol, d)
    assert not ok
    assert "missing" in detail


def test_check_dataset_structure_passes_on_valid_dataset(tmp_path):
    d = _make_valid_dataset(tmp_path)
    ok, detail = _run_with_dataset_dir(check_dataset_structure, d)
    assert ok


def test_check_dataset_structure_fails_when_a_dir_is_missing(tmp_path):
    d = _make_valid_dataset(tmp_path)
    import shutil
    shutil.rmtree(d / "debug")
    ok, detail = _run_with_dataset_dir(check_dataset_structure, d)
    assert not ok
    assert "debug" in detail


def test_check_labels_schema_passes_on_correct_header(tmp_path):
    d = _make_valid_dataset(tmp_path)
    ok, detail = _run_with_dataset_dir(check_labels_schema, d)
    assert ok


def test_check_labels_schema_fails_on_wrong_header(tmp_path):
    d = _make_valid_dataset(tmp_path)
    (d / "labels.csv").write_text("id,color\n")
    ok, detail = _run_with_dataset_dir(check_labels_schema, d)
    assert not ok


def test_check_metadata_schema_fails_on_wrong_header(tmp_path):
    d = _make_valid_dataset(tmp_path)
    (d / "metadata.csv").write_text("image_id,foo\n")
    ok, detail = _run_with_dataset_dir(check_metadata_schema, d)
    assert not ok


def test_check_split_schema_passes_on_correct_header(tmp_path):
    d = _make_valid_dataset(tmp_path)
    ok, detail = _run_with_dataset_dir(check_split_schema, d)
    assert ok


def test_check_dataset_status_valid_true_for_draft(tmp_path):
    d = _make_valid_dataset(tmp_path)
    ok, detail = _run_with_dataset_dir(check_dataset_status_valid, d)
    assert ok


def test_evaluation_tooling_importable_is_true_in_this_repo():
    # Uses the real repo's services/eval modules -- if this ever fails, a
    # real import is broken, which is exactly what this check exists to catch.
    ok, detail = check_evaluation_tooling_importable()
    assert ok, detail
