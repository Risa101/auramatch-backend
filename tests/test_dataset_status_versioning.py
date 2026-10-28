import pytest

from services.eval.dataset_status import _bump_version, freeze, read_status


def test_bump_version_from_none():
    assert _bump_version(None) == "v1.0-frozen"


def test_bump_version_increments_major():
    assert _bump_version("v1.0-frozen") == "v2.0-frozen"
    assert _bump_version("v9.0-frozen") == "v10.0-frozen"


def test_bump_version_falls_back_on_unexpected_string():
    assert _bump_version("not-a-version") == "v1.0-frozen"


def test_first_freeze_defaults_to_v1(tmp_path):
    status = freeze(tmp_path, frozen_by="A", timestamp="t1")
    assert status["version"] == "v1.0-frozen"


def test_second_freeze_auto_bumps_when_version_omitted(tmp_path):
    freeze(tmp_path, frozen_by="A", timestamp="t1")
    status = freeze(tmp_path, frozen_by="A", timestamp="t2", notes="corrected 3 rows")
    assert status["version"] == "v2.0-frozen"


def test_refreezing_with_the_same_explicit_version_is_rejected(tmp_path):
    freeze(tmp_path, frozen_by="A", timestamp="t1", version="v1.0-frozen")
    with pytest.raises(ValueError):
        freeze(tmp_path, frozen_by="A", timestamp="t2", version="v1.0-frozen")


def test_refreezing_with_a_new_explicit_version_is_allowed(tmp_path):
    freeze(tmp_path, frozen_by="A", timestamp="t1", version="v1.0-frozen")
    status = freeze(tmp_path, frozen_by="A", timestamp="t2", version="v5.0-frozen")
    assert status["version"] == "v5.0-frozen"


def test_read_status_tolerates_a_file_written_before_versioning_existed(tmp_path):
    import json
    (tmp_path / "dataset_status.json").write_text(json.dumps({
        "status": "FROZEN", "frozen_at": "t1", "frozen_by": "A", "notes": "",
    }))
    status = read_status(tmp_path)
    assert status["version"] is None  # backfilled, doesn't crash
