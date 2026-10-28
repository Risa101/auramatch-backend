import pytest

from services.eval.dataset_status import freeze, is_frozen, read_status, unfreeze


def test_default_status_is_draft_when_no_file_exists(tmp_path):
    status = read_status(tmp_path)
    assert status["status"] == "DRAFT"
    assert not is_frozen(tmp_path)


def test_freeze_writes_status_and_is_frozen_true(tmp_path):
    freeze(tmp_path, frozen_by="Reviewer A", timestamp="2026-08-30T00:00:00Z", notes="80 images labeled")
    status = read_status(tmp_path)
    assert status["status"] == "FROZEN"
    assert status["frozen_by"] == "Reviewer A"
    assert status["frozen_at"] == "2026-08-30T00:00:00Z"
    assert is_frozen(tmp_path)


def test_freeze_requires_frozen_by(tmp_path):
    with pytest.raises(ValueError):
        freeze(tmp_path, frozen_by="", timestamp="2026-08-30T00:00:00Z")


def test_freeze_requires_timestamp(tmp_path):
    with pytest.raises(ValueError):
        freeze(tmp_path, frozen_by="Reviewer A", timestamp="")


def test_unfreeze_returns_to_draft_and_records_reason(tmp_path):
    freeze(tmp_path, frozen_by="Reviewer A", timestamp="2026-08-30T00:00:00Z")
    unfreeze(tmp_path, reason="found 3 mislabeled rows", timestamp="2026-08-31T00:00:00Z")
    status = read_status(tmp_path)
    assert status["status"] == "DRAFT"
    assert status["frozen_at"] is None
    assert "found 3 mislabeled rows" in status["notes"]
    assert "Reviewer A" in status["notes"]  # trace of the prior freeze is preserved
    assert not is_frozen(tmp_path)


def test_unfreeze_requires_reason(tmp_path):
    freeze(tmp_path, frozen_by="Reviewer A", timestamp="2026-08-30T00:00:00Z")
    with pytest.raises(ValueError):
        unfreeze(tmp_path, reason="", timestamp="2026-08-31T00:00:00Z")


def test_reversible_at_filesystem_level_not_permanent(tmp_path):
    """Protocol section 27: 'not irreversible at the filesystem level.'"""
    freeze(tmp_path, frozen_by="A", timestamp="t1")
    unfreeze(tmp_path, reason="r", timestamp="t2")
    freeze(tmp_path, frozen_by="B", timestamp="t3")
    assert is_frozen(tmp_path)
    assert read_status(tmp_path)["frozen_by"] == "B"
