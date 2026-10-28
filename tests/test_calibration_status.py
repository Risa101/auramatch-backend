"""Mirrors tests/test_dataset_status.py's pattern for the new, separate
calibration-freeze gate (services/eval/calibration_status.py)."""

import pytest

from services.eval.calibration_status import freeze, is_frozen, read_status, unfreeze


def test_default_status_is_draft_when_no_file_exists(tmp_path):
    status = read_status(tmp_path)
    assert status["status"] == "DRAFT"
    assert not is_frozen(tmp_path)


def test_freeze_writes_status_and_is_frozen_true(tmp_path):
    freeze(tmp_path, frozen_by="Reviewer A", timestamp="2026-08-30T00:00:00Z",
           approved_candidate={"value_pivot": 60.0}, notes="reviewed after dev-set analysis")
    status = read_status(tmp_path)
    assert status["status"] == "FROZEN"
    assert status["frozen_by"] == "Reviewer A"
    assert status["approved_candidate"] == {"value_pivot": 60.0}
    assert is_frozen(tmp_path)


def test_freeze_accepts_empty_dict_as_valid_approved_outcome(tmp_path):
    """Calibration concluding 'don't change anything' is a real, valid
    result -- must be recordable, not rejected as falsy/missing."""
    freeze(tmp_path, frozen_by="Reviewer A", timestamp="t1", approved_candidate={})
    status = read_status(tmp_path)
    assert status["status"] == "FROZEN"
    assert status["approved_candidate"] == {}


def test_freeze_requires_approved_candidate_not_none(tmp_path):
    with pytest.raises(ValueError):
        freeze(tmp_path, frozen_by="Reviewer A", timestamp="t1", approved_candidate=None)


def test_freeze_requires_frozen_by(tmp_path):
    with pytest.raises(ValueError):
        freeze(tmp_path, frozen_by="", timestamp="t1", approved_candidate={})


def test_freeze_requires_timestamp(tmp_path):
    with pytest.raises(ValueError):
        freeze(tmp_path, frozen_by="Reviewer A", timestamp="", approved_candidate={})


def test_unfreeze_returns_to_draft_and_records_reason(tmp_path):
    freeze(tmp_path, frozen_by="Reviewer A", timestamp="t1", approved_candidate={"value_pivot": 60.0})
    unfreeze(tmp_path, reason="found a bug in candidate evaluation", timestamp="t2")
    status = read_status(tmp_path)
    assert status["status"] == "DRAFT"
    assert status["approved_candidate"] is None
    assert "found a bug" in status["notes"]
    assert "Reviewer A" in status["notes"]
    assert not is_frozen(tmp_path)


def test_unfreeze_requires_reason(tmp_path):
    freeze(tmp_path, frozen_by="Reviewer A", timestamp="t1", approved_candidate={})
    with pytest.raises(ValueError):
        unfreeze(tmp_path, reason="", timestamp="t2")


def test_calibration_status_is_independent_file_from_dataset_status(tmp_path):
    """The whole point of this module: a separate gate from
    services/eval/dataset_status.py's dataset_status.json."""
    from services.eval import dataset_status
    dataset_status.freeze(tmp_path, frozen_by="Curator", timestamp="t1")
    assert dataset_status.is_frozen(tmp_path) is True
    assert is_frozen(tmp_path) is False  # calibration status unaffected by dataset freeze
