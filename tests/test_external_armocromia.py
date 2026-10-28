"""Unit tests for services/eval/external_armocromia.py.

Synthetic fixtures only -- no real face images or real Deep Armocromia rows
here, per this task's own rule (real external data is used only by the
explicit evaluation CLI, scripts/evaluate_external_armocromia.py, never in
unit tests). All CIELAB numbers below are fabricated for test coverage.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from services.eval.external_armocromia import (
    SEASONS,
    SchemaError,
    build_season_input,
    detect_schema,
    evaluate_all,
    evaluate_sample,
    load_rows,
)

REGION30_HEADER = [
    "L_mean_skin", "a_mean_skin", "b_mean_skin",
    "L_mean_hair", "a_mean_hair", "b_mean_hair",
    "L_mean_eye", "a_mean_eye", "b_mean_eye",
    "season", "season_subtype", "partition", "celeba", "path_rgb_masked",
]

GLOBAL10_HEADER = ["L_mean", "a_mean", "b_mean", "season", "partition", "celeba"]


def _write_csv(path: Path, header: list, rows: list) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def _region30_row(**overrides) -> dict:
    row = {
        "L_mean_skin": "60.0", "a_mean_skin": "20.0", "b_mean_skin": "25.0",
        "L_mean_hair": "20.0", "a_mean_hair": "5.0", "b_mean_hair": "8.0",
        "L_mean_eye": "35.0", "a_mean_eye": "6.0", "b_mean_eye": "10.0",
        "season": "Autumn", "season_subtype": "Autumn_Deep",
        "partition": "train", "celeba": "TRUE", "path_rgb_masked": "RGB-M/train/autunno/deep/1.png",
    }
    row.update(overrides)
    return row


# --- schema detection --------------------------------------------------


def test_detect_schema_region30():
    assert detect_schema(REGION30_HEADER) == "region30"


def test_detect_schema_global10():
    assert detect_schema(GLOBAL10_HEADER) == "global10"


def test_detect_schema_unrecognized_raises():
    with pytest.raises(SchemaError):
        detect_schema(["foo", "bar", "baz"])


def test_load_rows_empty_file_raises(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("", encoding="utf-8")
    with pytest.raises(SchemaError):
        load_rows(p)


# --- loading + label validation ------------------------------------------


def test_load_rows_region30_end_to_end(tmp_path):
    p = _write_csv(tmp_path / "f30.csv", REGION30_HEADER, [
        _region30_row(season="Spring"),
        _region30_row(season="Summer", path_rgb_masked="RGB-M/train/estate/mute/2.png"),
    ])
    schema, samples = load_rows(p)
    assert schema == "region30"
    assert len(samples) == 2
    assert samples[0].source_label == "Spring"
    assert samples[0].skin_lab == (60.0, 20.0, 25.0)
    assert samples[0].hair_lab == (20.0, 5.0, 8.0)
    assert samples[0].exclude_reason is None


def test_unsupported_label_excluded_not_raised(tmp_path):
    p = _write_csv(tmp_path / "f30.csv", REGION30_HEADER, [
        _region30_row(season="Invernale"),  # not one of Spring/Summer/Autumn/Winter
    ])
    schema, samples = load_rows(p)
    assert len(samples) == 1
    assert samples[0].exclude_reason == "unsupported_label('Invernale')"


def test_missing_required_skin_lab_excluded_gracefully(tmp_path):
    # A blank cell is a normal, expected state (matches labels.csv's own
    # blank-cell convention) -- excluded from evaluation with a recorded
    # reason, not a fatal error for the whole run. Genuinely malformed
    # (non-numeric) data is the case that raises -- see the next test.
    p = _write_csv(tmp_path / "f30.csv", REGION30_HEADER, [
        _region30_row(L_mean_skin=""),
    ])
    schema, samples = load_rows(p)
    assert samples[0].exclude_reason == "missing_skin_lab"
    assert samples[0].skin_lab is None


def test_missing_optional_hair_eye_not_excluded(tmp_path):
    p = _write_csv(tmp_path / "f30.csv", REGION30_HEADER, [
        _region30_row(L_mean_hair="", a_mean_hair="", b_mean_hair=""),
    ])
    schema, samples = load_rows(p)
    assert samples[0].exclude_reason is None
    assert samples[0].hair_lab is None
    assert samples[0].skin_lab is not None


def test_malformed_numeric_field_raises(tmp_path):
    p = _write_csv(tmp_path / "f30.csv", REGION30_HEADER, [
        _region30_row(L_mean_skin="not_a_number"),
    ])
    with pytest.raises(SchemaError):
        load_rows(p)


def test_global10_schema_has_no_hair_eye(tmp_path):
    p = _write_csv(tmp_path / "f10.csv", GLOBAL10_HEADER, [
        {"L_mean": "55.0", "a_mean": "15.0", "b_mean": "20.0", "season": "Winter", "partition": "test", "celeba": "FALSE"},
    ])
    schema, samples = load_rows(p)
    assert schema == "global10"
    assert samples[0].hair_lab is None
    assert samples[0].eye_lab is None


# --- build_season_input / evaluate_sample --------------------------------


def test_build_season_input_derives_from_color_math_not_csv_columns():
    from services.eval.external_armocromia import ExternalSample
    sample = ExternalSample(
        row_index=0, sample_id="x", source_label="Autumn", source_subtype=None,
        partition="train", celeba=True, skin_lab=(60.0, 20.0, 25.0),
        hair_lab=None, eye_lab=None,
    )
    features = build_season_input(sample, use_region_contrast=False)
    assert features["valid"] is True
    assert features["skin_lab"] == pytest.approx([60.0, 20.0, 25.0])
    # chroma must come from color_math.skin_features(), not a pre-baked CSV column
    import math
    assert features["chroma"] == pytest.approx(math.hypot(20.0, 25.0), abs=1e-6)
    assert "confidence" not in features
    assert "wb_trustworthy" not in features
    assert features["skin_hair_contrast"] is None
    assert features["skin_iris_contrast"] is None


def test_build_season_input_none_when_skin_lab_missing():
    from services.eval.external_armocromia import ExternalSample
    sample = ExternalSample(
        row_index=0, sample_id="x", source_label="Autumn", source_subtype=None,
        partition="train", celeba=True, skin_lab=None, hair_lab=None, eye_lab=None,
    )
    assert build_season_input(sample, use_region_contrast=False) is None


def test_region_contrast_computed_only_when_requested():
    from services.eval.external_armocromia import ExternalSample
    sample = ExternalSample(
        row_index=0, sample_id="x", source_label="Autumn", source_subtype=None,
        partition="train", celeba=True, skin_lab=(60.0, 20.0, 25.0),
        hair_lab=(20.0, 5.0, 8.0), eye_lab=(35.0, 6.0, 10.0),
    )
    faithful = build_season_input(sample, use_region_contrast=False)
    hypothetical = build_season_input(sample, use_region_contrast=True)
    assert faithful["skin_hair_contrast"] is None
    assert faithful["skin_iris_contrast"] is None
    assert hypothetical["skin_hair_contrast"] is not None
    assert hypothetical["skin_iris_contrast"] is not None
    assert hypothetical["skin_hair_contrast"] > 0


def test_evaluate_sample_excluded_row_never_calls_decide_season():
    from services.eval.external_armocromia import ExternalSample
    sample = ExternalSample(
        row_index=0, sample_id="x", source_label="MISSING", source_subtype=None,
        partition="train", celeba=True, skin_lab=None, hair_lab=None, eye_lab=None,
        exclude_reason="unsupported_label('')",
    )
    result = evaluate_sample(sample)
    assert result.valid is False
    assert result.reason == "unsupported_label('')"
    assert result.predicted is None


def test_evaluate_sample_valid_row_produces_prediction():
    from services.eval.external_armocromia import ExternalSample
    # warm(+a/+b), light L, moderate chroma -> should land near Spring/Autumn territory
    sample = ExternalSample(
        row_index=0, sample_id="x", source_label="Spring", source_subtype=None,
        partition="test", celeba=False, skin_lab=(65.0, 18.0, 22.0),
        hair_lab=None, eye_lab=None,
    )
    result = evaluate_sample(sample)
    assert result.valid is True
    assert result.predicted in SEASONS
    assert 0.0 <= result.top_confidence <= 1.0
    assert result.margin is not None and result.margin >= 0.0
    # reliability is expected to be constant "low" -- see module docstring;
    # this pins that documented, verified behavior so a future season.py
    # change to the defaulting logic is caught, not silently drifted past.
    assert result.reliability == "low"


# --- determinism ---------------------------------------------------------


def test_deterministic_same_input_same_output():
    from services.eval.external_armocromia import ExternalSample
    sample = ExternalSample(
        row_index=0, sample_id="x", source_label="Winter", source_subtype=None,
        partition="test", celeba=False, skin_lab=(40.0, 2.0, 4.0),
        hair_lab=(15.0, 1.0, 2.0), eye_lab=(30.0, 1.0, 3.0),
    )
    r1 = evaluate_sample(sample, use_region_contrast=True)
    r2 = evaluate_sample(sample, use_region_contrast=True)
    assert r1.predicted == r2.predicted
    assert r1.top_confidence == r2.top_confidence
    assert r1.margin == r2.margin
    assert r1.reliability == r2.reliability


def test_evaluate_all_is_order_preserving_and_deterministic(tmp_path):
    p = _write_csv(tmp_path / "f30.csv", REGION30_HEADER, [
        _region30_row(season="Spring", path_rgb_masked="a.png"),
        _region30_row(season="Winter", L_mean_skin="40.0", a_mean_skin="2.0", b_mean_skin="4.0", path_rgb_masked="b.png"),
    ])
    _, samples = load_rows(p)
    run1 = evaluate_all(samples)
    run2 = evaluate_all(samples)
    assert [r.predicted for r in run1] == [r.predicted for r in run2]
    assert [r.sample_id for r in run1] == ["a.png", "b.png"]


# --- output generation (CLI-level) ---------------------------------------


def test_cli_end_to_end_produces_output_files(tmp_path):
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    csv_path = _write_csv(tmp_path / "features_30_per_region.csv", REGION30_HEADER, [
        _region30_row(season="Spring", path_rgb_masked="a.png"),
        _region30_row(season="Summer", L_mean_skin="55.0", a_mean_skin="5.0", b_mean_skin="10.0", path_rgb_masked="b.png"),
        _region30_row(season="Autumn", L_mean_skin="45.0", a_mean_skin="18.0", b_mean_skin="20.0", path_rgb_masked="c.png"),
        _region30_row(season="Winter", L_mean_skin="35.0", a_mean_skin="2.0", b_mean_skin="3.0", path_rgb_masked="d.png"),
    ])
    out_csv = tmp_path / "out.csv"
    out_md = tmp_path / "out.md"
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "evaluate_external_armocromia.py"),
         "--input", str(csv_path), "--output", str(out_csv), "--report", str(out_md)],
        cwd=repo_root, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert out_csv.exists()
    assert out_md.exists()
    assert "EXPLORATORY ONLY" in out_md.read_text(encoding="utf-8")
    assert "STATUS" in out_md.read_text(encoding="utf-8")


def test_cli_contrast_region_only_writes_nonempty_csv(tmp_path):
    # Regression test: --contrast region alone must not write a header-only
    # CSV while the report shows full results for that configuration.
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    csv_path = _write_csv(tmp_path / "features_30_per_region.csv", REGION30_HEADER, [
        _region30_row(season="Spring", path_rgb_masked="a.png"),
        _region30_row(season="Summer", L_mean_skin="55.0", a_mean_skin="5.0", b_mean_skin="10.0", path_rgb_masked="b.png"),
    ])
    out_csv = tmp_path / "out.csv"
    out_md = tmp_path / "out.md"
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "evaluate_external_armocromia.py"),
         "--input", str(csv_path), "--output", str(out_csv), "--report", str(out_md),
         "--contrast", "region"],
        cwd=repo_root, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(open(out_csv)))
    assert len(rows) == 2, f"expected 2 data rows, got {len(rows)}: {rows}"
    assert all(r["used_region_contrast"] == "True" for r in rows)


def test_cli_fails_safely_on_bad_schema(tmp_path):
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("foo,bar\n1,2\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "evaluate_external_armocromia.py"),
         "--input", str(bad_csv), "--output", str(tmp_path / "out.csv"), "--report", str(tmp_path / "out.md")],
        cwd=repo_root, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert "FATAL" in result.stderr
    assert not (tmp_path / "out.csv").exists()
