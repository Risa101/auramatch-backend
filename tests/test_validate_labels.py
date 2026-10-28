"""Tests for services.eval.validate_labels -- synthetic CSV/directory
fixtures only, no real photos needed (per the real-photo evaluation brief's
own rule: dataset-validation tests must not require real images).
"""

from services.eval.validate_labels import (
    format_issues, has_blocking_errors, validate_labels_csv,
)

HEADER = "image_id,season,undertone,face_shape,lighting,makeup,notes\n"


def _write(tmp_path, text, name="labels.csv"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_valid_csv_has_no_issues(tmp_path):
    p = _write(tmp_path, HEADER + "sample_001,Spring,Warm,Oval,daylight,no,\n")
    issues = validate_labels_csv(p)
    assert issues == []
    assert not has_blocking_errors(issues)


def test_missing_optional_labels_is_not_an_error(tmp_path):
    """A row with only image_id filled in -- every other column blank --
    must validate cleanly. Partial labeling is the expected, normal state
    during dataset collection, never a violation.
    """
    p = _write(tmp_path, HEADER + "sample_001,,,,,,\n")
    issues = validate_labels_csv(p)
    assert not has_blocking_errors(issues)


def test_partially_labeled_dataset_is_allowed(tmp_path):
    """Some rows fully labeled, some only partially, some with just an
    image_id -- all valid simultaneously.
    """
    p = _write(tmp_path, HEADER
               + "sample_001,Spring,Warm,Oval,daylight,no,\n"
               + "sample_002,,,Round,,,\n"
               + "sample_003,,,,,,\n")
    issues = validate_labels_csv(p)
    assert not has_blocking_errors(issues)


def test_empty_dataset_header_only_has_no_issues(tmp_path):
    p = _write(tmp_path, HEADER)
    issues = validate_labels_csv(p)
    assert issues == []


def test_invalid_season_is_a_blocking_error(tmp_path):
    p = _write(tmp_path, HEADER + "sample_001,Autmn,,,,,\n")  # typo
    issues = validate_labels_csv(p)
    assert has_blocking_errors(issues)
    assert any(i.field == "season" and i.image_id == "sample_001" for i in issues)


def test_invalid_undertone_is_a_blocking_error(tmp_path):
    p = _write(tmp_path, HEADER + "sample_001,,Warmish,,,,\n")
    issues = validate_labels_csv(p)
    assert has_blocking_errors(issues)
    assert any(i.field == "undertone" for i in issues)


def test_invalid_face_shape_is_a_blocking_error(tmp_path):
    p = _write(tmp_path, HEADER + "sample_001,,,Circle,,,\n")
    issues = validate_labels_csv(p)
    assert has_blocking_errors(issues)
    assert any(i.field == "face_shape" for i in issues)


def test_invalid_lighting_is_a_blocking_error(tmp_path):
    p = _write(tmp_path, HEADER + "sample_001,,,,candlelight,,\n")
    issues = validate_labels_csv(p)
    assert has_blocking_errors(issues)
    assert any(i.field == "lighting" for i in issues)


def test_invalid_makeup_is_a_blocking_error(tmp_path):
    p = _write(tmp_path, HEADER + "sample_001,,,,,heavy,\n")
    issues = validate_labels_csv(p)
    assert has_blocking_errors(issues)
    assert any(i.field == "makeup" for i in issues)


def test_season_is_case_insensitive(tmp_path):
    p = _write(tmp_path, HEADER + "sample_001,spring,,,,,\n")
    issues = validate_labels_csv(p)
    assert not has_blocking_errors(issues)


def test_duplicate_image_id_is_a_blocking_error(tmp_path):
    p = _write(tmp_path, HEADER
               + "sample_001,Spring,,,,,\n"
               + "sample_001,Winter,,,,,\n")
    issues = validate_labels_csv(p)
    assert has_blocking_errors(issues)
    dup = next(i for i in issues if i.field == "image_id" and "duplicate" in i.message)
    assert dup.image_id == "sample_001"
    assert "line 2" in dup.message  # points back at the FIRST occurrence


def test_blank_image_id_row_is_a_warning_not_an_error(tmp_path):
    p = _write(tmp_path, HEADER + ",Spring,,,,,\n")
    issues = validate_labels_csv(p)
    assert not has_blocking_errors(issues)
    assert any(i.severity == "warning" and i.field == "image_id" for i in issues)


def test_nonexistent_image_reference_is_a_warning_not_an_error(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    p = _write(tmp_path, HEADER + "sample_001,Spring,,,,,\n")
    issues = validate_labels_csv(p, images_dir=images_dir)
    assert not has_blocking_errors(issues)
    assert any(i.severity == "warning" and "no image file found" in i.message for i in issues)


def test_existing_image_reference_has_no_warning(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "sample_001.jpg").touch()
    p = _write(tmp_path, HEADER + "sample_001,Spring,,,,,\n")
    issues = validate_labels_csv(p, images_dir=images_dir)
    assert issues == []


def test_malformed_row_wrong_column_count_is_a_blocking_error(tmp_path):
    # Real photo of a bug found last session: an unescaped comma in `notes`
    # silently created an extra field. Now it's caught explicitly instead.
    p = _write(tmp_path, HEADER + "sample_001,Spring,,,,,unquoted, comma, here\n")
    issues = validate_labels_csv(p)
    assert has_blocking_errors(issues)
    assert any("malformed row" in i.message for i in issues)


def test_malformed_row_too_few_columns_is_a_blocking_error(tmp_path):
    p = _write(tmp_path, HEADER + "sample_001,Spring\n")
    issues = validate_labels_csv(p)
    assert has_blocking_errors(issues)


def test_properly_quoted_comma_in_notes_is_not_malformed(tmp_path):
    p = _write(tmp_path, HEADER + 'sample_001,Spring,,,,,"taken outdoors, slightly cloudy"\n')
    issues = validate_labels_csv(p)
    assert not has_blocking_errors(issues)


def test_missing_required_column_is_a_blocking_error(tmp_path):
    p = _write(tmp_path, "season,undertone\nSpring,Warm\n")  # no image_id column at all
    issues = validate_labels_csv(p)
    assert has_blocking_errors(issues)


def test_unrecognized_extra_column_is_a_warning_not_an_error(tmp_path):
    p = _write(tmp_path, "image_id,season,extra_column\nsample_001,Spring,whatever\n")
    issues = validate_labels_csv(p)
    assert not has_blocking_errors(issues)
    assert any(i.severity == "warning" and "unrecognized column" in i.message for i in issues)


def test_missing_file_is_a_blocking_error(tmp_path):
    issues = validate_labels_csv(tmp_path / "does_not_exist.csv")
    assert has_blocking_errors(issues)


def test_format_issues_with_no_issues():
    assert "no validation issues" in format_issues([])
