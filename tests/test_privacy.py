from services.eval.privacy import Path_stem, scan_csv_rows, scan_text


def test_safe_id_pattern_never_flagged():
    assert scan_text("AM-EVAL-0001") == []
    assert scan_text("am-eval-0042") == []


def test_generic_camera_filenames_never_flagged():
    assert scan_text("IMG_1234") == []
    assert scan_text("DSC_5678") == []
    assert scan_text("PXL_9999") == []


def test_email_is_flagged():
    findings = scan_text("contact jane.doe@example.com about this")
    assert any("email" in f for f in findings)


def test_phone_number_is_flagged():
    findings = scan_text("call 555-123-4567 for details")
    assert any("phone" in f for f in findings)


def test_social_handle_is_flagged():
    findings = scan_text("posted by @realjanedoe last week")
    assert any("handle" in f for f in findings)


def test_name_like_pattern_is_flagged_without_echoing_the_name():
    findings = scan_text("photo of jane_doe from the collection")
    assert any("personal name" in f for f in findings)
    # the actual matched substring is deliberately not echoed for this category
    assert "jane_doe" not in " ".join(findings)


def test_plain_technical_note_is_not_flagged():
    assert scan_text("taken outdoors, slightly cloudy, no makeup") == []


def test_empty_string_is_not_flagged():
    assert scan_text("") == []
    assert scan_text(None) == []


def test_path_stem_strips_directory_and_extension():
    assert Path_stem("/some/dir/AM-EVAL-0001.jpg") == "AM-EVAL-0001"
    assert Path_stem("plain_file.png") == "plain_file"


def test_scan_csv_rows_finds_issues_only_in_flagged_cells():
    rows = [
        {"image_id": "AM-EVAL-0001", "notes": "clean note"},
        {"image_id": "AM-EVAL-0002", "notes": "contact john.smith@example.com"},
    ]
    results = scan_csv_rows(rows, ["notes"])
    assert ("AM-EVAL-0001", "notes") not in results
    assert ("AM-EVAL-0002", "notes") in results
