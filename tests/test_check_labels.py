from scripts.check_labels import find_duplicate_row_content


def _row(image_id, season="", undertone="", face_shape="", lighting="", makeup=""):
    return {"image_id": image_id, "season": season, "undertone": undertone,
            "face_shape": face_shape, "lighting": lighting, "makeup": makeup}


def test_flags_two_rows_with_identical_content():
    rows = [
        _row("AM-EVAL-0001", season="Spring", undertone="Warm", face_shape="Oval"),
        _row("AM-EVAL-0002", season="Spring", undertone="Warm", face_shape="Oval"),
    ]
    dups = find_duplicate_row_content(rows)
    assert len(dups) == 1
    _, ids = dups[0]
    assert set(ids) == {"AM-EVAL-0001", "AM-EVAL-0002"}


def test_does_not_flag_rows_that_differ_in_any_field():
    rows = [
        _row("AM-EVAL-0001", season="Spring", undertone="Warm", face_shape="Oval"),
        _row("AM-EVAL-0002", season="Spring", undertone="Warm", face_shape="Round"),
    ]
    assert find_duplicate_row_content(rows) == []


def test_does_not_flag_two_all_blank_rows():
    """Two unlabeled rows aren't a meaningful 'duplicate' -- they just
    haven't been labeled yet."""
    rows = [_row("AM-EVAL-0001"), _row("AM-EVAL-0002")]
    assert find_duplicate_row_content(rows) == []


def test_single_unique_row_is_never_flagged():
    rows = [_row("AM-EVAL-0001", season="Spring")]
    assert find_duplicate_row_content(rows) == []
