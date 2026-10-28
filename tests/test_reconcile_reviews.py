from scripts.reconcile_reviews import reconcile_one


def _review(reviewer, **fields):
    return {"reviewer": reviewer, **fields}


def test_single_reviewer_is_accepted_as_is():
    result = reconcile_one("AM-EVAL-0001", [_review("R1", season="Spring", undertone="Warm")])
    assert result["season"] == "Spring"
    assert result["undertone"] == "Warm"
    assert "notes" not in result


def test_two_reviewers_agree_is_accepted():
    result = reconcile_one("AM-EVAL-0001", [
        _review("R1", season="Spring"),
        _review("R2", season="Spring"),
    ])
    assert result["season"] == "Spring"
    assert "notes" not in result


def test_two_reviewers_disagree_leaves_field_blank_and_records_both():
    result = reconcile_one("AM-EVAL-0001", [
        _review("R1", season="Spring"),
        _review("R2", season="Autumn"),
    ])
    assert result["season"] is None
    assert "R1=Spring" in result["notes"]
    assert "R2=Autumn" in result["notes"]


def test_fields_are_reconciled_independently():
    """Season disagrees but undertone agrees -- only season goes blank."""
    result = reconcile_one("AM-EVAL-0001", [
        _review("R1", season="Spring", undertone="Warm"),
        _review("R2", season="Autumn", undertone="Warm"),
    ])
    assert result["season"] is None
    assert result["undertone"] == "Warm"
    assert "season disagreement" in result["notes"]
    assert "undertone" not in result["notes"]  # only the disagreeing field is mentioned


def test_no_reviewer_answered_a_field_stays_blank_not_a_disagreement():
    result = reconcile_one("AM-EVAL-0001", [
        _review("R1", season="Spring", undertone=""),
        _review("R2", season="Spring", undertone=""),
    ])
    assert result["undertone"] is None
    assert "notes" not in result  # blank-by-nobody-answering is not a disagreement


def test_one_reviewer_blank_one_answered_counts_as_a_single_answer_not_a_disagreement():
    result = reconcile_one("AM-EVAL-0001", [
        _review("R1", season="Spring"),
        _review("R2", season=""),  # R2 uncertain, left it blank
    ])
    assert result["season"] == "Spring"
    assert "notes" not in result
