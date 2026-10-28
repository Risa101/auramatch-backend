"""Tests for services.color_engine.season -- boundary cases per the
implementation-phase brief (Phase 25: "Season classification: Boundary
cases", Phase 11: near-tie must NOT be reported as a confident single
answer).
"""

import pytest

from services.color_engine.season import decide_season


def _features(hue, chroma, L, ita_category="intermediate", confidence="high",
              wb_trustworthy=True, skin_hair_contrast=None, skin_iris_contrast=None):
    return {
        "valid": True,
        "skin_lab": [L, 0.0, 0.0],  # only L is read from skin_lab by decide_season
        "hue_angle": hue,
        "chroma": chroma,
        "ita_category": ita_category,
        "confidence": confidence,
        "wb_trustworthy": wb_trustworthy,
        "skin_hair_contrast": skin_hair_contrast,
        "skin_iris_contrast": skin_iris_contrast,
    }


def test_invalid_input_returns_invalid_decision():
    decision = decide_season({"valid": False, "reason": "no_face"})
    assert decision.valid is False
    assert decision.reason == "no_face"
    assert decision.ranked == []
    assert decision.top_season is None


def test_missing_skin_lab_is_invalid():
    decision = decide_season({"valid": True})
    assert decision.valid is False
    assert decision.reason == "no_skin_lab"


def test_clear_spring_case():
    """Warm hue, light value, bright chroma -> unambiguously Spring."""
    decision = decide_season(_features(hue=75, chroma=35, L=85))
    assert decision.valid is True
    assert decision.top_season == "spring"
    assert decision.top_confidence > 0.9
    assert decision.warm_cool == "warm"
    assert decision.light_deep == "light"
    assert decision.bright_mute == "bright"
    assert decision.reliability == "high"


def test_clear_winter_case():
    """Cool hue, deep value, bright chroma -> unambiguously Winter."""
    decision = decide_season(_features(hue=25, chroma=35, L=30))
    assert decision.valid is True
    assert decision.top_season == "winter"
    assert decision.top_confidence > 0.9


def test_ambiguous_boundary_case_has_low_margin_and_is_not_high_reliability():
    """Sitting exactly on the spring/summer boundary (warm=0, bright=0) must
    produce a genuine tie in probability, not a confident single answer --
    this is the "Spring 0.51 / Autumn 0.49 must not read as 95% confidence"
    case from the implementation brief.
    """
    decision = decide_season(_features(hue=50, chroma=18, L=76))
    assert decision.valid is True
    top, second = decision.ranked[0][1], decision.ranked[1][1]
    assert top == pytest.approx(second, abs=1e-6)
    assert decision.reliability != "high"
    assert any("low_season_margin" in f for f in decision.flags)


def test_untrustworthy_white_balance_demotes_reliability():
    trustworthy = decide_season(_features(hue=75, chroma=35, L=85, wb_trustworthy=True))
    untrustworthy = decide_season(_features(hue=75, chroma=35, L=85, wb_trustworthy=False))
    assert trustworthy.reliability == "high"
    assert untrustworthy.reliability != "high"  # demoted by one full tier
    assert any("wb_untrustworthy" in f for f in untrustworthy.flags)


def test_untrustworthy_white_balance_plus_low_upstream_confidence_is_low():
    decision = decide_season(_features(
        hue=75, chroma=35, L=85, confidence="low", wb_trustworthy=False,
    ))
    assert decision.reliability == "low"


def test_olive_guard_shifts_warm_toward_cool():
    """Warm-surface + high-contrast + mid-depth (intermediate/tan/brown ITA)
    triggers the olive correction and pulls the warm axis toward cool.
    """
    plain = decide_season(_features(hue=60, chroma=20, L=55, ita_category="tan"))
    guarded = decide_season(_features(
        hue=60, chroma=20, L=55, ita_category="tan", skin_hair_contrast=60.0,
    ))
    assert guarded.olive_adjusted is True
    assert plain.olive_adjusted is False
    assert guarded.warm_cool_score < plain.warm_cool_score
    assert any("olive_guard" in f for f in guarded.flags)


def test_ranked_probabilities_sum_to_one():
    decision = decide_season(_features(hue=40, chroma=22, L=50))
    total = sum(p for _, p in decision.ranked)
    # ranked is truncated to top_k=4 == all 4 seasons, so this must be ~1.0
    assert total == pytest.approx(1.0, abs=1e-6)


def test_determinism_same_input_same_output():
    """Same features in -> same decision out, every time (no RNG anywhere in
    this pure-math module).
    """
    feats = _features(hue=45, chroma=25, L=60)
    a = decide_season(feats)
    b = decide_season(feats)
    assert a.as_dict() == b.as_dict()
