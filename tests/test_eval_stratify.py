from services.eval.stratify import (
    MAJOR_CONFUSION_PAIRS, by_lighting, by_makeup, confusion_pair_feature_stats,
)

CLASSES = ["Spring", "Summer", "Autumn", "Winter"]


def test_by_lighting_groups_and_flags_small_samples():
    rows = [
        {"truth": "Spring", "predicted": "Spring", "lighting": "daylight"},
        {"truth": "Winter", "predicted": "Summer", "lighting": "indoor warm"},
    ]
    report = by_lighting(rows, CLASSES, min_group_size=5)
    daylight = next(r for r in report if r["group"] == "daylight")
    assert daylight["n"] == 1
    assert daylight["too_small_to_trust"] is True  # only 1 sample, min is 5


def test_by_lighting_defaults_to_unknown_when_unlabeled():
    rows = [{"truth": "Spring", "predicted": "Spring"}]  # no "lighting" key
    report = by_makeup(rows, CLASSES)
    assert report[0]["group"] == "unknown"


def test_confusion_pair_feature_stats_summarizes_matching_rows():
    rows = [
        {"truth": "Spring", "predicted": "Autumn", "L": 60.0, "hue": 55.0},
        {"truth": "Spring", "predicted": "Autumn", "L": 64.0, "hue": 57.0},
        {"truth": "Spring", "predicted": "Spring", "L": 90.0, "hue": 70.0},  # not this confusion
    ]
    stats = confusion_pair_feature_stats(rows, "Spring", "Autumn", ["L", "hue"])
    assert stats["n"] == 2
    assert stats["features"]["L"]["mean"] == 62.0
    assert stats["features"]["L"]["n"] == 2


def test_confusion_pair_feature_stats_empty_is_honest_not_fabricated():
    stats = confusion_pair_feature_stats([], "Spring", "Autumn", ["L"])
    assert stats["n"] == 0
    assert stats["features"] == {}


def test_major_confusion_pairs_cover_the_brief_list():
    pairs = set(MAJOR_CONFUSION_PAIRS)
    for a, b in [("Spring", "Autumn"), ("Spring", "Winter"), ("Summer", "Winter"), ("Summer", "Autumn")]:
        assert (a, b) in pairs
        assert (b, a) in pairs
