"""Stratified breakdowns for the real-photo evaluation harness (audit
real-photo brief, sections 9-11, 13).

Lighting and makeup groupings come straight from the dataset's labels.csv
(developer-provided, never inferred). Skin-tone grouping is intentionally
NOT a demographic label -- it buckets by the same measured ITA category
services/color_engine/color_math.py already computes for every prediction
(very_light/light/intermediate/tan/brown/dark, Chardon et al. 1991 cut
points), reusing existing code rather than inventing a new grouping scheme.
"""

from __future__ import annotations

from collections import defaultdict

from services.eval.metrics import accuracy, balanced_accuracy, macro_f1


def group_by(rows, key_fn):
    groups = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)
    return dict(groups)


def stratified_report(rows, key_fn, classes, min_group_size=1) -> list[dict]:
    """One row per group: n, accuracy, balanced_accuracy, macro_f1.
    Groups smaller than `min_group_size` are still listed (never hidden) but
    flagged as too small to trust -- silent omission would misrepresent
    coverage.
    """
    groups = group_by(rows, key_fn)
    out = []
    for key, group_rows in sorted(groups.items(), key=lambda kv: str(kv[0])):
        out.append({
            "group": key,
            "n": len(group_rows),
            "accuracy": accuracy(group_rows),
            "balanced_accuracy": balanced_accuracy(group_rows, classes),
            "macro_f1": macro_f1(group_rows, classes),
            "too_small_to_trust": len(group_rows) < min_group_size,
        })
    return out


def by_lighting(rows, classes, min_group_size=5):
    return stratified_report(rows, lambda r: r.get("lighting") or "unknown", classes, min_group_size)


def by_makeup(rows, classes, min_group_size=5):
    return stratified_report(rows, lambda r: r.get("makeup") or "unknown", classes, min_group_size)


def by_skin_tone_ita_category(rows, classes, min_group_size=5):
    return stratified_report(rows, lambda r: r.get("ita_category") or "unknown", classes, min_group_size)


def confusion_pair_feature_stats(rows, truth_cls, pred_cls, feature_keys):
    """For rows where truth==truth_cls and predicted==pred_cls (a specific
    confusion, e.g. Spring mistaken for Autumn), summarize the requested
    feature keys (e.g. L*, a*, b*, hue_angle, chroma, ita) -- the point of
    section 13's confusion analysis: which FEATURE is driving the error, not
    just that the error exists.
    """
    matches = [r for r in rows if r["truth"] == truth_cls and r["predicted"] == pred_cls]
    if not matches:
        return {"truth": truth_cls, "predicted": pred_cls, "n": 0, "features": {}}

    features = {}
    for key in feature_keys:
        values = [r[key] for r in matches if r.get(key) is not None]
        if values:
            features[key] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "n": len(values),
            }
        else:
            features[key] = None
    return {"truth": truth_cls, "predicted": pred_cls, "n": len(matches), "features": features}


MAJOR_CONFUSION_PAIRS = [
    ("Spring", "Autumn"), ("Autumn", "Spring"),
    ("Spring", "Winter"), ("Winter", "Spring"),
    ("Summer", "Winter"), ("Winter", "Summer"),
    ("Summer", "Autumn"), ("Autumn", "Summer"),
]
