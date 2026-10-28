"""Markdown report generator matching the exact template from the audit's
real-photo evaluation brief (section 26). Renders honestly with no data --
section 27 is explicit: "If no real labeled dataset exists yet: DO NOT
generate fake accuracy numbers... provide an example empty report
structure."

Every table cell is either a real computed number or a literal
"no data yet" -- never a placeholder that could be mistaken for a result.
"""

from __future__ import annotations

SEASONS = ["Spring", "Summer", "Autumn", "Winter"]
UNDERTONES = ["Warm", "Cool", "Neutral"]


def _fmt(x, pct=False):
    if x is None:
        return "no data yet"
    if pct:
        return f"{x:.1%}"
    return f"{x:.3f}"


def _dataset_section(ctx) -> str:
    d = ctx.get("dataset", {})
    lines = [
        "## Dataset\n",
        f"* Number of images found: {d.get('n_images', 0)}",
        f"* Number of labeled images evaluated: {d.get('n_labeled', 0)}",
        f"* Skipped (missing image or landmarks export): {d.get('n_skipped', 0)}",
    ]
    per_season = d.get("per_season", {})
    if per_season:
        lines.append("* Per-season count: " + ", ".join(f"{s}={per_season.get(s, 0)}" for s in SEASONS))
    else:
        lines.append("* Per-season count: no data yet")
    lighting = d.get("lighting_distribution")
    lines.append("* Lighting distribution: " + (", ".join(f"{k}={v}" for k, v in lighting.items()) if lighting else "no data yet"))
    makeup = d.get("makeup_distribution")
    lines.append("* Makeup distribution: " + (", ".join(f"{k}={v}" for k, v in makeup.items()) if makeup else "no data yet"))
    return "\n".join(lines) + "\n"


def _overall_metrics_section(ctx) -> str:
    m = ctx.get("overall_metrics", {})
    rows = [
        ("Accuracy", _fmt(m.get("accuracy"), pct=True)),
        ("Balanced Accuracy", _fmt(m.get("balanced_accuracy"), pct=True)),
        ("Macro F1", _fmt(m.get("macro_f1"))),
        ("Consistency", _fmt(m.get("consistency"), pct=True)),
        ("Failure Rate", _fmt(m.get("failure_rate"), pct=True)),
        ("Low Reliability Rate", _fmt(m.get("low_reliability_rate"), pct=True)),
    ]
    lines = ["## Overall Metrics\n", "| Metric | Score |", "| --- | ---: |"]
    lines += [f"| {name} | {score} |" for name, score in rows]
    return "\n".join(lines) + "\n"


def _confusion_matrix_section(ctx) -> str:
    matrix = ctx.get("confusion_matrix")
    lines = ["## Confusion Matrix\n"]
    if not matrix:
        lines.append("no data yet\n")
        return "\n".join(lines)
    cols = SEASONS + ["NO_PREDICTION"]
    header = "| truth \\ predicted | " + " | ".join(cols) + " |"
    sep = "| --- | " + " | ".join("---:" for _ in cols) + " |"
    lines += [header, sep]
    for truth in SEASONS:
        row = matrix.get(truth, {})
        lines.append(f"| {truth} | " + " | ".join(str(row.get(c, 0)) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _per_season_section(ctx) -> str:
    rows = ctx.get("per_season", [])
    lines = ["## Per-Season Results\n", "| Season | Precision | Recall | F1 | Support |", "| --- | ---: | ---: | ---: | ---: |"]
    if not rows:
        lines.append("| no data yet | | | | |")
    for r in rows:
        lines.append(
            f"| {r['season']} | {_fmt(r['precision'])} | {_fmt(r['recall'])} | {_fmt(r['f1'])} | {r['support']} |"
        )
    return "\n".join(lines) + "\n"


def _stratified_section(title, rows) -> str:
    lines = [f"## {title}\n"]
    if not rows:
        lines.append("no data yet\n")
        return "\n".join(lines)
    lines += ["| Group | n | Accuracy | Balanced Accuracy | Macro F1 | |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for r in rows:
        flag = "⚠ too few samples" if r.get("too_small_to_trust") else ""
        lines.append(
            f"| {r['group']} | {r['n']} | {_fmt(r['accuracy'], pct=True)} | "
            f"{_fmt(r['balanced_accuracy'], pct=True)} | {_fmt(r['macro_f1'])} | {flag} |"
        )
    return "\n".join(lines) + "\n"


def _calibration_section(ctx) -> str:
    buckets = ctx.get("calibration_buckets", [])
    summary = ctx.get("calibration_summary", "no data yet")
    lines = ["## Confidence Calibration\n"]
    if buckets:
        lines += ["| Confidence Range | n | Empirical Accuracy | Mean Reported Confidence |", "| --- | ---: | ---: | ---: |"]
        for b in buckets:
            lines.append(f"| {b['range']} | {b['n']} | {_fmt(b['empirical_accuracy'], pct=True)} | {_fmt(b['mean_reported_confidence'])} |")
    else:
        lines.append("no data yet")
    lines.append(f"\n**Summary:** {summary}\n")
    return "\n".join(lines)


def _ablation_section(ctx) -> str:
    ab = ctx.get("ablation_summary")
    lines = ["## Ablation Results\n"]
    lines.append(ab if ab else "no data yet — see scripts/evaluate_personal_color.py --ablation")
    return "\n".join(lines) + "\n"


def _error_patterns_section(ctx) -> str:
    patterns = ctx.get("major_confusions", [])
    lines = ["## Major Error Patterns\n"]
    if not patterns:
        lines.append("no data yet\n")
        return "\n".join(lines)
    for p in patterns:
        if p["n"] == 0:
            continue
        lines.append(f"### {p['truth']} mistaken for {p['predicted']} (n={p['n']})\n")
        for key, stats in p["features"].items():
            if stats:
                lines.append(f"* {key}: mean={stats['mean']:.2f}, range=[{stats['min']:.2f}, {stats['max']:.2f}] (n={stats['n']})")
        lines.append("")
    if len(lines) == 1:
        lines.append("No confusion pairs had any samples yet.\n")
    return "\n".join(lines)


def _undertone_section(ctx) -> str:
    """Closes the "no dedicated undertone table" gap EVALUATION_PROTOCOL.md
    section 10 flagged. Same minimum-sample discipline as everything
    else here -- `no data yet` when there isn't enough to say something,
    never a number computed from too little to trust.
    """
    matrix = ctx.get("undertone_confusion_matrix")
    per_class = ctx.get("undertone_per_class", [])
    lines = ["## Undertone Results\n"]

    if not matrix and not per_class:
        lines.append("no data yet\n")
        return "\n".join(lines)

    cols = UNDERTONES + ["NO_PREDICTION"]
    header = "| truth \\ predicted | " + " | ".join(cols) + " |"
    sep = "| --- | " + " | ".join("---:" for _ in cols) + " |"
    lines += ["### Confusion Matrix\n", header, sep]
    for truth in UNDERTONES:
        row = (matrix or {}).get(truth, {})
        lines.append(f"| {truth} | " + " | ".join(str(row.get(c, 0)) for c in cols) + " |")

    lines.append("\n### Per-Class\n")
    lines += ["| Undertone | Precision | Recall | F1 | Support |", "| --- | ---: | ---: | ---: | ---: |"]
    if not per_class:
        lines.append("| no data yet | | | | |")
    for r in per_class:
        lines.append(
            f"| {r['undertone']} | {_fmt(r['precision'])} | {_fmt(r['recall'])} | {_fmt(r['f1'])} | {r['support']} |"
        )
    return "\n".join(lines) + "\n"


def _reproducibility_section(ctx) -> str:
    """Closes EVALUATION_PROTOCOL.md section 10 item 12's documented gap:
    "not yet implemented; a real gap." Renders whatever
    services/eval/repro.py::collect() produced; `no data yet` if the
    caller never supplied it (e.g. an older/empty-dataset run).
    """
    info_md = ctx.get("reproducibility_markdown")
    lines = ["## Reproducibility Information\n"]
    lines.append(info_md if info_md else "no data yet")
    return "\n".join(lines) + "\n"


def render_report(ctx: dict) -> str:
    parts = [
        "# AuraMatch Real Photo Evaluation\n",
        _dataset_section(ctx),
        _overall_metrics_section(ctx),
        _confusion_matrix_section(ctx),
        _per_season_section(ctx),
        _undertone_section(ctx),
        _stratified_section("Lighting Results", ctx.get("by_lighting", [])),
        _stratified_section("Makeup Results", ctx.get("by_makeup", [])),
        _calibration_section(ctx),
        _ablation_section(ctx),
        _error_patterns_section(ctx),
        "## Recommended Calibration\n",
        ctx.get("recommended_calibration", "Not enough labeled data to recommend any threshold change. See services/color_engine/season.py's SeasonModel docstring: thresholds stay at their current provisional values until a real, sufficiently large labeled set justifies a specific change.") + "\n",
        "## Remaining Limitations\n",
        ctx.get("remaining_limitations", "See scripts/evaluate_personal_color.py's own output for the current run's specific gaps (e.g. sample size, missing lighting/makeup labels, no hair/iris confidence tier).") + "\n",
        _reproducibility_section(ctx),
    ]
    return "\n".join(parts)
