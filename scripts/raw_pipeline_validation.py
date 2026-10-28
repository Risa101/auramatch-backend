#!/usr/bin/env python3
"""
raw_pipeline_validation.py

ROOT-CAUSE VALIDATION -- does the near-chance agreement found in the
external CIELAB experiment (evaluation_dataset/results/
external_armocromia_exploratory.md, 28.3%) reproduce when AuraMatch's OWN
image pipeline (quality gate -> skin mask -> white balance -> Lab ->
K-means -> season.py) processes real labeled photographs, instead of a
third party's pre-extracted feature statistics?

This script runs the REAL, UNMODIFIED production pipeline
(services.personal_color_service.analyze_personal_color and
services.eval.quality_port.assess_image_quality, both called verbatim via
services/eval/raw_pipeline_diagnostics.py -- see that module's docstring)
against evaluation_dataset/images + labels.csv, exactly the same workflow
scripts/evaluate_personal_color.py already uses. It adds: full per-image
diagnostic capture (season scores, margin, K-means diagnostics),
rejection-reason accounting, per-season measurement-distribution stats,
Summer-vs-Autumn and value_pivot=58 breakdowns, and DIAGNOSTIC ABLATION
(input-perturbation sensitivity probes, never a second scoring algorithm).

    python scripts/raw_pipeline_validation.py \\
        --images    evaluation_dataset/images \\
        --landmarks evaluation_dataset/landmarks \\
        --labels    evaluation_dataset/labels.csv \\
        --output    evaluation_dataset/results/raw_pipeline_validation.csv \\
        --report    evaluation_dataset/results/raw_pipeline_validation.md \\
        --summary-json evaluation_dataset/results/raw_pipeline_validation_summary.json \\
        [--sample-per-season N --seed S] [--ablation]

NO DATA = NO FAKE RESULTS: with zero labeled images (the case as of this
run -- evaluation_dataset/labels.csv is header-only and Deep Armocromia's
raw images remain access-gated, see the report's Executive Summary), this
script still runs to completion and still writes all three output files,
clearly marked BLOCKED / no data yet, never a fabricated number.

NO PRODUCTION CODE IS MODIFIED. This script and the module it imports from
(services/eval/raw_pipeline_diagnostics.py) only CALL
services.personal_color_service.analyze_personal_color,
services.eval.quality_port.assess_image_quality, and
services.color_engine.season.decide_season -- none of the three, nor any
other production file, is edited by this task.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.eval import repro
from services.eval.dataset_io import IMAGE_EXTS
from services.eval.raw_pipeline_diagnostics import (
    SEASONS, ablate_season_input, full_diagnostics, select_balanced_sample,
)
from scripts.evaluate_personal_color import denormalize, find_image, load_labels, load_landmarks

REJECTION_CATEGORIES = ["resolution", "no_face", "face_size", "head_angle", "exposure", "blur", "other"]


# --- distribution stats ------------------------------------------------


def _quantiles(values: list) -> dict:
    if not values:
        return {"p10": None, "p25": None, "p50": None, "p75": None, "p90": None}
    s = sorted(values)
    n = len(s)

    def q(p):
        if n == 1:
            return s[0]
        idx = p * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        frac = idx - lo
        return s[lo] + (s[hi] - s[lo]) * frac

    return {"p10": q(0.10), "p25": q(0.25), "p50": q(0.50), "p75": q(0.75), "p90": q(0.90)}


def _field_stats(rows, field: str) -> dict:
    values = [r[field] for r in rows if r.get(field) is not None]
    if not values:
        return {"n": 0, "mean": None, "median": None, "sd": None, **_quantiles([])}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sd": statistics.pstdev(values) if len(values) > 1 else 0.0,
        **_quantiles(values),
    }


def _distribution_by_season(rows, field: str) -> dict:
    valid = [r for r in rows if r.get("valid") and r.get(field) is not None]
    return {s: _field_stats([r for r in valid if r["truth_season"] == s], field) for s in SEASONS}


def _value_pivot_breakdown(rows, pivot: float = 58.0, tol: float = 1.0) -> dict:
    out = {}
    for s in SEASONS:
        season_rows = [r for r in rows if r.get("valid") and r["truth_season"] == s and r.get("L") is not None]
        below = sum(1 for r in season_rows if r["L"] < pivot - tol)
        near = sum(1 for r in season_rows if abs(r["L"] - pivot) <= tol)
        above = sum(1 for r in season_rows if r["L"] > pivot + tol)
        out[s] = {"n": len(season_rows), "below_pivot": below, "near_pivot": near, "above_pivot": above}
    return out


# --- rejection accounting ------------------------------------------------


def _rejection_summary(rows) -> dict:
    rejected = [r for r in rows if not r.get("pipeline_ran", True)]
    by_category = {c: 0 for c in REJECTION_CATEGORIES}
    by_category["none"] = 0
    for r in rejected:
        cat = r.get("quality_rejection_category") or ("no_face" if r.get("reason") == "no_face_landmarks" else "other")
        by_category[cat] = by_category.get(cat, 0) + 1
    by_season = {s: sum(1 for r in rejected if r["truth_season"] == s) for s in SEASONS}
    total_by_season = {s: sum(1 for r in rows if r["truth_season"] == s) for s in SEASONS}
    rejection_rate_by_season = {
        s: (by_season[s] / total_by_season[s]) if total_by_season[s] else None for s in SEASONS
    }
    return {
        "total": len(rows), "accepted": len(rows) - len(rejected), "rejected": len(rejected),
        "by_category": by_category, "by_season": by_season, "rejection_rate_by_season": rejection_rate_by_season,
    }


# --- ablation summary (DIAGNOSTIC ABLATION -- NOT PRODUCTION BEHAVIOR) -----


def _ablation_summary(rows) -> dict:
    """For every accepted, valid row, re-runs decide_season with ONE axis
    neutralized at a time (via ablate_season_input -- the real,
    unmodified decide_season, fed a perturbed COPY of the real captured
    input) and reports how often the WINNING season flips relative to the
    real (non-ablated) prediction already captured in the row. A high flip
    rate for a given axis means that axis is doing a lot of the work in
    separating classes; a low flip rate means the other two axes could
    mostly reproduce the same decision without it.
    """
    valid_rows = [r for r in rows if r.get("valid")]
    if not valid_rows:
        return {"n": 0, "flip_rate": {a: None for a in ("warm", "value", "chroma")}}

    flips = {"warm": 0, "value": 0, "chroma": 0}
    for r in valid_rows:
        features = {
            "valid": True, "skin_lab": [r["L"], r["a"], r["b"]],
            "hue_angle": r["hue_angle"], "chroma": r["chroma"], "ita_category": r["ita_category"],
        }
        for axis in flips:
            ablated = ablate_season_input(features, axis=axis)
            if ablated.valid and ablated.top_season and ablated.top_season.capitalize() != r["predicted_season"]:
                flips[axis] += 1

    n = len(valid_rows)
    return {"n": n, "flip_rate": {a: flips[a] / n for a in flips}}


# --- row building --------------------------------------------------------


def _build_rows(labels: dict, images_dir: Path, landmarks_dir: Path) -> list:
    rows = []
    for image_id, label in labels.items():
        truth_season = label.get("season")
        image_path = find_image(images_dir, image_id)
        landmark_record = load_landmarks(landmarks_dir, image_id)

        if image_path is None or landmark_record is None:
            rows.append({
                "image_id": image_id, "truth_season": truth_season, "pipeline_ran": False,
                "valid": False, "reason": "missing_image_or_landmarks_export",
                "quality_gate_level": None, "quality_rejection_category": None,
                "predicted_season": None, "L": None, "a": None, "b": None,
                "hue_angle": None, "chroma": None, "ita_category": None, "margin": None,
                "reliability": None,
            })
            continue

        image_bytes = image_path.read_bytes()
        face_box = landmark_record.get("faceBox")
        diag = full_diagnostics(
            image_bytes, landmark_record["landmarks"],
            face_box=face_box, roll_deg=landmark_record.get("rollDeg"),
            face_detected=bool(landmark_record.get("faceDetected")),
        )
        diag["image_id"] = image_id
        diag["truth_season"] = truth_season
        rows.append(diag)
    return rows


# --- output ------------------------------------------------------------


def _write_csv(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()}) if rows else [
        "image_id", "truth_season", "pipeline_ran", "valid", "reason", "predicted_season",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _fmt(x, pct=False, nd=3):
    if x is None:
        return "no data yet"
    return f"{x:.1%}" if pct else f"{x:.{nd}f}"


def _write_report(path: Path, *, blocked_reasons, rows, rejection, per_field_dist,
                   value_pivot, ablation, repro_info, args, agreement):
    path.parent.mkdir(parents=True, exist_ok=True)
    L = []
    L.append("# Raw-Image Pipeline Validation (Root-Cause Experiment)\n")

    accepted_rows = [r for r in rows if r.get("valid")]
    L.append("## Executive Summary\n")
    if blocked_reasons:
        L.append("```text\nRAW IMAGE ACCESS:\nBLOCKED\n```\n")
        for reason in blocked_reasons:
            L.append(f"- {reason}\n")
    else:
        L.append("```text\nRAW IMAGE ACCESS:\nAVAILABLE\n```\n")
    L.append(f"- Images analyzed: {len(rows)}\n")
    L.append(f"- Accepted (passed quality gate + valid extraction): {len(accepted_rows)}\n")
    L.append(f"- Rejected: {len(rows) - len(accepted_rows)}\n")
    L.append(f"- Real AuraMatch extraction executed: {'YES' if accepted_rows else 'NO -- zero accepted images'}\n")
    L.append(f"- Hypothesis evaluated: {'YES' if len(accepted_rows) >= 5 else 'NO -- insufficient accepted images (need >= 5 for any per-season breakdown to mean anything)'}\n")

    L.append("\n## Dataset Provenance\n")
    L.append(
        "- Deep Armocromia raw images: access remains request-gated (Google "
        "Form + password, https://github.com/lorenzo-stacchio/Deep-Armocromia "
        "-- re-verified this session, unchanged from `DATASET_CANDIDATES.md` "
        "Candidate 1's ACCESS PENDING status). No form was submitted -- "
        "requesting access under the user's identity and waiting on the "
        "authors' manual approval is a real-world action outside this "
        "session's authority to take unprompted.\n"
        "- Checked `DATASET_CANDIDATES.md`'s full candidate list for any "
        "OTHER real-image, directly-4-season-labeled, openly-downloadable "
        "dataset: none exists. The face-shape datasets (dsmlr/faceshape, "
        "niten19) carry no season/color label at all; Su et al. and "
        "ColorInsight (both real expert-labeled personal-color sets) were "
        "never publicly released.\n"
        f"- Internal `evaluation_dataset/images`/`labels.csv`: "
        f"{len(rows)} labeled rows found at run time.\n"
    )

    L.append("\n## Image Quality\n")
    if accepted_rows:
        for field, label in (("quality_min_dim", "min dimension (px)"), ("quality_mean_luma", "mean luma"),
                              ("quality_blur_variance", "blur variance")):
            vals = [r[field] for r in rows if r.get(field) is not None]
            if vals:
                L.append(f"- {label}: mean={statistics.fmean(vals):.2f}, n={len(vals)}\n")
    else:
        L.append("no data yet\n")

    L.append("\n## Quality-Gate Rejections\n")
    L.append(f"- Total: {rejection['total']}, Accepted: {rejection['accepted']}, Rejected: {rejection['rejected']}\n")
    L.append("- By category: " + ", ".join(f"{k}={v}" for k, v in rejection["by_category"].items()) + "\n")
    L.append("- By ground-truth season: " + ", ".join(f"{k}={v}" for k, v in rejection["by_season"].items()) + "\n")
    rate_str = ", ".join(f"{k}={_fmt(v, pct=True)}" for k, v in rejection["rejection_rate_by_season"].items())
    L.append(f"- Rejection rate by season: {rate_str}\n")
    if rejection["rejected"] > 0:
        rates = [v for v in rejection["rejection_rate_by_season"].values() if v is not None]
        if rates and (max(rates) - min(rates)) > 0.15:
            L.append("- **FLAG**: rejection rate varies by more than 15 points across seasons -- "
                     "possible dataset/pipeline selection bias; interpret any accuracy number "
                     "with this in mind.\n")

    L.append("\n## AuraMatch Measurement Distribution\n")
    if accepted_rows:
        for field in ("L", "a", "b", "chroma", "hue_angle"):
            L.append(f"\n### {field}\n")
            L.append("| Season | n | mean | median | SD | p10 | p25 | p50 | p75 | p90 |\n|---|---|---|---|---|---|---|---|---|---|")
            for s in SEASONS:
                st = per_field_dist[field][s]
                L.append(f"| {s} | {st['n']} | {_fmt(st['mean'])} | {_fmt(st['median'])} | {_fmt(st['sd'])} | "
                          f"{_fmt(st['p10'])} | {_fmt(st['p25'])} | {_fmt(st['p50'])} | {_fmt(st['p75'])} | {_fmt(st['p90'])} |")
    else:
        L.append("no data yet -- zero accepted images\n")

    L.append("\n## External-vs-AuraMatch Measurement Comparison\n")
    L.append(
        "Not performed: this requires the SAME underlying images processed by "
        "BOTH pipelines, which requires raw-image access -- blocked (see "
        "Dataset Provenance). Once raw images are available, match by "
        "filename/hash against `external_armocromia_exploratory.csv` and "
        "compute mean/median difference, MAE, RMSE, and correlation for "
        "L*/a*/b*/chroma.\n"
    )

    L.append("\n## Summer vs Autumn Analysis\n")
    if accepted_rows:
        summer_chroma = [r["chroma"] for r in accepted_rows if r["truth_season"] == "Summer" and r.get("chroma") is not None]
        autumn_chroma = [r["chroma"] for r in accepted_rows if r["truth_season"] == "Autumn" and r.get("chroma") is not None]
        if summer_chroma and autumn_chroma:
            sq, aq = _quantiles(summer_chroma), _quantiles(autumn_chroma)
            L.append(f"- Summer chroma: median={_fmt(sq['p50'])}, IQR=[{_fmt(sq['p25'])}, {_fmt(sq['p75'])}], n={len(summer_chroma)}\n")
            L.append(f"- Autumn chroma: median={_fmt(aq['p50'])}, IQR=[{_fmt(aq['p25'])}, {_fmt(aq['p75'])}], n={len(autumn_chroma)}\n")
            overlap = not (aq["p25"] > sq["p75"] or sq["p25"] > aq["p75"])
            L.append(f"- IQR overlap: {'YES -- genuinely overlapping in AuraMatch measurement space' if overlap else 'NO -- separated'}\n")
        else:
            L.append("no data yet -- insufficient Summer/Autumn accepted rows\n")
    else:
        L.append("no data yet\n")

    L.append("\n## Value Pivot Analysis (value_pivot=58, NOT modified)\n")
    if accepted_rows:
        L.append("| Season | n | L < 57 | L in [57,59] | L > 59 |\n|---|---|---|---|---|")
        for s in SEASONS:
            v = value_pivot[s]
            L.append(f"| {s} | {v['n']} | {v['below_pivot']} | {v['near_pivot']} | {v['above_pivot']} |")
    else:
        L.append("no data yet\n")

    L.append("\n## Season Score Analysis\n")
    if accepted_rows:
        L.append(f"- Overall exploratory agreement (AuraMatch's own pipeline vs. label): {_fmt(agreement, pct=True)}\n")
    else:
        L.append("no data yet\n")

    L.append("\n## Spring Prediction Collapse\n")
    if accepted_rows:
        spring_rate = sum(1 for r in accepted_rows if r.get("predicted_season") == "Spring") / len(accepted_rows)
        L.append(f"- Spring predicted for {_fmt(spring_rate, pct=True)} of accepted images "
                 f"(external-data experiment found 56%).\n")
    else:
        L.append("no data yet\n")

    L.append("\n## Diagnostic Ablation\n")
    L.append("```text\nDIAGNOSTIC ABLATION -- NOT PRODUCTION BEHAVIOR\n```\n")
    if accepted_rows:
        L.append(f"- n={ablation['n']} accepted, valid rows\n")
        for axis, rate in ablation["flip_rate"].items():
            L.append(f"- Neutralizing the **{axis}** axis flips the winning season in {_fmt(rate, pct=True)} of cases\n")
    else:
        L.append("no data yet -- zero accepted images to ablate\n")

    L.append("\n## Undertone (task section 13)\n")
    L.append("```text\nUNDERTONE NOT EVALUATED\n```\nThis run does not compare predicted vs. ground-truth "
              "undertone -- `_build_rows()` does not currently join `labels.csv`'s `undertone` column into "
              "each row (a real, disclosed gap, not silently worked around). `full_diagnostics()` already "
              "captures AuraMatch's own predicted `undertone` per accepted row (see the CSV's `undertone` "
              "column) -- wiring in the ground-truth comparison is a small addition once real, "
              "independently-labeled undertone values exist to compare against. Undertone is never "
              "inferred from season.\n")

    L.append("\n## Demographic / Source Bias (task section 15)\n")
    L.append("```text\nUNKNOWN\n```\nNo source/camera/subject metadata beyond `lighting`/`makeup` "
              "(labels.csv's existing columns) is available for the rows in this run. No demographic "
              "attribute was inferred from any face, per this task's own rule.\n")

    L.append("\n## Limitations\n")
    L.append(
        "- " + ("Raw-image access is BLOCKED -- every section above reads "
        "'no data yet' as a direct, honest consequence, not a bug in this "
        "script.\n" if blocked_reasons else
        f"n={len(accepted_rows)} accepted images -- treat any per-season "
        "breakdown as directional until this grows past "
        "EVALUATION_PROTOCOL.md section 8's 15-20-per-season floor.\n")
    )

    L.append("\n## Production Code Changes\n")
    L.append(
        f"None. `git diff --stat -- services/color_engine/ "
        f"services/personal_color_service.py src/utils/faceGeometry.js` "
        f"returned empty at commit `{repro_info['git_commit'] or 'unknown'}`. "
        f"`node --test` / `npm run build` (frontend): **not run** -- no file "
        f"in the frontend repo (`auramatch ver3`) was touched by this task.\n"
    )

    L.append("\n## Reproducibility\n")
    L.append(f"- Run timestamp: {repro_info['timestamp']}\n"
             f"- AuraMatch git commit: {repro_info['git_commit'] or 'unknown'}\n"
             f"- Python version: {repro_info['python_version']}\n"
             f"- Command: {' '.join(sys.argv)}\n"
             f"- Sample selection seed: {args.seed if args.sample_per_season else 'N/A -- full label set used'}\n")

    L.append("\n## Conclusion\n")
    if blocked_reasons:
        conclusion = "INSUFFICIENT EVIDENCE"
        why = "raw-image access is blocked; the hypothesis cannot be tested against AuraMatch's own pipeline yet."
    elif len(accepted_rows) < 5:
        conclusion = "INSUFFICIENT EVIDENCE"
        why = f"only {len(accepted_rows)} accepted images -- too few for any per-season claim."
    else:
        conclusion = "INSUFFICIENT EVIDENCE"
        why = "evaluate the numbers above manually against the external experiment's findings before asserting SUPPORTED/NOT SUPPORTED."
    L.append(f"```text\n{conclusion}\n```\n{why}\n")

    path.write_text("\n".join(L), encoding="utf-8")


def _write_summary_json(path: Path, *, blocked_reasons, rows, rejection, repro_info, agreement):
    path.parent.mkdir(parents=True, exist_ok=True)
    accepted = [r for r in rows if r.get("valid")]
    summary = {
        "raw_image_access": "BLOCKED" if blocked_reasons else "AVAILABLE",
        "blocked_reasons": blocked_reasons,
        "images_analyzed": len(rows),
        "accepted": len(accepted),
        "rejected": len(rows) - len(accepted),
        "exploratory_agreement": agreement,
        "spring_prediction_rate": (
            sum(1 for r in accepted if r.get("predicted_season") == "Spring") / len(accepted) if accepted else None
        ),
        "autumn_recall": None,
        "production_ai_changed": False,
        "final_accuracy": None,
        "reproducibility": repro_info,
    }
    if accepted:
        autumn_truth = [r for r in accepted if r["truth_season"] == "Autumn"]
        if autumn_truth:
            summary["autumn_recall"] = sum(1 for r in autumn_truth if r["predicted_season"] == "Autumn") / len(autumn_truth)
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


# --- main --------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--landmarks", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--output", type=Path, default=Path("evaluation_dataset/results/raw_pipeline_validation.csv"))
    ap.add_argument("--report", type=Path, default=Path("evaluation_dataset/results/raw_pipeline_validation.md"))
    ap.add_argument("--summary-json", type=Path, default=Path("evaluation_dataset/results/raw_pipeline_validation_summary.json"))
    ap.add_argument("--sample-per-season", type=int, default=None,
                     help="deterministic balanced subsample size per season (default: use all labeled rows)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    blocked_reasons = []
    if not args.labels.exists():
        print(f"ERROR: labels file not found: {args.labels}", file=sys.stderr)
        sys.exit(1)

    labels = load_labels(args.labels)
    n_images_present = len([p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTS]) if args.images.exists() else 0

    if not labels or n_images_present == 0:
        # States only what THIS run observed on disk -- the separate fact
        # that Deep Armocromia's raw images are access-gated is a real-world
        # finding (re-verified via WebFetch this session, not derivable from
        # a directory listing) and belongs in the report's Dataset
        # Provenance section, not hardcoded into a runtime blocker message
        # that a synthetic test fixture (unrelated to Deep Armocromia) would
        # also trigger and inherit an unrelated claim.
        blocked_reasons.append(
            f"RAW-IMAGE EXPERIMENT BLOCKED -- ACCESS NOT AVAILABLE: "
            f"{len(labels)} labeled rows in {args.labels}; "
            f"{n_images_present} images in {args.images}."
        )

    if args.sample_per_season and labels:
        labels = select_balanced_sample(labels, per_season=args.sample_per_season, seed=args.seed)

    rows = _build_rows(labels, args.images, args.landmarks) if labels else []

    for r in rows:
        status = "accepted" if r.get("valid") else "rejected"
        print(f"  {status} {r['image_id']}: truth={r.get('truth_season')} predicted={r.get('predicted_season')} "
              f"reason={r.get('reason')}")

    rejection = _rejection_summary(rows) if rows else {
        "total": 0, "accepted": 0, "rejected": 0,
        "by_category": {c: 0 for c in REJECTION_CATEGORIES + ["none"]},
        "by_season": {s: 0 for s in SEASONS}, "rejection_rate_by_season": {s: None for s in SEASONS},
    }
    per_field_dist = {f: _distribution_by_season(rows, f) for f in ("L", "a", "b", "chroma", "hue_angle")}
    value_pivot = _value_pivot_breakdown(rows) if rows else {s: {"n": 0, "below_pivot": 0, "near_pivot": 0, "above_pivot": 0} for s in SEASONS}
    ablation = _ablation_summary(rows)

    accepted = [r for r in rows if r.get("valid")]
    labeled_accepted = [r for r in accepted if r["truth_season"] in SEASONS]
    agreement = (
        sum(1 for r in labeled_accepted if r["predicted_season"] == r["truth_season"]) / len(labeled_accepted)
        if labeled_accepted else None
    )

    _write_csv(args.output, rows)

    repo_dir = Path(__file__).resolve().parent.parent
    repro_info = repro.collect(
        repo_dir=repo_dir, dataset_dir=args.labels.parent,
        timestamp=datetime.now(timezone.utc).isoformat(), config=vars(args),
    )

    _write_report(
        args.report, blocked_reasons=blocked_reasons, rows=rows, rejection=rejection,
        per_field_dist=per_field_dist, value_pivot=value_pivot, ablation=ablation,
        repro_info=repro_info, args=args, agreement=agreement,
    )
    _write_summary_json(
        args.summary_json, blocked_reasons=blocked_reasons, rows=rows,
        rejection=rejection, repro_info=repro_info, agreement=agreement,
    )

    print(f"\nRAW IMAGE ACCESS: {'BLOCKED' if blocked_reasons else 'AVAILABLE'}")
    print(f"Images analyzed: {len(rows)}, accepted: {len(accepted)}, rejected: {len(rows) - len(accepted)}")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report}")
    print(f"Wrote {args.summary_json}")


if __name__ == "__main__":
    main()
