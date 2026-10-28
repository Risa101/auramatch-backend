#!/usr/bin/env python3
"""
calibrate_season.py

CALIBRATION INFRASTRUCTURE ONLY -- see evaluation_dataset/CALIBRATION_PROTOCOL.md
for the full policy this implements. This CLI NEVER modifies
services/color_engine/season.py or any other production file. It runs the
UNMODIFIED decide_season() against candidate SeasonModel configurations
(services/eval/season_calibration.py), always on development-split rows
(calibration + validation) only -- holdout is read exactly once, and only
via --evaluate-holdout, which refuses to run unless
evaluation_dataset/calibration_status.json says FROZEN (see
services/eval/calibration_status.py).

Three modes, run in this order once real data exists:

    1. Baseline (always first):
       python scripts/calibrate_season.py --baseline-only \\
           --images evaluation_dataset/images --landmarks evaluation_dataset/landmarks \\
           --labels evaluation_dataset/labels.csv --split evaluation_dataset/split.csv \\
           --output evaluation_dataset/results/baseline_results.json \\
           --report evaluation_dataset/results/baseline_report.md

    2. Candidate search (development split only):
       python scripts/calibrate_season.py --candidates candidates.json \\
           --images ... --landmarks ... --labels ... --split ... \\
           --baseline evaluation_dataset/results/baseline_results.json \\
           --output evaluation_dataset/results/calibration_results.json \\
           [--k-fold 5 --seed 42]

    3. Final holdout (once, after a human has frozen calibration_status.json):
       python scripts/calibrate_season.py --evaluate-holdout \\
           --i-understand-this-is-final \\
           --images ... --landmarks ... --labels ... --split ... \\
           --output evaluation_dataset/results/final_holdout_results.json

`candidates.json` is a JSON list of SeasonModel override dicts, e.g.:
    [{"value_pivot": 60.0}, {"chroma_pivot": 22.0, "chroma_half": 12.0}]
Each is validated against SeasonModel's actual fields (unknown parameter
names raise, never silently ignored) before any evaluation runs.

NO DATA = NO FAKE RESULTS: with zero development-split rows (the case as
of this run), this script still completes and still writes its output
files, explicitly marked CALIBRATION STATUS: NOT RUN -- never a fabricated
number, never the 28.3% external CIELAB result substituted in as if it
were this project's own calibration baseline (CALIBRATION_PROTOCOL.md's
explicit rule).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.color_engine.season import DEFAULT_MODEL, SEASONS as SEASON_KEYS
from services.eval import calibration_status, repro
from services.eval.dataset_status import is_frozen as dataset_is_frozen
from services.eval.raw_pipeline_diagnostics import categorize_rejection, full_diagnostics
from services.eval.season_calibration import (
    SEASONS_TITLE,
    VALID_PARAMETER_FIELDS,
    InvalidCandidateError,
    compare_candidates,
    cross_validate,
    detect_class_collapse,
    evaluate_candidate,
    make_candidate,
    margin_for_row,
)
from services.eval.stratify import confusion_pair_feature_stats
from scripts.evaluate_personal_color import find_image, load_labels, load_landmarks

FEATURE_KEYS = ["L", "a", "b", "hue_angle", "chroma"]
MIN_FOLD_N = 20  # below this, k-fold variance is noise-dominated -- report honestly, don't refuse to run


def _load_split(split_path: Path) -> dict:
    import csv
    assignments = {}
    if not split_path.exists():
        return assignments
    with open(split_path, newline="") as f:
        for row in csv.DictReader(f):
            image_id = (row.get("image_id") or "").strip()
            split = (row.get("split") or "").strip()
            if image_id:
                assignments[image_id] = split
    return assignments


def _filter_labels_by_split(labels: dict, split_assignments: dict, splits_allowed: tuple) -> dict:
    """Filters the LABELS dict (not yet-built rows) down to only the
    requested split(s) BEFORE any image is decoded or any row is built.
    This is what makes holdout isolation structural rather than a
    post-hoc filter: in baseline/candidates modes, a holdout image is
    never read, never processed, and never printed anywhere -- there is
    no intermediate "all_rows" collection that ever holds a holdout
    image_id alongside its ground-truth season for those modes to leak.
    """
    return {
        image_id: label for image_id, label in labels.items()
        if split_assignments.get(image_id) in splits_allowed
    }


def _empty_rejection() -> dict:
    return {"total": 0, "accepted": 0, "rejected": 0, "unlabeled": 0, "by_category": {}, "by_season": {s: 0 for s in SEASONS_TITLE}}


def _build_rows(labels: dict, images_dir: Path, landmarks_dir: Path) -> tuple:
    """Returns (calibration_rows, rejection_summary). calibration_rows have
    image_id/truth/L/a/b/hue_angle/chroma/confidence/wb_trustworthy/
    skin_hair_contrast/skin_iris_contrast -- ready for
    services.eval.season_calibration. Reuses full_diagnostics() (the SAME
    real, unmodified pipeline wrapper the root-cause experiment already
    verified) rather than re-deriving any color value.

    `labels` must already be split-filtered by the caller (see
    _filter_labels_by_split) -- this function processes every image_id it
    is given, with no split awareness of its own.
    """
    rows = []
    rejected = _empty_rejection()

    for image_id, label in labels.items():
        truth = label.get("season")
        image_path = find_image(images_dir, image_id)
        landmark_record = load_landmarks(landmarks_dir, image_id)
        rejected["total"] += 1

        if image_path is None or landmark_record is None:
            rejected["rejected"] += 1
            rejected["by_category"]["missing_export"] = rejected["by_category"].get("missing_export", 0) + 1
            if truth in SEASONS_TITLE:
                rejected["by_season"][truth] += 1
            continue

        image_bytes = image_path.read_bytes()
        diag = full_diagnostics(
            image_bytes, landmark_record["landmarks"],
            face_box=landmark_record.get("faceBox"), roll_deg=landmark_record.get("rollDeg"),
            face_detected=bool(landmark_record.get("faceDetected")),
        )

        if not diag.get("valid"):
            rejected["rejected"] += 1
            cat = diag.get("quality_rejection_category") or "other"
            rejected["by_category"][cat] = rejected["by_category"].get(cat, 0) + 1
            if truth in SEASONS_TITLE:
                rejected["by_season"][truth] += 1
            continue

        if truth not in SEASONS_TITLE:
            rejected["unlabeled"] += 1  # accepted by the pipeline but no usable ground truth -- not a rejection, recorded so accepted+rejected+unlabeled==total
            continue

        rejected["accepted"] += 1
        rows.append({
            "image_id": image_id, "truth": truth,
            "L": diag["L"], "a": diag["a"], "b": diag["b"],
            "hue_angle": diag["hue_angle"], "chroma": diag["chroma"], "ita_category": diag["ita_category"],
            "confidence": diag["confidence"], "wb_trustworthy": diag["wb_trustworthy"],
        })

    return rows, rejected


def _model_fingerprint(model) -> str:
    """Tamper-evidence for baseline capture: a hash of every SeasonModel
    field VALUE, so a later reader can verify the baseline really was
    captured against an unmodified DEFAULT_MODEL, not a silently-edited one.
    """
    import hashlib
    from dataclasses import asdict
    payload = json.dumps(asdict(model), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def _boundary_analysis(rows: list, model, *, top_n: int = 10) -> dict:
    margins = [margin_for_row(r, model) for r in rows]
    valid_margins = [m for m in margins if m["margin"] is not None]
    lowest = sorted(valid_margins, key=lambda m: m["margin"])[:top_n]

    by_id = {r["image_id"]: r for r in rows}
    enriched = [{**by_id[m["image_id"]], "predicted": m["predicted"]} for m in margins if m["predicted"] is not None]

    pairs_to_check = [(t, p) for t in SEASONS_TITLE for p in SEASONS_TITLE if t != p]
    confusion_pairs = []
    for truth_cls, pred_cls in pairs_to_check:
        stats = confusion_pair_feature_stats(enriched, truth_cls, pred_cls, FEATURE_KEYS)
        if stats["n"] > 0:
            confusion_pairs.append(stats)
    confusion_pairs.sort(key=lambda s: -s["n"])

    return {"lowest_margin_samples": lowest, "confusion_pairs": confusion_pairs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=Path("evaluation_dataset/images"))
    ap.add_argument("--landmarks", type=Path, default=Path("evaluation_dataset/landmarks"))
    ap.add_argument("--labels", type=Path, default=Path("evaluation_dataset/labels.csv"))
    ap.add_argument("--split", type=Path, default=Path("evaluation_dataset/split.csv"))
    ap.add_argument("--dataset", type=Path, default=Path("evaluation_dataset"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--baseline", type=Path, default=None, help="path to a previously-written baseline_results.json (required for --candidates)")

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--baseline-only", action="store_true")
    mode.add_argument("--candidates", type=Path, default=None, help="JSON file: list of SeasonModel override dicts")
    mode.add_argument("--evaluate-holdout", action="store_true")

    ap.add_argument("--k-fold", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--i-understand-this-is-final", action="store_true",
                     help="required with --evaluate-holdout -- extra, deliberate friction for a one-time, consequential action")
    args = ap.parse_args()

    if args.report and not args.baseline_only:
        print("ERROR: --report is only supported with --baseline-only in this version "
              "(candidates/evaluate-holdout modes do not render a markdown report; use the JSON --output).",
              file=sys.stderr)
        sys.exit(2)

    repo_dir = Path(__file__).resolve().parent.parent
    labels = load_labels(args.labels) if args.labels.exists() else {}
    split_assignments = _load_split(args.split)
    timestamp = datetime.now(timezone.utc).isoformat()
    repro_info = repro.collect(repo_dir=repo_dir, dataset_dir=args.dataset, timestamp=timestamp, config=vars(args))

    # --- EVALUATE-HOLDOUT: the one path allowed to touch holdout rows, and
    # only after two independent, explicit confirmations. Labels are
    # filtered to ONLY the holdout split before any image is read, so
    # development-split rows are never decoded or reported here. ---
    if args.evaluate_holdout:
        if not args.i_understand_this_is_final:
            print("ERROR: --evaluate-holdout requires --i-understand-this-is-final "
                  "(deliberate friction -- this is a one-time, consequential action).", file=sys.stderr)
            sys.exit(1)
        if not calibration_status.is_frozen(args.dataset):
            print("ERROR: calibration_status.json is not FROZEN -- refusing to read holdout rows. "
                  "Calibration parameters must be explicitly reviewed and frozen "
                  "(services/eval/calibration_status.py freeze()) before the holdout split may ever be read. "
                  "This is enforced in code, not just documented.", file=sys.stderr)
            sys.exit(1)
        if not dataset_is_frozen(args.dataset):
            print("ERROR: dataset_status.json is not FROZEN -- the ground-truth snapshot itself "
                  "must be frozen before any final evaluation.", file=sys.stderr)
            sys.exit(1)

        holdout_labels = _filter_labels_by_split(labels, split_assignments, ("holdout",))
        holdout_rows, _holdout_rejection = _build_rows(holdout_labels, args.images, args.landmarks) if holdout_labels else ([], _empty_rejection())
        if not holdout_rows:
            result = {"status": "NOT RUN", "reason": "0 holdout rows available", "n": 0}
        else:
            approved = calibration_status.read_status(args.dataset).get("approved_candidate") or {}
            model = make_candidate(approved)
            result = {"status": "RUN", "approved_candidate": approved, **evaluate_candidate(holdout_rows, model)}
        result["reproducibility"] = repro_info
        _write_json(args.output, result)
        print(f"Final holdout evaluation: {result['status']}. Wrote {args.output}")
        return

    # --- BASELINE / CANDIDATES: labels are filtered to ONLY the
    # development split (calibration + validation) before any image is
    # read -- holdout image_ids never enter _build_rows, so they are
    # never decoded, never scored, and never printed in these modes. ---
    dev_labels = _filter_labels_by_split(labels, split_assignments, ("calibration", "validation"))
    dev_rows, rejection = _build_rows(dev_labels, args.images, args.landmarks) if dev_labels else ([], _empty_rejection())

    # --- BASELINE ---
    if args.baseline_only:
        if not dev_rows:
            result = {
                "status": "NOT RUN", "reason": "0 development-split rows available",
                "n": 0, "model_fingerprint": _model_fingerprint(DEFAULT_MODEL),
                "reproducibility": repro_info,
            }
        else:
            baseline_metrics = evaluate_candidate(dev_rows, DEFAULT_MODEL)
            collapse = detect_class_collapse(baseline_metrics)
            boundary = _boundary_analysis(dev_rows, DEFAULT_MODEL)
            result = {
                "status": "RUN", "n": len(dev_rows), "model_fingerprint": _model_fingerprint(DEFAULT_MODEL),
                "metrics": baseline_metrics, "class_collapse": collapse, "boundary_analysis": boundary,
                "rejection_summary": rejection, "reproducibility": repro_info,
            }
        _write_json(args.output, result)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(_render_baseline_report(result))
        print(f"Baseline: {result['status']}. Wrote {args.output}")
        return

    # --- CANDIDATES ---
    if args.candidates:
        if not args.baseline or not args.baseline.exists():
            print("ERROR: --candidates requires --baseline pointing to an existing baseline_results.json "
                  "(run --baseline-only first -- baseline must always be captured before any candidate is "
                  "evaluated).", file=sys.stderr)
            sys.exit(1)
        baseline_data = json.loads(args.baseline.read_text())
        if baseline_data.get("status") != "RUN":
            print(f"ERROR: baseline at {args.baseline} has status={baseline_data.get('status')!r}, not RUN -- "
                  "cannot compare candidates against a baseline that was never actually measured.", file=sys.stderr)
            sys.exit(1)
        baseline_metrics = baseline_data["metrics"]

        overrides_list = json.loads(args.candidates.read_text())
        if not isinstance(overrides_list, list):
            print("ERROR: --candidates file must contain a JSON list of override dicts.", file=sys.stderr)
            sys.exit(1)

        if not dev_rows:
            result = {"status": "NOT RUN", "reason": "0 development-split rows available", "candidates": [], "reproducibility": repro_info}
            _write_json(args.output, result)
            print(f"Candidates: NOT RUN. Wrote {args.output}")
            return

        candidate_reports = []
        for overrides in overrides_list:
            try:
                model = make_candidate(overrides)
            except InvalidCandidateError as exc:
                candidate_reports.append({"overrides": overrides, "error": str(exc)})
                continue

            candidate_metrics = evaluate_candidate(dev_rows, model)
            collapse = detect_class_collapse(candidate_metrics)
            comparison = compare_candidates(baseline_metrics, candidate_metrics, candidate_overrides=overrides)

            entry = {
                "overrides": overrides, "metrics": candidate_metrics,
                "class_collapse": collapse, "comparison": comparison,
            }

            if args.k_fold:
                cv = cross_validate(dev_rows, model, k=args.k_fold, seed=args.seed)
                if any(f["n"] < MIN_FOLD_N for f in cv["folds"]):
                    cv["warning"] = (f"at least one fold has n < {MIN_FOLD_N} -- variance is "
                                      f"noise-dominated at this sample size, do not treat as statistical confidence")
                entry["cross_validation"] = cv

            candidate_reports.append(entry)

        result = {
            "status": "RUN", "n_development_rows": len(dev_rows), "baseline_source": str(args.baseline),
            "candidates": candidate_reports, "reproducibility": repro_info,
        }
        _write_json(args.output, result)
        print(f"Candidates: RUN ({len(candidate_reports)} evaluated, development split only, "
              f"n={len(dev_rows)}). Wrote {args.output}")
        print("No candidate was applied to production. Human review required before any approval.")
        return


def _render_baseline_report(result: dict) -> str:
    L = ["# Baseline Calibration Report (Development Split Only)\n"]
    if result["status"] != "RUN":
        L.append(f"```text\nCALIBRATION STATUS:\nNOT RUN\n```\nReason: {result.get('reason')}\n")
        L.append(f"\nModel fingerprint (unchanged DEFAULT_MODEL): `{result.get('model_fingerprint')}`\n")
        return "\n".join(L)

    m = result["metrics"]
    L.append(f"- n = {result['n']} development-split (calibration+validation) rows\n")
    L.append(f"- Model fingerprint: `{result['model_fingerprint']}` (unchanged DEFAULT_MODEL)\n")
    L.append(f"- **Balanced accuracy (primary, EVALUATION_PROTOCOL.md section 7)**: {m['balanced_accuracy']}\n")
    L.append(f"- Accuracy: {m['accuracy']}, Macro F1: {m['macro_f1']}, Coverage: {m['coverage']}\n")
    L.append(f"- Class collapse flags: {result['class_collapse']['flags']}\n")
    L.append(f"- Rejection summary: {result['rejection_summary']}\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
