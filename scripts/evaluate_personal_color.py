#!/usr/bin/env python3
"""
evaluate_personal_color.py

Batch real-photo evaluation for services/personal_color_service.py. See
evaluation_dataset/README.md for the full dataset format and workflow.

    python scripts/evaluate_personal_color.py \\
        --images    evaluation_dataset/images \\
        --landmarks evaluation_dataset/landmarks \\
        --labels    evaluation_dataset/labels.csv \\
        --output    evaluation_dataset/results/results.csv \\
        --report    evaluation_dataset/results/evaluation_report.md \\
        [--consistency] [--ablation] [--baselines] [--debug evaluation_dataset/debug] \\
        [--validate-only] [--require-frozen]

Run with backend_auramatch's normal `.venv` -- this script only reads
already-exported landmark JSON (see scripts/export_landmarks.py), it does
not run MediaPipe itself, so it needs no mediapipe dependency.

NO DATA = NO FAKE RESULTS. If `--labels` has no rows with a filled-in
`season`, this script still runs to completion and still writes
`evaluation_report.md` -- every metric in it reads "no data yet" rather than
a fabricated number (see services/eval/report.py). Nothing in this file
invents a label, guesses at missing data, or reports a number computed from
fewer samples than it's presented as if it were reliable.

PRIVACY: no network calls, nothing sent to Gemini or any third party, no
raw pixel data logged. --debug writes visualizations only to the local
directory you specify.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2 as cv

from services.color_engine import color_math as cm
from services.color_engine.skin_mask import build_skin_mask
from services.color_engine.white_balance import white_balance
from services.eval import ablation, calibration, metrics, repro, stratify
from services.eval import report as report_mod
from services.eval.dataset_io import IMAGE_EXTS, find_image
from services.eval.debug_viz import save_debug_bundle
from services.eval.quality_port import assess_image_quality
from services.eval.validate_labels import (
    format_issues, has_blocking_errors, validate_labels_csv,
)
from services.eval.dataset_status import is_frozen, read_status
from services.eval.integrity import (
    check_all_landmarks, find_exact_duplicates, find_near_duplicates,
)
from services.personal_color_service import analyze_personal_color

SEASONS = ["Spring", "Summer", "Autumn", "Winter"]
UNDERTONES = ["Warm", "Cool", "Neutral"]
FEATURE_KEYS = ["L", "a", "b", "hue_angle", "chroma", "ita"]


# --- dataset loading ----------------------------------------------------


def load_labels(path: Path) -> dict:
    """Reads labels.csv into {image_id: {column: value_or_None}}.

    Tolerates a `notes` (or any) column containing an unescaped comma --
    csv.DictReader bundles anything past the declared columns under the
    `None` key as a list; that's dropped here rather than crashing, since a
    developer hand-editing a notes field is a normal, expected occurrence,
    not a malformed dataset.
    """
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            image_id = (row.get("image_id") or "").strip()
            if not image_id:
                continue
            rows[image_id] = {
                k: (v.strip() or None) for k, v in row.items() if k is not None and isinstance(v, str)
            }
    return rows


def load_landmarks(landmarks_dir: Path, image_id: str) -> dict | None:
    stem = Path(image_id).stem
    p = landmarks_dir / f"{stem}.landmarks.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def denormalize(landmarks_norm, width: int, height: int) -> np.ndarray:
    arr = np.asarray([[pt[0], pt[1]] for pt in landmarks_norm], dtype=float)
    arr[:, 0] *= width
    arr[:, 1] *= height
    return arr


def encode_png(bgr) -> bytes:
    ok, buf = cv.imencode(".png", bgr)
    if not ok:
        raise ValueError("failed to encode image")
    return buf.tobytes()


# --- consistency test (audit real-photo brief, section 8) ---------------


def make_consistency_variants(bgr) -> dict:
    """In-memory only -- never writes back to the source file. Landmarks are
    reused unchanged across all variants (see this function's caller):
    resizing doesn't change normalized landmark coordinates at all, and for
    the brightness/warmth/compression variants reusing the ORIGINAL
    landmarks is the deliberate experimental design -- it isolates "does the
    COLOR pipeline change its answer when only pixel values shift" from "did
    landmark detection also drift," which are two different questions.
    """
    variants = {
        "resized_75pct": cv.resize(bgr, None, fx=0.75, fy=0.75),
        "resized_50pct": cv.resize(bgr, None, fx=0.5, fy=0.5),
        "brighter": np.clip(bgr.astype(np.float64) * 1.15, 0, 255).astype(np.uint8),
        "darker": np.clip(bgr.astype(np.float64) * 0.85, 0, 255).astype(np.uint8),
    }
    warm = bgr.astype(np.float64).copy()
    warm[..., 2] *= 1.10  # BGR: boost R
    warm[..., 0] *= 0.92  # cut B
    variants["warmer"] = np.clip(warm, 0, 255).astype(np.uint8)

    cool = bgr.astype(np.float64).copy()
    cool[..., 0] *= 1.10
    cool[..., 2] *= 0.92
    variants["cooler"] = np.clip(cool, 0, 255).astype(np.uint8)

    ok, enc = cv.imencode(".jpg", bgr, [int(cv.IMWRITE_JPEG_QUALITY), 40])
    if ok:
        variants["jpeg_q40"] = cv.imdecode(enc, cv.IMREAD_COLOR)
    return variants


def run_consistency_check(bgr, landmarks_norm) -> dict:
    base = analyze_personal_color(encode_png(bgr), landmarks_norm)
    base_season = base.get("season") if base.get("valid") else None
    per_variant = {}
    for name, variant_bgr in make_consistency_variants(bgr).items():
        r = analyze_personal_color(encode_png(variant_bgr), landmarks_norm)
        per_variant[name] = r.get("season") if r.get("valid") else None
    matches = sum(1 for v in per_variant.values() if v == base_season)
    return {
        "base_season": base_season,
        "variants": per_variant,
        "consistency": (matches / len(per_variant)) if per_variant else None,
    }


# --- human-readable validation summary (collection-workflow brief, section 12) ---


def print_validation_summary(labels: dict, issues: list, images_dir: Path, landmarks_dir: Path) -> None:
    """Prints the "Images / Labels / Landmarks / Status" block. Every
    number here is computed from the actual files just read -- nothing is
    hard-coded. This is purely additional VISIBILITY on top of
    validate_labels_csv()'s existing pass/fail behavior; it never changes
    the exit code by itself (duplicates and missing landmarks are
    informational here; a genuinely blocking CSV error already exits
    non-zero before this function is even called with landmarks in view).
    """
    n_images = len([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]) if images_dir.exists() else 0
    missing_image_refs = sum(1 for i in issues if i.severity == "warning" and "no image file found" in i.message)
    exact_dup_groups = find_exact_duplicates(images_dir) if images_dir.exists() else {}
    n_exact_dups = sum(len(v) for v in exact_dup_groups.values())

    n_total_labels = len(labels)
    n_disagreements = sum(1 for row in labels.values() if "disagreement" in (row.get("notes") or "").lower())
    n_uncertain = sum(
        1 for row in labels.values()
        if (row.get("notes") or "").strip()
        and "disagreement" not in (row.get("notes") or "").lower()
        and not all((row.get(k) or "").strip() for k in ("season", "undertone", "face_shape"))
    )
    n_invalid = sum(1 for i in issues if i.severity == "error")
    n_valid = max(0, n_total_labels - n_disagreements - n_uncertain)

    landmark_states = check_all_landmarks(list(labels.keys()), landmarks_dir) if landmarks_dir.exists() else []
    n_lm_present = sum(1 for r in landmark_states if r["state"] == "SUCCESS")
    n_lm_missing = sum(1 for r in landmark_states if r["state"] == "MISSING")
    n_lm_invalid = sum(1 for r in landmark_states if r["state"] in ("INVALID", "NO_FACE", "MULTIPLE_FACES"))

    status_line = "FAIL" if has_blocking_errors(issues) else "PASS"

    print("\nAuraMatch Evaluation Dataset Validation\n")
    print("Images:")
    print(f"  Total: {n_images}")
    print(f"  Missing (labeled but no file found): {missing_image_refs}")
    print(f"  Exact duplicates: {n_exact_dups}")
    print("\nLabels:")
    print(f"  Valid: {n_valid}")
    print(f"  Uncertain (notes present, label incomplete): {n_uncertain}")
    print(f"  Disagreements: {n_disagreements}")
    print(f"  Invalid (blocking errors): {n_invalid}")
    if landmarks_dir.exists():
        print("\nLandmarks:")
        print(f"  Present: {n_lm_present}")
        print(f"  Missing: {n_lm_missing}")
        print(f"  Invalid/no-usable-face: {n_lm_invalid}")
    else:
        print(f"\nLandmarks: {landmarks_dir} does not exist yet -- run scripts/export_landmarks.py "
              f"(this is expected if you're validating BEFORE landmark export, per the documented workflow order)")
    print(f"\nStatus: {status_line}")


# --- main -----------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--landmarks", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--output", type=Path, default=None, help="per-image results CSV")
    ap.add_argument("--report", type=Path, default=None, help="evaluation_report.md path")
    ap.add_argument("--consistency", action="store_true", help="section 8: brightness/warmth/resize/JPEG robustness")
    ap.add_argument("--ablation", action="store_true", help="sections 18-20: white-balance and skin-sampling ablations")
    ap.add_argument("--baselines", action="store_true", help="section 17: baseline A/B/C comparison")
    ap.add_argument("--debug", type=Path, default=None, help="write visual debug bundles here (section 22)")
    ap.add_argument("--validate-only", action="store_true",
                     help="check labels.csv, images, and (if exported) landmarks for errors, "
                          "print the human-readable summary, and exit -- no analysis is run")
    ap.add_argument("--require-frozen", action="store_true",
                     help="refuse to run a full evaluation unless evaluation_dataset/dataset_status.json "
                          "says FROZEN (EVALUATION_PROTOCOL.md section 27) -- has no effect with --validate-only")
    args = ap.parse_args()

    if not args.labels.exists():
        print(f"ERROR: labels file not found: {args.labels}", file=sys.stderr)
        sys.exit(1)

    # Validate BEFORE loading into a dict -- a dict keyed by image_id can no
    # longer see a duplicate row once it's been silently overwritten by a
    # later one with the same key, so duplicate detection has to happen on
    # the raw file first.
    issues = validate_labels_csv(args.labels, images_dir=args.images if args.images.exists() else None)
    print(format_issues(issues))
    if has_blocking_errors(issues):
        print("\nRefusing to evaluate against labels.csv with the error(s) above -- fix them first "
              "(see evaluation_dataset/README.md for the valid values). No analysis was run.", file=sys.stderr)
        sys.exit(1)

    labels = load_labels(args.labels)

    if args.validate_only:
        print_validation_summary(labels, issues, args.images, args.landmarks)
        return

    if args.require_frozen:
        dataset_dir = args.labels.resolve().parent
        if not is_frozen(dataset_dir):
            status = read_status(dataset_dir)
            print(f"ERROR: --require-frozen was set but dataset_status.json says {status['status']!r}, "
                  f"not FROZEN (in {dataset_dir}). Refusing to run a full evaluation against a dataset "
                  f"that may still change -- see scripts/dataset_status.py and EVALUATION_PROTOCOL.md section 27.",
                  file=sys.stderr)
            sys.exit(1)

    n_images_present = len([p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTS]) if args.images.exists() else 0

    if not labels:
        print(f"No rows in {args.labels} -- see evaluation_dataset/README.md for the format.")
        empty_report = report_mod.render_report({"dataset": {"n_images": n_images_present, "n_labeled": 0, "n_skipped": 0}})
        print("\n" + empty_report)
        if args.report:
            args.report.write_text(empty_report)
            print(f"\nWrote an empty report (no data yet) to {args.report}")
        return

    rows = []
    consistency_scores = []
    n_skipped = 0
    per_season_count = {s: 0 for s in SEASONS}
    lighting_dist, makeup_dist = {}, {}

    for image_id, label in labels.items():
        image_path = find_image(args.images, image_id)
        landmark_record = load_landmarks(args.landmarks, image_id)

        if image_path is None or landmark_record is None:
            n_skipped += 1
            print(f"  skip {image_id}: missing image or landmarks export")
            continue
        if not landmark_record.get("faceDetected"):
            n_skipped += 1
            print(f"  skip {image_id}: no face detected during landmark export")
            continue

        bgr = cv.imread(str(image_path))
        if bgr is None:
            n_skipped += 1
            print(f"  skip {image_id}: could not decode image")
            continue

        image_bytes = image_path.read_bytes()
        lm_norm = landmark_record["landmarks"]
        result = analyze_personal_color(image_bytes, lm_norm)

        face_box = landmark_record.get("faceBox")
        quality = assess_image_quality(
            bgr,
            face_width_fraction=(face_box["width"] if face_box else None),
            roll_deg=landmark_record.get("rollDeg"),
        )

        truth_season = label.get("season")
        if truth_season in SEASONS:
            per_season_count[truth_season] += 1
        lighting_key = label.get("lighting") or "unknown"
        lighting_dist[lighting_key] = lighting_dist.get(lighting_key, 0) + 1
        makeup_key = label.get("makeup") or "unknown"
        makeup_dist[makeup_key] = makeup_dist.get(makeup_key, 0) + 1

        skin_lab = result.get("skin_lab")
        L, a, b = (skin_lab if skin_lab else (None, None, None))
        hue_angle = float(cm.hue_angle(np.array(skin_lab))) if skin_lab else None
        chroma = float(cm.chroma(np.array(skin_lab))) if skin_lab else None

        row = {
            "image_id": image_id,
            "truth": truth_season,
            "predicted": result.get("season") if result.get("valid") else None,
            "confidence": result.get("confidence"),
            "reliability": result.get("reliability"),
            "valid": result.get("valid"),
            "reason": result.get("reason"),
            "quality_level": quality["level"],
            "quality_reasons": ";".join(quality["reasons"]),
            "lighting": label.get("lighting"),
            "makeup": label.get("makeup"),
            "L": L, "a": a, "b": b,
            "hue_angle": hue_angle, "chroma": chroma,
            "ita": result.get("ita"),
            "ita_category": result.get("ita_category"),
            # season.py's warm_cool axis is lowercase ("warm"/"cool"/"neutral");
            # labels.csv's canonical values are Title case (validate_labels.py) --
            # capitalize here so the undertone confusion matrix compares like
            # with like instead of silently never matching on case alone.
            "undertone_predicted": (result.get("undertone") or "").capitalize() or None,
            "undertone_truth": label.get("undertone"),
        }

        if args.baselines:
            baseline_a = ablation.baseline_a_legacy(image_bytes)
            row["baseline_a_season"] = baseline_a.get("season")
            row["baseline_b_season"] = row["predicted"]  # the pipeline already run above
            row["candidate_c_season"] = row["predicted"]  # == B today, see services/eval/ablation.py

        if args.ablation and result.get("valid"):
            lm_px = denormalize(lm_norm, landmark_record["width"], landmark_record["height"])
            mask = build_skin_mask(bgr, lm_px)
            wb_result = ablation.white_balance_ablation(bgr, mask)
            row["wb_delta_e_raw_vs_corrected"] = wb_result["delta_e_raw_vs_wb"]
            sampling = ablation.skin_sampling_ablation(bgr, mask)
            if "mean" in sampling:
                l_values = [sampling[m][0] for m in ("mean", "median", "trimmed_mean", "kmeans")]
                row["sampling_method_L_spread"] = round(max(l_values) - min(l_values), 4)

        if args.consistency and result.get("valid"):
            c = run_consistency_check(bgr, lm_norm)
            row["consistency"] = c["consistency"]
            if c["consistency"] is not None:
                consistency_scores.append(c["consistency"])

        if args.debug:
            lm_px = denormalize(lm_norm, landmark_record["width"], landmark_record["height"])
            mask = build_skin_mask(bgr, lm_px)
            wb_image = white_balance(bgr)
            save_debug_bundle(
                image_bgr=bgr, landmarks_px=lm_px, skin_mask=mask, wb_image=wb_image,
                rep_lab=skin_lab, out_dir=args.debug, stem=Path(image_id).stem,
            )

        rows.append(row)
        status = "✓" if row["predicted"] == row["truth"] else ("?" if not row["truth"] else "✗")
        print(f"  {status} {image_id}: truth={row['truth']} predicted={row['predicted']} "
              f"conf={row['confidence']} reliability={row['reliability']}")

    if args.output:
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote per-image results to {args.output}")

    labeled_rows = [r for r in rows if r["truth"] in SEASONS]

    overall_metrics = {
        "accuracy": metrics.accuracy(labeled_rows),
        "balanced_accuracy": metrics.balanced_accuracy(labeled_rows, SEASONS),
        "macro_f1": metrics.macro_f1(labeled_rows, SEASONS),
        "consistency": (sum(consistency_scores) / len(consistency_scores)) if consistency_scores else None,
        "failure_rate": metrics.rate(rows, lambda r: not r["valid"]),
        "low_reliability_rate": metrics.rate(rows, lambda r: r.get("reliability") == "low"),
    }

    per_season = []
    for s in SEASONS:
        p, r, f1, support = metrics.precision_recall_f1(labeled_rows, s)
        per_season.append({"season": s, "precision": p, "recall": r, "f1": f1, "support": support})

    # Undertone: closes EVALUATION_PROTOCOL.md's documented "no dedicated
    # undertone table" gap. Reuses the same generic metrics.* functions
    # SEASONS already uses -- confusion_matrix/precision_recall_f1 take an
    # arbitrary class list, no season-specific logic to duplicate.
    undertone_rows = [
        {"truth": r["undertone_truth"], "predicted": r["undertone_predicted"]}
        for r in rows if r["undertone_truth"] in UNDERTONES
    ]
    undertone_confusion_matrix = metrics.confusion_matrix(undertone_rows, UNDERTONES) if undertone_rows else None
    undertone_per_class = [
        {"undertone": u, **dict(zip(("precision", "recall", "f1", "support"), metrics.precision_recall_f1(undertone_rows, u)))}
        for u in UNDERTONES
    ] if undertone_rows else []

    major_confusions = [
        stratify.confusion_pair_feature_stats(labeled_rows, truth, pred, FEATURE_KEYS)
        for truth, pred in stratify.MAJOR_CONFUSION_PAIRS
    ]

    ablation_summary = None
    if args.ablation:
        deltas = [r["wb_delta_e_raw_vs_corrected"] for r in rows if r.get("wb_delta_e_raw_vs_corrected") is not None]
        spreads = [r["sampling_method_L_spread"] for r in rows if r.get("sampling_method_L_spread") is not None]
        ablation_summary = (
            f"White balance shifted skin Lab by a mean Delta E of {sum(deltas)/len(deltas):.2f} "
            f"across {len(deltas)} images (see results.csv's wb_delta_e_raw_vs_corrected column for "
            f"per-image detail; whether that shift moved predictions TOWARD or AWAY from ground truth "
            f"needs `--baselines` cross-referenced against the confusion matrix, not this number alone).\n\n"
            f"Skin-sampling methods (mean/median/trimmed-mean/K-means) spread by up to "
            f"{max(spreads):.2f} L* units on at least one image (mean spread {sum(spreads)/len(spreads):.2f})."
            if deltas and spreads else
            "No ablation rows produced (see results.csv for per-image reasons -- likely all analyses failed or --ablation ran with 0 valid results)."
        )

    if args.baselines:
        a_rows = [{"truth": r["truth"], "predicted": r.get("baseline_a_season")} for r in labeled_rows]
        b_rows = [{"truth": r["truth"], "predicted": r.get("baseline_b_season")} for r in labeled_rows]
        baseline_text = (
            f"Baseline A (original production classifier): accuracy={metrics.accuracy(a_rows)}, "
            f"balanced_accuracy={metrics.balanced_accuracy(a_rows, SEASONS)}\n"
            f"Baseline B (new ColorEngine): accuracy={metrics.accuracy(b_rows)}, "
            f"balanced_accuracy={metrics.balanced_accuracy(b_rows, SEASONS)}\n"
            f"Candidate C (calibrated ColorEngine): identical to Baseline B today -- "
            f"no real calibration has been performed yet (see services/eval/ablation.py)."
        )
        ablation_summary = (ablation_summary + "\n\n" + baseline_text) if ablation_summary else baseline_text

    n_total = len(rows)
    limitations = [f"{n_total} images evaluated, {len(labeled_rows)} with a season label."]
    if len(labeled_rows) < 60:
        limitations.append(
            f"That is below the ~60-80 (15-20/season) floor evaluation_dataset/README.md recommends "
            f"before treating per-season precision/recall as meaningful -- treat every number above as "
            f"directional, not a verified accuracy figure."
        )
    if not any(r.get("lighting") for r in rows):
        limitations.append("No lighting labels were provided -- the lighting-stratified breakdown is empty.")
    if not any(r.get("makeup") for r in rows):
        limitations.append("No makeup labels were provided -- the makeup-stratified breakdown is empty.")
    limitations.append("No hair/iris contrast signal is computed (see services/color_engine/extract.py's docstring) -- confidence never reaches \"high\".")
    limitations.append("Cross-validation (section 16) was not run -- this dataset is not yet large enough for stratified K-fold to mean anything.")
    if not undertone_rows:
        limitations.append("No undertone labels were provided -- the Undertone Results section reports no data yet.")

    from datetime import datetime, timezone
    repro_info = repro.collect(
        repo_dir=Path(__file__).resolve().parent.parent,
        dataset_dir=args.labels.resolve().parent,
        timestamp=datetime.now(timezone.utc).isoformat(),
        config={
            "consistency": args.consistency, "ablation": args.ablation,
            "baselines": args.baselines, "debug": bool(args.debug),
            "require_frozen": args.require_frozen,
        },
    )

    ctx = {
        "dataset": {
            "n_images": n_images_present, "n_labeled": len(labeled_rows), "n_skipped": n_skipped,
            "per_season": per_season_count, "lighting_distribution": lighting_dist, "makeup_distribution": makeup_dist,
        },
        "overall_metrics": overall_metrics,
        "confusion_matrix": metrics.confusion_matrix(labeled_rows, SEASONS),
        "per_season": per_season,
        "undertone_confusion_matrix": undertone_confusion_matrix,
        "undertone_per_class": undertone_per_class,
        "by_lighting": stratify.by_lighting(labeled_rows, SEASONS),
        "by_makeup": stratify.by_makeup(labeled_rows, SEASONS),
        "calibration_buckets": calibration.calibration_buckets(labeled_rows),
        "calibration_summary": calibration.calibration_gap_summary(calibration.calibration_buckets(labeled_rows)),
        "major_confusions": major_confusions,
        "ablation_summary": ablation_summary,
        "remaining_limitations": " ".join(limitations),
        "reproducibility_markdown": repro.format_markdown(repro_info),
    }

    report_text = report_mod.render_report(ctx)
    print("\n" + report_text)
    if args.report:
        args.report.write_text(report_text)
        print(f"\nWrote {args.report}")

    if not labeled_rows:
        print("\n*** NO LABELED SEASON DATA -- every metric above reads 'no data yet'. ***")


if __name__ == "__main__":
    main()
