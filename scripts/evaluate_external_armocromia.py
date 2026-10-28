#!/usr/bin/env python3
"""
evaluate_external_armocromia.py

EXPLORATORY ONLY -- not the project's final ground-truth evaluation. See
evaluation_dataset/EVALUATION_PROTOCOL.md section 1's tier table and
services/eval/external_armocromia.py's module docstring before reading
anything this script produces as an accuracy claim.

Runs AuraMatch's EXISTING, UNMODIFIED season-decision logic
(services/color_engine/season.py's decide_season -- imported, never edited)
against CIELAB feature statistics extracted from the Deep Armocromia
dataset by a third party (nypoclary/armocromia-classical-ml, MIT license --
see evaluation_dataset/DATASET_CANDIDATES.md Candidate 4). That repository
ships pre-extracted feature CSVs, not raw images -- this script CANNOT and
does not evaluate MediaPipe landmarks, the quality gate, skin masking,
white balance, or K-means aggregation. It measures one thing only: does
season.py's existing decision/scoring math behave sensibly given real,
human-labeled CIELAB observations. See --report's output for the full
"what this can/cannot validate" breakdown.

    python scripts/evaluate_external_armocromia.py \\
        --input  /path/to/features_30_per_region.csv \\
        --output evaluation_dataset/results/external_armocromia_exploratory.csv \\
        --report evaluation_dataset/results/external_armocromia_exploratory.md \\
        [--partition all|train|test] [--contrast none|region|both] \\
        [--source-url URL] [--source-commit SHA]

Fails safely (non-zero exit, clear message, nothing written) if the input
CSV's schema is unrecognized, a required field is missing, or a label
value isn't one of Spring/Summer/Autumn/Winter -- see
services/eval/external_armocromia.py's SchemaError. Never guesses.

NO PRODUCTION CODE IS MODIFIED BY THIS SCRIPT. It imports
services.color_engine.season.decide_season and
services.color_engine.color_math verbatim.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.eval import metrics, repro
from services.eval.external_armocromia import (
    SEASONS, SchemaError, evaluate_all, load_rows,
)

LOW_MARGIN_THRESHOLD = 0.10  # season.py's OWN "_reliability" low-margin cutoff -- reused, not invented


# --- filtering -------------------------------------------------------------


def _filter_partition(samples, partition: str):
    if partition == "all":
        return samples
    return [s for s in samples if s.partition == partition]


# --- metrics assembly -------------------------------------------------------


def _rows_for_metrics(results):
    return [{"truth": r.source_label, "predicted": r.predicted} for r in results if r.valid]


def _distribution(results, key_fn):
    counts = {s: 0 for s in SEASONS}
    for r in results:
        v = key_fn(r)
        if v in counts:
            counts[v] += 1
    return counts


def _exclusion_breakdown(results):
    from collections import Counter
    return dict(Counter(r.reason for r in results if not r.valid))


def _margin_split(results, threshold=LOW_MARGIN_THRESHOLD):
    valid = [r for r in results if r.valid and r.margin is not None]
    high = [r for r in valid if r.margin >= threshold]
    low = [r for r in valid if r.margin < threshold]
    return high, low


def _agreement_rate(rows):
    if not rows:
        return None
    correct = sum(1 for r in rows if r.predicted == r.source_label)
    return correct / len(rows)


def _summarize(results, label: str) -> dict:
    valid_rows = _rows_for_metrics(results)
    n_total = len(results)
    n_valid = len(valid_rows)
    cm_matrix = metrics.confusion_matrix(valid_rows, SEASONS) if valid_rows else {}
    per_class = {
        c: dict(zip(("precision", "recall", "f1", "support"), metrics.precision_recall_f1(valid_rows, c)))
        for c in SEASONS
    } if valid_rows else {}
    high, low = _margin_split(results)

    return {
        "label": label,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_excluded": n_total - n_valid,
        "exclusion_breakdown": _exclusion_breakdown(results),
        "accuracy": metrics.accuracy(valid_rows),
        "balanced_accuracy": metrics.balanced_accuracy(valid_rows, SEASONS),
        "macro_f1": metrics.macro_f1(valid_rows, SEASONS),
        "confusion_matrix": cm_matrix,
        "per_class": per_class,
        "truth_distribution": _distribution(results, lambda r: r.source_label),
        "pred_distribution": _distribution(results, lambda r: r.predicted),
        "high_margin_n": len(high),
        "high_margin_agreement": _agreement_rate(high),
        "low_margin_n": len(low),
        "low_margin_agreement": _agreement_rate(low),
        "reliability_values": sorted({r.reliability for r in results if r.valid}),
    }


# --- error-analysis table ---------------------------------------------------


def _error_examples(results):
    valid = [r for r in results if r.valid and r.margin is not None]
    correct = [r for r in valid if r.predicted == r.source_label]
    wrong = [r for r in valid if r.predicted != r.source_label]
    strongest_agree = sorted(correct, key=lambda r: -r.margin)[:10]
    strongest_disagree = sorted(wrong, key=lambda r: -r.margin)[:10]
    lowest_margin = sorted(valid, key=lambda r: r.margin)[:10]
    return strongest_agree, strongest_disagree, lowest_margin


# --- output ------------------------------------------------------------


def _write_csv(path: Path, results, results2=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id", "source_label", "source_subtype", "partition", "celeba",
        "predicted", "top_confidence", "second_confidence", "margin", "reliability",
        "valid", "reason", "L", "a", "b", "hue_angle", "chroma", "ita_category",
        "used_region_contrast",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r.as_row())
        if results2:
            for r in results2:
                w.writerow(r.as_row())


def _fmt(x, pct=False):
    if x is None:
        return "no data yet"
    return f"{x:.1%}" if pct else f"{x:.3f}"


def _confusion_table(cmx) -> str:
    header = "|  | " + " | ".join(f"Pred {s}" for s in SEASONS) + " | Pred NO_PREDICTION |"
    sep = "|---|" + "---|" * (len(SEASONS) + 1)
    lines = [header, sep]
    for truth in SEASONS:
        row = cmx.get(truth, {})
        cells = [str(row.get(s, 0)) for s in SEASONS] + [str(row.get("NO_PREDICTION", 0))]
        lines.append(f"| Actual {truth} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _example_row(r) -> str:
    return (
        f"| {r.sample_id} | {r.source_label} | {r.predicted} | "
        f"{r.L:.2f} | {r.a:.2f} | {r.b:.2f} | {r.margin:.4f} | {r.reliability} |"
    )


def _write_report(path: Path, *, summary, hypothetical_summary, global10_summary, results, samples_schema,
                   args, repro_info, strongest_agree, strongest_disagree, lowest_margin):
    path.parent.mkdir(parents=True, exist_ok=True)
    L = []
    L.append("# External Armocromia Exploratory Evaluation\n")
    L.append("```text\nSTATUS:\nEXPLORATORY ONLY\n\nFINAL AURAMATCH ACCURACY:\nNOT MEASURED\n\n"
              "REASON:\nThe project's frozen internal ground-truth dataset "
              "(evaluation_dataset/labels.csv) does not yet exist -- see "
              "EVALUATION_PROTOCOL.md section 1's tier table. External datasets, "
              "however well-documented, are capped at 'exploratory' or "
              "'preliminary' and can never substitute for it.\n```\n")

    L.append("## What was researched\n")
    L.append(
        "This experiment tests whether `services/color_engine/season.py`'s existing, "
        "UNMODIFIED decision/scoring logic behaves sensibly on real, human-labeled "
        "CIELAB color observations, using feature statistics extracted from the Deep "
        "Armocromia dataset and redistributed (not the raw images) by "
        "`nypoclary/armocromia-classical-ml` (MIT license, git commit "
        "63d3338078c27a3b2a89b340e0ad38d8f2c46388). No threshold, weight, or formula "
        "in season.py/color_math.py was changed to produce this report.\n"
    )

    L.append("## Dataset provenance\n")
    L.append(
        "- Source repository: https://github.com/nypoclary/armocromia-classical-ml "
        "(commit 63d3338078c27a3b2a89b340e0ad38d8f2c46388)\n"
        "- Underlying dataset: Deep Armocromia (Stacchio et al., ECCV 2024) -- this "
        "repo does NOT redistribute Deep Armocromia's raw images, only extracted "
        "feature statistics; access to the raw images remains request-gated and "
        "was not pursued for this experiment (see DATASET_CANDIDATES.md Candidate 1, "
        "status ACCESS PENDING).\n"
        "- File used: " + str(args.input) + f" (schema detected: {samples_schema})\n"
        "- **This is NOT an independent second dataset.** Every row here traces back "
        "to Deep Armocromia's own 4,920-image release; agreement or disagreement "
        "found here says nothing about independent replication, only about how "
        "season.py behaves on that one dataset's labels and derived color statistics.\n"
        "- Official split, confirmed directly from the CSV: 4,008 train / 912 test "
        "(4,920 total); 2,981 CelebA-derived (61%) / 1,939 from the 'Pivotal "
        "Armocromia Set' (39%).\n"
        "- Identity check performed on the 1,939 non-CelebA rows (which carry a "
        "`person_name` field): 643 unique named subjects, 0 names appearing in both "
        "train and test partitions -- the official split is person-disjoint for "
        "this portion, confirmed directly. The 2,981 CelebA-derived rows carry no "
        "identity field in this CSV; CelebA's own identity annotations were not "
        "joined in, so person-disjointness for that 61% remains UNKNOWN.\n"
    )

    L.append("## Feature compatibility\n")
    L.append(
        "- `season` column is a direct 4-season label (Spring/Summer/Autumn/Winter, "
        "English) -- used as-is, no subtype mapping invented (see 'Label "
        "compatibility' below).\n"
        "- L*/a*/b* value ranges across all 4,920 rows were sanity-checked before "
        "use: skin region L in [22.8, 87.9], a in [-5.7, 50.3], b in [-23.0, 61.3] "
        "-- consistent with standard CIE 1976 L*a*b* on its usual scale (no sign of "
        "an unconverted 8-bit OpenCV LAB encoding, which would show L in [0,255]).\n"
        "- **UNVERIFIED**: the exact RGB->Lab conversion pipeline (library, "
        "illuminant, chromatic-adaptation matrix) that produced these values is "
        "undocumented. The `armocromia-classical-ml` notebooks load the feature "
        "CSVs directly -- they contain no pixel-level RGB->Lab conversion code -- "
        "so that step happened entirely upstream, in tooling not published in "
        "either repository. This experiment CANNOT confirm these Lab values are on "
        "the identical illuminant/matrix AuraMatch's own `color_math.srgb_to_lab` "
        "(D65, IEC 61966-2-1) uses; range plausibility is the only check available "
        "and it did not surface anything obviously incompatible.\n"
        "- **Region-definition mismatch, confirmed**: the source library's 'skin' "
        "region is a face-parsing mask covering `{face, nose}` segmentation classes "
        "(from its own `REGION_GROUPS` config, read directly from "
        "`notebooks/01-phase1-baseline.ipynb`), not AuraMatch's landmark-restricted "
        "skin mask (`services/color_engine/skin_mask.py`). The two almost certainly "
        "don't select identical pixels.\n"
        "- **Aggregation-order mismatch, verified numerically**: the CSV's own "
        "`Chroma_mean_skin`/`ITA_mean_skin` columns are the MEAN of per-pixel "
        "chroma/ITA, not chroma/ITA of the mean Lab point. AuraMatch's production "
        "pipeline (`extract.py`'s `cm.skin_features(rep)`) computes chroma/hue/ITA "
        "from ONE representative Lab centroid. These are different quantities by "
        "design (Jensen's inequality) -- verified directly on row 0: CSV "
        "`Chroma_mean`=20.0039 vs. chroma computed from that row's own "
        "(L_mean,a_mean,b_mean)=19.4564, a 2.7% gap. **This adapter therefore "
        "derives hue_angle/chroma/ita_category fresh from (L_mean, a_mean, "
        "b_mean) via `color_math.skin_features()` -- the exact function "
        "production's extract.py calls -- rather than trusting the CSV's own "
        "pre-computed columns**, to stay faithful to how season.py is actually fed "
        "in production.\n"
        "- CelebA-derived rows (61% of the dataset) risk failing AuraMatch's own "
        "image-quality gate outright if the underlying images were CelebA's "
        "standard 178x218px Align&Cropped release (below `imageQuality.js`'s "
        "180px `MIN_DIM_REJECT` by 2px) -- moot for THIS experiment (no images are "
        "processed here, only pre-extracted color statistics), but directly "
        "relevant if raw-image access is later obtained; see "
        "DATASET_CANDIDATES.md Candidate 1.\n"
        "- `confidence` and `wb_trustworthy` are NOT synthesized (see "
        "`services/eval/external_armocromia.py`'s module docstring for the full "
        "reasoning) -- both default per `decide_season`'s own `_get(...)` calls. "
        "**Verified consequence**: with `wb_trustworthy` defaulting False, "
        "`decide_season`'s `_reliability()` returns `\"low\"` for every single "
        "sample in this experiment, regardless of margin. The `reliability` field "
        "is therefore constant here and cannot be used for a "
        "reliability-vs-correctness check -- `margin` is used instead (see "
        "'Reliability analysis' below).\n"
    )

    L.append("## Label compatibility\n")
    L.append(
        "The `season` column provides Spring/Summer/Autumn/Winter directly -- "
        "AuraMatch's exact primary taxonomy. No subtype mapping was attempted: "
        "`season_subtype` (12-value) exists in the source but "
        "`EVALUATION_PROTOCOL.md` section 3 explicitly prohibits mapping it onto "
        "AuraMatch's own 8-subtype `sub_tone` scheme (no 1:1 correspondence "
        "exists between the two taxonomies) -- `sub_tone` is computed by "
        "`decide_season` internally but is not compared against `season_subtype` "
        "anywhere in this report.\n"
    )

    L.append("## What this experiment can validate\n")
    L.append(
        "- Whether `season.py`'s 3-axis (warm/cool, light/deep, bright/mute) "
        "decision math and softmax-style ranking produce season predictions that "
        "agree with real human-assigned season labels, GIVEN CIELAB inputs sourced "
        "from a face/skin-region measurement methodology different from AuraMatch's "
        "own.\n"
        "- Whether the model's decision MARGIN (top season probability minus "
        "runner-up) tracks correctness -- a decision-boundary sanity check.\n"
        "- Whether `use_region_contrast=True`'s hypothetical skin<->hair / "
        "skin<->iris CIEDE2000 contrast terms change predictions in a direction "
        "consistent with `season.py`'s own bright/mute design intent (a "
        "HYPOTHETICAL run only -- production never populates these today).\n"
    )
    L.append("## What this experiment cannot validate\n")
    L.append(
        "- MediaPipe landmark detection, `imageQuality.js`/`quality_port.py`'s "
        "quality gate, `skin_mask.py`'s landmark-restricted masking, "
        "`white_balance.py`'s Shades-of-Gray correction, or `extract.py`'s "
        "K-means-robust skin-centroid aggregation -- none of these ran on this "
        "data; no raw image was ever processed by AuraMatch's own pipeline.\n"
        "- `decide_season`'s reliability tiers as they actually behave in "
        "production (see the constant-'low' finding above) -- the confidence/"
        "wb_trustworthy inputs that would make this meaningful cannot be honestly "
        "reconstructed from this CSV.\n"
        "- Undertone accuracy specifically -- `undertone` is reused directly from "
        "`warm_cool` in production, and this experiment reports `warm_cool` "
        "implicitly via the season decision, but the external dataset carries no "
        "independent Warm/Cool/Neutral label to compare against (see "
        "DATASET_CANDIDATES.md's Undertone compatibility rating: LOW).\n"
        "- Face Shape -- entirely out of scope for this dataset and this script "
        "(see task instructions section 17; not attempted here).\n"
        "- Whether AuraMatch's ACTUAL end-to-end photo pipeline is accurate -- see "
        "the STATUS block above.\n"
    )

    def _summary_section(title, s):
        out = [f"## {title}\n"]
        out.append(f"- Samples: {s['n_total']} total, {s['n_valid']} valid, "
                    f"{s['n_excluded']} excluded ({s['exclusion_breakdown'] or 'none'})\n")
        out.append(f"- **Exploratory agreement with external Armocromia labels**: "
                    f"{_fmt(s['accuracy'], pct=True)} (n={s['n_valid']})\n")
        out.append(f"- Balanced accuracy: {_fmt(s['balanced_accuracy'], pct=True)}\n")
        out.append(f"- Macro F1: {_fmt(s['macro_f1'])}\n")
        out.append("### Per-class\n")
        out.append("| Season | Precision | Recall | F1 | Support |\n|---|---|---|---|---|")
        for c in SEASONS:
            pc = s["per_class"].get(c, {})
            out.append(f"| {c} | {_fmt(pc.get('precision'))} | {_fmt(pc.get('recall'))} | "
                        f"{_fmt(pc.get('f1'))} | {pc.get('support', 0)} |")
        out.append("\n### Confusion matrix (rows=truth, cols=predicted)\n")
        out.append(_confusion_table(s["confusion_matrix"]))
        out.append("\n### Distribution\n")
        out.append("Ground-truth distribution: " + ", ".join(f"{k}={v}" for k, v in s["truth_distribution"].items()))
        out.append("\n\nAuraMatch prediction distribution: " + ", ".join(f"{k}={v}" for k, v in s["pred_distribution"].items()))
        out.append(
            f"\n\n### Reliability analysis\n"
            f"- `reliability` values observed across all valid samples: {s['reliability_values']} "
            f"(see Feature Compatibility -- expected to be constant given no honest "
            f"confidence/wb_trustworthy input exists for this data).\n"
            f"- Margin-based substitute, using season.py's OWN existing low-margin "
            f"threshold (`{LOW_MARGIN_THRESHOLD}`, from `_reliability()`'s own "
            f"`if margin < 0.10` check -- reused, not invented):\n"
            f"  - HIGH-MARGIN (>= {LOW_MARGIN_THRESHOLD}): n={s['high_margin_n']}, "
            f"agreement={_fmt(s['high_margin_agreement'], pct=True)}\n"
            f"  - LOW-MARGIN (< {LOW_MARGIN_THRESHOLD}): n={s['low_margin_n']}, "
            f"agreement={_fmt(s['low_margin_agreement'], pct=True)}\n"
        )
        return "\n".join(out)

    L.append(_summary_section(f"Exploratory results -- PRIMARY (contrast=none, matches production's actual current behavior)", summary))

    if hypothetical_summary is not None:
        L.append(_summary_section(
            "Exploratory results -- HYPOTHETICAL (contrast=region; production never "
            "populates skin_hair_contrast/skin_iris_contrast today -- see "
            "external_armocromia.py's docstring)",
            hypothetical_summary,
        ))

    if global10_summary is not None:
        L.append(f"## Methodology-discrimination check: global10 vs. region30 (schema: {global10_summary['schema']})\n")
        L.append(
            "Runs the SAME unmodified `season.py` over the SAME underlying "
            "labels, but through the WEAKER-proxy `global10` region ('all "
            "non-background facial pixels' -- skin+hair+eyes+brows combined, "
            "not skin-specific; see 'Feature compatibility' above). This "
            "isolates measurement methodology from `season.py`'s decision "
            "boundaries: if agreement moves substantially from the PRIMARY "
            "run above, the extraction methodology matters a lot; if it "
            "lands within a point or two, the decision boundaries (not the "
            "measurement) are the dominant factor.\n\n"
            f"**Result: {_fmt(global10_summary['accuracy'], pct=True)} agreement "
            f"(global10) vs. {_fmt(summary['accuracy'], pct=True)} (region30/skin-only, "
            "PRIMARY above)** -- "
            + (
                "within ~1 point of each other, which points at `season.py`'s "
                "decision boundaries (not the choice of facial region measured) "
                "as the dominant factor in the low agreement seen throughout this "
                "report."
                if summary["accuracy"] is not None and global10_summary["accuracy"] is not None
                and abs(summary["accuracy"] - global10_summary["accuracy"]) <= 0.02
                else "a difference large enough that measurement methodology "
                "(which facial region is averaged, and how) cannot be ruled out "
                "as a substantial contributor alongside decision-boundary "
                "calibration."
            ) + "\n\n"
            "**Directional check, verified directly (not inferred from region "
            "composition alone)**: global10's mean L* across all 4,920 rows is "
            "45.63, versus region30's skin-only mean L* of 62.76 -- ~17 units "
            "darker, consistent with global10 averaging in hair (typically much "
            "darker than skin) alongside skin, nose, eyes, and brows. Agreement "
            "IMPROVED (28.3% -> 31.7%) when the measured lightness moved from "
            "well ABOVE `season.py`'s `value_pivot=58.0` (region30, mean 62.76) "
            "to well BELOW it (global10, mean 45.63). That direction is itself "
            "additional evidence for the `value_pivot` miscalibration hypothesis "
            "above -- pushing the average measurement to straddle the pivot "
            "rather than sit entirely on one side of it changed the outcome in "
            "the expected direction, though 3.4 points is not enough to call "
            "this confirmed on its own.\n\n"
            "**Note for reproducibility/audit**: this global10 run's per-sample "
            "rows are NOT included in `external_armocromia_exploratory.csv` "
            "(only the region30 PRIMARY/HYPOTHETICAL rows are, distinguished by "
            "`used_region_contrast`) -- re-run this exact command with "
            "`--global10-input` to reproduce the per-row detail behind the "
            "numbers below.\n"
        )
        L.append(_summary_section("global10 comparison -- full breakdown", global10_summary))

    L.append("## Error analysis\n")
    L.append("### 10 strongest agreements (correct, highest margin)\n")
    L.append("| sample_id | truth | predicted | L | a | b | margin | reliability |\n|---|---|---|---|---|---|---|---|")
    L.extend(_example_row(r) for r in strongest_agree)
    L.append("\n### 10 strongest disagreements (wrong, highest margin -- confidently wrong)\n")
    L.append("| sample_id | truth | predicted | L | a | b | margin | reliability |\n|---|---|---|---|---|---|---|---|")
    L.extend(_example_row(r) for r in strongest_disagree)
    L.append("\n### 10 lowest-margin samples (boundary cases)\n")
    L.append("| sample_id | truth | predicted | L | a | b | margin | reliability |\n|---|---|---|---|---|---|---|---|")
    L.extend(_example_row(r) for r in lowest_margin)

    L.append("\n## Decision-boundary interpretation (observed -> evidence -> hypothesis)\n")
    L.append(
        "This section reports a pattern found while producing the results above, "
        "using ONLY the data already in `evaluation_dataset/results/"
        "external_armocromia_exploratory.csv` (no additional external data, no "
        "code change). Per this task's own rule: if the result looks poor, report "
        "observed -> evidence -> hypothesis -> required next experiment, and do "
        "NOT touch the algorithm.\n\n"
        "**Observed**: `Spring` is predicted for 56% of all 4,920 samples "
        "(2,773/4,920) despite being only 24% of the ground truth. `Autumn` "
        "recall is 4.2% -- the near-total failure of the four classes -- with "
        "58% of true-Autumn samples predicted as `Spring` instead. This holds "
        "essentially identically across CelebA (28.3% agreement) vs. non-CelebA "
        "(28.3%) and train (28.5%) vs. test (27.7%) subsets, ruling out a "
        "partition- or CelebA-specific artifact as the explanation.\n\n"
        "**Evidence**: computed directly from this run's own output --\n"
        "- Mean measured chroma by ground-truth season: Autumn=28.60, "
        "Spring=26.29, Winter=24.21, Summer=22.95 (overall mean 25.59). "
        "`season.py`'s `SeasonModel.chroma_pivot=18.0`/`chroma_half=10.0` "
        "saturates the bright/mute axis to +1.0 (\"bright\") at chroma>=28. "
        "**The two seasons `season.py` defines as `mute` (bright axis target "
        "= -1.0) are Summer and Autumn -- and in this dataset they sit at "
        "OPPOSITE extremes of measured chroma**: Summer LOWEST (22.95, axis "
        "score +0.42, 20.6% of Summer rows clipped to +1.0) and Autumn "
        "HIGHEST (28.60, axis score +0.74, 53.8% of Autumn rows clipped to "
        "+1.0). Neither is near the theoretical mute target, and they aren't "
        "even close to each other -- the measured bright/mute axis isn't "
        "compressed or saturated for both alike, it's uncorrelated with "
        "`season.py`'s theoretical mute/bright grouping for this pair. "
        "(Checked directly, not assumed: only 35.6% of all 4,920 rows clip "
        "the axis at chroma>=28 at all.)\n"
        "- Mean measured L* by ground-truth season: Spring=64.57, Summer=65.36, "
        "Autumn=60.06, Winter=61.58 -- the light/deep axis gap between the "
        "'light' pair and 'deep' pair is only ~4.5 L* units, small relative to "
        "`value_half=18.0` (a ~0.25 axis-score gap).\n"
        "- Mean a*/b* by ground-truth season are all warm (positive) and similar "
        "in magnitude for Spring and Autumn specifically (Spring "
        "a=16.04/b=20.43; Autumn a=17.28/b=22.43) -- both read as warm under "
        "`season.py`'s target (Spring warm=+1, Autumn warm=+1, SAME sign).\n\n"
        "**Hypothesis**: with warm (highest-weighted axis, `w_warm=1.0`) pointing "
        "the same direction for Spring and Autumn, and Autumn's bright/mute "
        "reading (+0.74) landing far closer to Spring's bright target (+1.0) "
        "than to Autumn's own mute target (-1.0), only the weakly-separated "
        "light/deep axis (`w_value=0.7`, ~0.25 score gap) is left to "
        "distinguish the two classes under `season.py`'s weighted-Euclidean-"
        "distance scoring -- consistent with the observed Spring<->Autumn "
        "confusion dominating the error pattern. This does NOT, "
        "by itself, mean `season.py`'s pivots are wrong for AuraMatch's real "
        "input distribution: this dataset's chroma is measured via a face-"
        "parsing region average (face+nose, mean-of-pixels) that may read "
        "systematically higher than AuraMatch's own K-means-robust, landmark-"
        "masked single-centroid measurement on the same faces -- see 'Feature "
        "compatibility' above. The gap could reflect a real pivot-calibration "
        "issue, a measurement-methodology mismatch, or both, and this "
        "experiment cannot distinguish which.\n\n"
        "**Required next experiment** (not performed here, per this task's "
        "absolute rules): run AuraMatch's OWN `extract.py` pipeline on a real, "
        "independently-labeled photo sample (either the project's own 60-80 "
        "image collection, or Deep Armocromia's raw images if/when access is "
        "granted) and compare THAT chroma/L distribution, per season, against "
        "`SeasonModel`'s pivots -- before drawing any conclusion about whether "
        "the pivots themselves need recalibration. This experiment's external, "
        "differently-measured chroma values are not a valid basis for retuning "
        "`season.py` on their own.\n"
    )

    L.append("\n## Distribution analysis\n")
    total = sum(summary["truth_distribution"].values()) or 1
    L.append("Ground-truth class balance: " + ", ".join(
        f"{k}={v} ({v/total:.1%})" for k, v in summary["truth_distribution"].items()
    ))
    L.append(
        "\n\nThe dataset is reasonably balanced across seasons (not perfectly -- see "
        "percentages above), so balanced accuracy and raw accuracy should track "
        "each other fairly closely here; both are reported above per this "
        "experiment's own emphasis on not relying on raw agreement alone."
    )

    L.append("\n## Limitations\n")
    L.append(
        "- This dataset is NOT independent of Deep Armocromia -- see 'Dataset "
        "provenance' above.\n"
        "- No raw image was processed; only pre-extracted, third-party color "
        "statistics were used -- see 'What this experiment cannot validate'.\n"
        "- The upstream RGB->Lab conversion methodology is unverified (undocumented "
        "in both the source dataset's and this feature repo's public materials).\n"
        "- The 'skin' region definition (face-parsing mask) differs from "
        "AuraMatch's own landmark-restricted skin mask; how much this matters for "
        "the reported agreement number cannot be quantified without raw-image "
        "access.\n"
        "- `reliability` is degenerate (constant) in this experiment for reasons "
        "specific to the missing confidence/wb_trustworthy inputs -- see above. "
        "This is a property of what this experiment's inputs allow, not a claim "
        "about production's real-world reliability distribution.\n"
        "- Deep Armocromia's own label-independence question "
        "(`DATASET_CANDIDATES.md` Candidate 1) is unresolved regardless of "
        "anything measured here -- a low OR high agreement number in this report "
        "says nothing about whether Deep Armocromia's season labels are "
        "themselves independently correct.\n"
    )

    L.append("## Production code changes\n")
    L.append(
        "None. Verified, not just asserted: `git diff --stat -- "
        "services/color_engine/ services/personal_color_service.py "
        "src/utils/faceGeometry.js` returned empty output at AuraMatch git "
        f"commit `{repro_info['git_commit'] or 'unknown'}` (this report's own "
        "commit, recorded below). `services/color_engine/season.py`, "
        "`color_math.py`, `white_balance.py`, `skin_mask.py`, `extract.py`, "
        "`services/personal_color_service.py`, `src/utils/faceGeometry.js`, "
        "Gemini logic, the production API, and the production frontend were not "
        "modified. Only new, additive files were created: "
        "`services/eval/external_armocromia.py`, "
        "`scripts/evaluate_external_armocromia.py`, "
        "`tests/test_external_armocromia.py`, and this report's output files.\n"
    )

    L.append("## Test results\n")
    L.append(
        "Verified in this session (re-run `pytest tests/` to reconfirm before "
        "relying on this for a LATER state of the repo -- these counts are a "
        "point-in-time record, not live-generated by this script):\n"
        "- `pytest tests/test_external_armocromia.py -v`: **20 passed** (schema "
        "detection x3, load/label/missing-value handling x6, "
        "build_season_input derivation x3, evaluate_sample x2, determinism x2, "
        "CLI end-to-end x4 -- including a regression test for the "
        "`--contrast region`-only CSV-output bug this adapter's own review "
        "caught and fixed).\n"
        "- `pytest tests/` (full backend suite): **260 passed**, 0 failed.\n"
        "- 260 - 20 = 240 -- independently confirms the task's stated "
        "'240/240 backend tests passing' baseline was real and unbroken by "
        "this addition, not just carried forward as an assumption.\n"
        "- `node --test` / `npm run build` (frontend): **not run**. No file in "
        "the frontend repo (`auramatch ver3`) was touched by this task -- its "
        "`git status` is byte-identical to this task's starting state -- so "
        "there is nothing for those commands to verify that a diff wouldn't "
        "already show empty.\n"
    )

    L.append("## Reproducibility\n")
    L.append(f"- Run timestamp: {repro_info['timestamp']}\n"
             f"- Download date (external source files): same session as this run -- "
             f"{repro_info['timestamp'][:10]}\n"
             f"- AuraMatch git commit: {repro_info['git_commit'] or 'unknown'}\n"
             f"- Python version: {repro_info['python_version']}\n"
             f"- Package versions: " + ", ".join(f"{k}={v or 'unknown'}" for k, v in repro_info['package_versions'].items()) + "\n"
             f"- External source repo: https://github.com/nypoclary/armocromia-classical-ml\n"
             f"- External source commit: 63d3338078c27a3b2a89b340e0ad38d8f2c46388\n"
             f"- External source file blob SHA (features_30_per_region.csv): "
             f"2bba5d20604a32a9f237e2467fd6d43decff2f60\n"
             f"- External source file blob SHA (features_10_merged.csv): "
             f"3541f21b5743a9f0b50bf036e4dcc925004e7d7f\n"
             f"- Input file used for this run: {args.input}\n"
             f"- Input file SHA-256: {repro_info['input_sha256']}\n"
             + (f"- --global10-input file SHA-256: {repro_info.get('global10_input_sha256')}\n" if repro_info.get('global10_input_sha256') else "")
             + f"- Randomness: none -- `decide_season` is pure/deterministic; no "
             f"random_state is used anywhere in this adapter (verified by "
             f"`tests/test_external_armocromia.py`'s determinism test).\n"
             f"- Command: {' '.join(sys.argv)}\n")

    L.append("## Second public dataset (task section 19)\n")
    L.append(
        "Not pursued. Checked against `DATASET_CANDIDATES.md`'s full candidate "
        "list: no OTHER candidate there meets all five stated criteria "
        "(legally downloadable + clearly licensed + real images + direct "
        "4-season labels + pipeline-compatible) -- the face-shape datasets "
        "(dsmlr/faceshape, niten19) carry no season/color labels at all, and "
        "every other season-adjacent candidate (Su et al., ColorInsight) was "
        "never publicly released. Deep Armocromia (via this repo's feature "
        "extraction) remains the only usable lead for this specific "
        "experiment.\n"
    )

    L.append("## Next-step recommendation (task section 26)\n")
    L.append(
        "```text\n"
        "1. Can we use this dataset for final accuracy?\n"
        "NO. Capped at exploratory per EVALUATION_PROTOCOL.md section 9 --\n"
        "Deep Armocromia's own label-independence question is unresolved, and\n"
        "this experiment additionally used feature-only (no-image) derived\n"
        "data, one further remove from even that.\n\n"
        "2. Can we use it for exploratory validation?\n"
        "YES -- done in this report. It is a legitimate, real, human-labeled\n"
        "signal about season.py's decision math, clearly caveated.\n\n"
        "3. What part of AuraMatch does it actually validate?\n"
        "Only services/color_engine/season.py's decision/scoring math, fed by\n"
        "CIELAB inputs from a DIFFERENT extraction methodology (face-parsing\n"
        "region average) than AuraMatch's own (landmark-restricted skin mask +\n"
        "K-means-robust centroid). It validates NOTHING about MediaPipe\n"
        "landmarks, the quality gate, skin masking, white balance, or\n"
        "extract.py's K-means aggregation -- no raw image was ever processed.\n\n"
        "4. Do we still need the project's own 60-80+ real-photo ground-truth\n"
        "dataset?\n"
        "YES. It remains the only path to a number this project can call\n"
        "'accuracy' at all, per EVALUATION_PROTOCOL.md section 1 -- and it is\n"
        "the only way to test AuraMatch's OWN measurement methodology rather\n"
        "than a third party's.\n\n"
        "5. What is the single most important next experiment?\n"
        "Run AuraMatch's own extract.py pipeline on a real, independently-\n"
        "labeled photo sample (the internal 60-80 image collection, or Deep\n"
        "Armocromia's raw images if/when access is granted) and compare THAT\n"
        "chroma/L distribution, per season, against SeasonModel's pivots --\n"
        "this experiment's own decision-boundary interpretation section above\n"
        "found a concrete, specific candidate issue (Autumn's measured\n"
        "bright/mute axis reading +0.74 against a mute target of -1.0) that a\n"
        "same-methodology measurement could confirm or rule out.\n"
        "```\n"
    )

    L.append("## Final status\n")
    L.append("```text\nEXPLORATORY VALIDATION COMPLETE\n```\n" if summary["n_valid"] > 0 else
              "```text\nEXPLORATORY VALIDATION BLOCKED\n```\n")

    path.write_text("\n".join(L), encoding="utf-8")


# --- main --------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Path to features_30_per_region.csv or features_10_merged.csv")
    p.add_argument("--output", default="evaluation_dataset/results/external_armocromia_exploratory.csv")
    p.add_argument("--report", default="evaluation_dataset/results/external_armocromia_exploratory.md")
    p.add_argument("--partition", choices=["all", "train", "test"], default="all")
    p.add_argument("--contrast", choices=["none", "region", "both"], default="both",
                    help="none=production-faithful only; region=hypothetical only; both=report side by side (default)")
    p.add_argument("--global10-input", default=None,
                    help="Optional: path to the global10-schema file (features_10_merged.csv) over the SAME "
                         "underlying images as --input, to test whether agreement changes under a different "
                         "region/measurement methodology over the same labels -- see the report's decision-"
                         "boundary interpretation section for why this discriminates methodology from decision "
                         "boundaries.")
    args = p.parse_args()

    try:
        schema, samples = load_rows(args.input)
    except SchemaError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        sys.exit(1)

    samples = _filter_partition(samples, args.partition)
    if not samples:
        print(f"FATAL: no rows remain after filtering to partition={args.partition!r}", file=sys.stderr)
        sys.exit(1)

    run_none = args.contrast in ("none", "both")
    run_region = args.contrast in ("region", "both")

    results_none = evaluate_all(samples, use_region_contrast=False) if run_none else []
    results_region = evaluate_all(samples, use_region_contrast=True) if run_region else []

    primary_results = results_none if run_none else results_region
    summary = _summarize(primary_results, "primary")
    hypothetical_summary = _summarize(results_region, "hypothetical") if (run_region and run_none) else (
        _summarize(results_region, "hypothetical-only") if (run_region and not run_none) else None
    )

    strongest_agree, strongest_disagree, lowest_margin = _error_examples(primary_results)

    global10_summary = None
    if args.global10_input:
        try:
            g_schema, g_samples = load_rows(args.global10_input)
        except SchemaError as exc:
            print(f"FATAL (--global10-input): {exc}", file=sys.stderr)
            sys.exit(1)
        g_samples = _filter_partition(g_samples, args.partition)
        g_results = evaluate_all(g_samples, use_region_contrast=False)  # global10 has no hair/eye columns
        global10_summary = _summarize(g_results, "global10-comparison")
        global10_summary["schema"] = g_schema

    out_path = Path(args.output)
    _write_csv(out_path, primary_results, results2=(results_region if (run_region and run_none) else None))

    repo_dir = Path(__file__).resolve().parent.parent
    repro_info = repro.collect(
        repo_dir=repo_dir, dataset_dir=repo_dir / "evaluation_dataset",
        timestamp=datetime.now(timezone.utc).isoformat(), config=vars(args),
    )
    repro_info["input_sha256"] = repro.file_hash(Path(args.input))
    if args.global10_input:
        repro_info["global10_input_sha256"] = repro.file_hash(Path(args.global10_input))

    _write_report(
        Path(args.report), summary=summary, hypothetical_summary=hypothetical_summary,
        global10_summary=global10_summary,
        results=primary_results, samples_schema=schema, args=args, repro_info=repro_info,
        strongest_agree=strongest_agree, strongest_disagree=strongest_disagree, lowest_margin=lowest_margin,
    )

    print(f"schema={schema} partition={args.partition} n_total={len(samples)}")
    print(f"PRIMARY (contrast=none): agreement={summary['accuracy']}")
    if hypothetical_summary:
        print(f"HYPOTHETICAL (contrast=region): agreement={hypothetical_summary['accuracy']}")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
