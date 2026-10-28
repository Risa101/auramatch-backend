"""Confidence-calibration analysis (audit real-photo brief, section 12).

Buckets predictions by reported confidence and checks whether that number
actually tracks empirical correctness. This is diagnostic only -- per the
brief, a poorly calibrated confidence must NOT be "fixed" by multiplying it
by a constant; it needs real labeled data to recalibrate the underlying
season-scoring temperature (see services/color_engine/season.py's
`SeasonModel.temperature`).
"""

from __future__ import annotations

DEFAULT_BUCKETS = [(0.0, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]


def calibration_buckets(rows, buckets=DEFAULT_BUCKETS) -> list[dict]:
    """rows need "confidence" (float or None) and correctness derivable from
    "truth"/"predicted". Rows with confidence=None are excluded (nothing to
    calibrate) and reported separately by the caller if desired.
    """
    out = []
    for lo, hi in buckets:
        bucket_rows = [r for r in rows if r.get("confidence") is not None and lo <= r["confidence"] < hi]
        n = len(bucket_rows)
        correct = sum(1 for r in bucket_rows if r["predicted"] == r["truth"])
        out.append({
            "range": f"{lo:.2f}-{hi:.2f}",
            "n": n,
            "empirical_accuracy": (correct / n) if n else None,
            "mean_reported_confidence": (
                sum(r["confidence"] for r in bucket_rows) / n if n else None
            ),
        })
    return out


def calibration_gap_summary(buckets: list[dict]) -> str:
    """One-line, honest summary -- never claims calibration is good/bad from
    too little data.
    """
    usable = [b for b in buckets if b["n"] >= 5]
    if not usable:
        return "Insufficient data in every confidence bucket (need >= 5 samples per bucket) to say anything about calibration."
    gaps = [abs(b["empirical_accuracy"] - b["mean_reported_confidence"]) for b in usable]
    max_gap = max(gaps)
    if max_gap > 0.15:
        return f"Confidence is poorly calibrated in at least one bucket (gap {max_gap:.2f}) -- recommend recalibrating season.py's SeasonModel.temperature against this labeled set, not scaling the number."
    return f"Confidence tracks empirical accuracy reasonably well so far (max gap {max_gap:.2f}) -- still only as reliable as the {sum(b['n'] for b in usable)} samples behind it."
