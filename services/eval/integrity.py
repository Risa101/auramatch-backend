"""Dataset integrity checks: duplicate detection, landmark-export
validation, and split-assignment leakage detection.

Explicitly NOT identity/face recognition (collection-workflow brief,
section 17: "the goal is dataset quality, not identity identification").
Near-duplicate detection here is a plain average-hash (aHash) over
downsampled pixel intensity -- it flags "this looks like the same shot,
possibly resized/recompressed," never "this is the same person as that
other, differently-posed photo." A perceptual-hash match is a prompt for a
HUMAN to look and decide, never an automatic delete (section 17: "if
near-duplicate detection is uncertain, flag for manual review instead of
automatically deleting" -- this module only ever flags).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from services.eval.dataset_io import IMAGE_EXTS


# --- exact duplicates (file content hash) ------------------------------


def hash_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_exact_duplicates(images_dir: Path) -> dict[str, list[str]]:
    """Returns {sha256: [filenames]} for every hash shared by 2+ files.
    A truly unique file never appears in the result.
    """
    by_hash = defaultdict(list)
    for p in sorted(Path(images_dir).iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            by_hash[hash_file(p)].append(p.name)
    return {h: names for h, names in by_hash.items() if len(names) > 1}


# --- near duplicates (average hash / aHash) -----------------------------


def average_hash(image_bgr, hash_size: int = 8) -> int:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean = small.mean()
    bits = (small > mean).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def find_near_duplicates(images_dir: Path, threshold: int = 5) -> list[tuple]:
    """Returns [(filename_a, filename_b, hamming_distance), ...] for every
    pair whose aHash differs by <= threshold bits (of 64) -- a coarse,
    fast, "probably near-identical" signal, not a certainty. O(n^2) pair
    comparison; fine for the dataset sizes this project targets (tens to
    low hundreds of images per evaluation_dataset/README.md), not intended
    to scale to a large corpus.
    """
    hashes = {}
    for p in sorted(Path(images_dir).iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            img = cv2.imread(str(p))
            if img is not None:
                hashes[p.name] = average_hash(img)

    names = sorted(hashes)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = hamming_distance(hashes[names[i]], hashes[names[j]])
            if d <= threshold:
                pairs.append((names[i], names[j], d))
    return pairs


# --- duplicate / colliding filenames -------------------------------------


def find_filename_collisions(images_dir: Path) -> list[list[str]]:
    """Detects two files whose name is identical except for case or
    extension (e.g. 'AM-EVAL-0001.jpg' and 'AM-EVAL-0001.JPG', or
    '.jpg' vs '.jpeg' for the same stem) -- ambiguous for
    services/eval/dataset_io.py's find_image(), which matches by stem.
    """
    by_stem_lower = defaultdict(list)
    for p in sorted(Path(images_dir).iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            by_stem_lower[p.stem.lower()].append(p.name)
    return [names for names in by_stem_lower.values() if len(names) > 1]


# --- landmark export validation ------------------------------------------


def check_landmark(landmarks_dir: Path, image_id: str) -> dict:
    """Classifies one image_id's landmark export per the collection-
    workflow brief's section 14 states: SUCCESS, NO_FACE, MULTIPLE_FACES,
    INVALID (malformed JSON / missing required keys), or MISSING (no
    export file at all -- a distinct state the brief's list didn't name
    but the validator has to represent, since "no landmarks were even
    exported yet" is not the same claim as "a face-less photo was
    exported").
    """
    path = Path(landmarks_dir) / f"{Path(image_id).stem}.landmarks.json"
    if not path.exists():
        return {"image_id": image_id, "state": "MISSING", "detail": f"no landmarks file at {path}"}

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return {"image_id": image_id, "state": "INVALID", "detail": f"malformed JSON: {exc}"}

    if not isinstance(data, dict) or "faceDetected" not in data:
        return {"image_id": image_id, "state": "INVALID", "detail": "missing required 'faceDetected' field"}

    if not data["faceDetected"]:
        return {"image_id": image_id, "state": "NO_FACE", "detail": data.get("reason", "no face detected during export")}

    face_count = data.get("faceCount", 1)
    if face_count and face_count > 1:
        return {"image_id": image_id, "state": "MULTIPLE_FACES", "detail": f"faceCount={face_count}"}

    landmarks = data.get("landmarks")
    if not isinstance(landmarks, list) or len(landmarks) < 468:
        return {
            "image_id": image_id, "state": "INVALID",
            "detail": f"expected >= 468 landmarks, got {len(landmarks) if isinstance(landmarks, list) else type(landmarks).__name__}",
        }

    return {"image_id": image_id, "state": "SUCCESS", "detail": None}


def check_all_landmarks(image_ids, landmarks_dir: Path) -> list[dict]:
    return [check_landmark(landmarks_dir, image_id) for image_id in image_ids]


# --- split-assignment leakage ---------------------------------------------


def check_split_leakage(split_rows: list[dict]) -> list[str]:
    """split_rows: [{"image_id":..., "split":...}, ...] as read from
    split.csv. Flags:
      - the same image_id appearing more than once with DIFFERENT splits
        (a real leakage risk -- ambiguous which split governs it)
      - a split value outside {calibration, validation, holdout}
    Does NOT flag the same image_id repeated with the SAME split twice
    (redundant, not leaky) -- only disagreement matters here.
    """
    valid_splits = {"calibration", "validation", "holdout"}
    issues = []
    seen = defaultdict(set)
    for row in split_rows:
        image_id = row.get("image_id")
        split = row.get("split")
        if not image_id:
            continue
        if split not in valid_splits:
            issues.append(f"{image_id}: invalid split {split!r} -- must be one of {sorted(valid_splits)}")
            continue
        seen[image_id].add(split)

    for image_id, splits in seen.items():
        if len(splits) > 1:
            issues.append(f"{image_id}: assigned to multiple different splits {sorted(splits)} -- this is a leakage risk, resolve to exactly one")

    return issues
