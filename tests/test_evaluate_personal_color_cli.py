"""Tests for the pure-logic helpers in scripts/evaluate_personal_color.py
(CSV loading, image/landmark lookup, in-memory consistency variants) --
does NOT invoke main() or touch any real photo.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_personal_color import (
    denormalize, find_image, load_labels, make_consistency_variants,
)


def test_load_labels_basic(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("image_id,season,undertone,face_shape,lighting,makeup,notes\n"
                 "sample_001,Spring,Warm,Oval,daylight,no,\n")
    labels = load_labels(p)
    assert labels["sample_001"]["season"] == "Spring"
    assert labels["sample_001"]["makeup"] == "no"


def test_load_labels_blank_cells_are_none(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("image_id,season,undertone,face_shape,lighting,makeup,notes\n"
                 "sample_001,,,,,,\n")
    labels = load_labels(p)
    assert labels["sample_001"]["season"] is None


def test_load_labels_tolerates_unescaped_comma_in_notes(tmp_path):
    """Regression test: a notes field with a raw comma used to crash
    load_labels with AttributeError (csv.DictReader stuffs the overflow
    into a `None`-keyed list) -- found via a real smoke-test run.
    """
    p = tmp_path / "labels.csv"
    p.write_text("image_id,season,undertone,face_shape,lighting,makeup,notes\n"
                 "sample_001,Spring,Warm,Oval,daylight,no,taken outdoors, slightly cloudy\n")
    labels = load_labels(p)  # must not raise
    assert labels["sample_001"]["season"] == "Spring"


def test_load_labels_skips_rows_without_image_id(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("image_id,season\n,Spring\nsample_001,Winter\n")
    labels = load_labels(p)
    assert list(labels.keys()) == ["sample_001"]


def test_find_image_matches_extension_variants(tmp_path):
    (tmp_path / "sample_001.png").touch()
    assert find_image(tmp_path, "sample_001") == tmp_path / "sample_001.png"
    assert find_image(tmp_path, "does_not_exist") is None


def test_find_image_matches_id_with_extension_already_included(tmp_path):
    (tmp_path / "sample_002.jpg").touch()
    assert find_image(tmp_path, "sample_002.jpg") == tmp_path / "sample_002.jpg"


def test_denormalize_scales_by_width_and_height():
    landmarks = [[0.5, 0.25, 0.0], [0.1, 0.9, 0.0]]
    px = denormalize(landmarks, width=200, height=100)
    assert px[0].tolist() == [100.0, 25.0]
    assert px[1].tolist() == [20.0, 90.0]


def test_make_consistency_variants_never_mutates_the_input():
    bgr = np.full((50, 50, 3), 128, dtype=np.uint8)
    original = bgr.copy()
    variants = make_consistency_variants(bgr)
    assert np.array_equal(bgr, original)  # input array untouched
    assert set(variants.keys()) >= {"resized_75pct", "brighter", "darker", "warmer", "cooler", "jpeg_q40"}
    assert variants["brighter"].mean() > bgr.mean()
    assert variants["darker"].mean() < bgr.mean()
