import numpy as np
import cv2

from services.eval.integrity import (
    average_hash, check_all_landmarks, check_landmark, check_split_leakage,
    find_exact_duplicates, find_filename_collisions, find_near_duplicates,
    hamming_distance, hash_file,
)


def _write_image(path, color=(100, 120, 140), size=(40, 40)):
    img = np.full((*size, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_hash_file_is_stable_and_content_sensitive(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    _write_image(a, color=(10, 10, 10))
    _write_image(b, color=(200, 200, 200))
    assert hash_file(a) == hash_file(a)  # stable
    assert hash_file(a) != hash_file(b)  # content-sensitive


def test_find_exact_duplicates_detects_identical_content(tmp_path):
    _write_image(tmp_path / "AM-EVAL-0001.png", color=(50, 60, 70))
    _write_image(tmp_path / "AM-EVAL-0002.png", color=(50, 60, 70))  # byte-identical
    _write_image(tmp_path / "AM-EVAL-0003.png", color=(90, 90, 90))  # unique
    dups = find_exact_duplicates(tmp_path)
    assert len(dups) == 1
    (names,) = dups.values()
    assert set(names) == {"AM-EVAL-0001.png", "AM-EVAL-0002.png"}


def test_find_exact_duplicates_empty_when_all_unique(tmp_path):
    _write_image(tmp_path / "a.png", color=(1, 2, 3))
    _write_image(tmp_path / "b.png", color=(4, 5, 6))
    assert find_exact_duplicates(tmp_path) == {}


def test_average_hash_identical_images_have_zero_distance():
    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    assert hamming_distance(average_hash(img), average_hash(img.copy())) == 0


def test_average_hash_very_different_images_have_high_distance():
    a = np.zeros((50, 50, 3), dtype=np.uint8)
    b = np.zeros((50, 50, 3), dtype=np.uint8)
    b[:25, :, :] = 255  # sharp half-white/half-black pattern, opposite of flat black
    assert hamming_distance(average_hash(a), average_hash(b)) > 0


def test_find_near_duplicates_flags_resized_copy(tmp_path):
    base = np.zeros((100, 100, 3), dtype=np.uint8)
    base[:, :50] = 200  # a real pattern, not flat (flat images alias to identical hashes trivially)
    cv2.imwrite(str(tmp_path / "orig.png"), base)
    resized = cv2.resize(base, (60, 60))
    cv2.imwrite(str(tmp_path / "resized.png"), resized)
    pairs = find_near_duplicates(tmp_path, threshold=5)
    names = {n for pair in pairs for n in pair[:2]}
    assert "orig.png" in names and "resized.png" in names


def test_find_near_duplicates_never_flags_unrelated_images(tmp_path):
    a = np.zeros((50, 50, 3), dtype=np.uint8)
    a[:25, :25] = 255
    b = np.zeros((50, 50, 3), dtype=np.uint8)
    b[25:, 25:] = 255  # opposite corner pattern
    cv2.imwrite(str(tmp_path / "a.png"), a)
    cv2.imwrite(str(tmp_path / "b.png"), b)
    pairs = find_near_duplicates(tmp_path, threshold=2)
    assert pairs == []


def test_find_filename_collisions_detects_extension_ambiguity(tmp_path):
    # Same stem, different extension -- genuinely two files on any
    # filesystem (unlike a pure case difference, which macOS's default
    # case-insensitive/case-preserving filesystem collapses into one file
    # before this function ever sees it).
    (tmp_path / "AM-EVAL-0001.jpg").touch()
    (tmp_path / "AM-EVAL-0001.jpeg").touch()
    (tmp_path / "AM-EVAL-0002.jpg").touch()
    collisions = find_filename_collisions(tmp_path)
    assert len(collisions) == 1
    assert set(collisions[0]) == {"AM-EVAL-0001.jpg", "AM-EVAL-0001.jpeg"}


def test_check_landmark_missing_when_no_file(tmp_path):
    result = check_landmark(tmp_path, "AM-EVAL-0001")
    assert result["state"] == "MISSING"


def test_check_landmark_invalid_on_malformed_json(tmp_path):
    (tmp_path / "AM-EVAL-0001.landmarks.json").write_text("{not valid json")
    result = check_landmark(tmp_path, "AM-EVAL-0001")
    assert result["state"] == "INVALID"


def test_check_landmark_no_face(tmp_path):
    (tmp_path / "AM-EVAL-0001.landmarks.json").write_text('{"faceDetected": false, "faceCount": 0}')
    result = check_landmark(tmp_path, "AM-EVAL-0001")
    assert result["state"] == "NO_FACE"


def test_check_landmark_multiple_faces(tmp_path):
    landmarks = [[0.1, 0.1, 0.0]] * 478
    import json
    (tmp_path / "AM-EVAL-0001.landmarks.json").write_text(json.dumps({
        "faceDetected": True, "faceCount": 2, "landmarks": landmarks,
    }))
    result = check_landmark(tmp_path, "AM-EVAL-0001")
    assert result["state"] == "MULTIPLE_FACES"


def test_check_landmark_success(tmp_path):
    import json
    landmarks = [[0.1, 0.1, 0.0]] * 478
    (tmp_path / "AM-EVAL-0001.landmarks.json").write_text(json.dumps({
        "faceDetected": True, "faceCount": 1, "landmarks": landmarks,
    }))
    result = check_landmark(tmp_path, "AM-EVAL-0001")
    assert result["state"] == "SUCCESS"


def test_check_landmark_invalid_too_few_points(tmp_path):
    import json
    (tmp_path / "AM-EVAL-0001.landmarks.json").write_text(json.dumps({
        "faceDetected": True, "faceCount": 1, "landmarks": [[0.1, 0.1, 0.0]] * 10,
    }))
    result = check_landmark(tmp_path, "AM-EVAL-0001")
    assert result["state"] == "INVALID"


def test_check_all_landmarks_aggregates_multiple_ids(tmp_path):
    import json
    (tmp_path / "AM-EVAL-0001.landmarks.json").write_text(json.dumps({
        "faceDetected": True, "faceCount": 1, "landmarks": [[0.1, 0.1, 0.0]] * 478,
    }))
    results = check_all_landmarks(["AM-EVAL-0001", "AM-EVAL-0002"], tmp_path)
    states = {r["image_id"]: r["state"] for r in results}
    assert states == {"AM-EVAL-0001": "SUCCESS", "AM-EVAL-0002": "MISSING"}


def test_check_split_leakage_detects_conflicting_assignment():
    rows = [
        {"image_id": "AM-EVAL-0001", "split": "calibration"},
        {"image_id": "AM-EVAL-0001", "split": "holdout"},
        {"image_id": "AM-EVAL-0002", "split": "validation"},
    ]
    issues = check_split_leakage(rows)
    assert len(issues) == 1
    assert "AM-EVAL-0001" in issues[0]


def test_check_split_leakage_ignores_exact_duplicate_rows():
    rows = [
        {"image_id": "AM-EVAL-0001", "split": "calibration"},
        {"image_id": "AM-EVAL-0001", "split": "calibration"},  # redundant, not leaky
    ]
    assert check_split_leakage(rows) == []


def test_check_split_leakage_flags_invalid_split_value():
    rows = [{"image_id": "AM-EVAL-0001", "split": "training"}]  # not one of the 3 valid values
    issues = check_split_leakage(rows)
    assert len(issues) == 1
    assert "invalid split" in issues[0]


def test_check_split_leakage_clean_dataset_has_no_issues():
    rows = [
        {"image_id": "AM-EVAL-0001", "split": "calibration"},
        {"image_id": "AM-EVAL-0002", "split": "validation"},
        {"image_id": "AM-EVAL-0003", "split": "holdout"},
    ]
    assert check_split_leakage(rows) == []
