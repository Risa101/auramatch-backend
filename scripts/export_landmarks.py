#!/usr/bin/env python3
"""
export_landmarks.py

Batch face-landmark export for real-photo evaluation. Reads a folder of
photos, runs MediaPipe FaceLandmarker (Tasks API, IMAGE mode) on each, and
writes one JSON file per image in the format
services/personal_color_service.py and src/utils/faceGeometry.js both
already consume: normalized [0,1] x/y/z, >= 468 points, standard MediaPipe
face-mesh topology. See FORMAT below.

WHY THIS EXISTS: production detects landmarks client-side, in the browser
(src/utils/faceLandmarkerMP.js) -- there was no way to batch-process a
folder of already-collected labeled photos without opening each one in the
app by hand. This is the offline equivalent: SAME model, SAME landmark
topology, SAME detection approach (MediaPipe FaceLandmarker, IMAGE mode),
just running in Python instead of the browser. It is not a second/different
landmark algorithm.

REUSES the exact model file auramatchgenz/backend already downloaded
(app/color_engine/models/face_landmarker.task) and mirrors its
FaceMesh.detect() wrapper (auramatchgenz/backend/app/color_engine/
face_mesh.py) -- the detection logic is duplicated here (not imported)
deliberately, so this tool stays self-contained and doesn't depend on a
sibling repo's internal module layout still existing on whatever machine
runs it later. "Reuse the existing implementation" means the same model +
approach, not a fragile cross-repo import.

ISOLATED ENVIRONMENT REQUIRED -- do NOT run this with backend_auramatch's
own .venv (the one `pytest tests/` uses). mediapipe pulls in
opencv-contrib-python, which collides with that venv's
opencv-python-headless (same `cv2` import, different wheel -- see this
project's requirements.txt scikit-learn comment, and
auramatchgenz/backend/requirements.txt's own warning about the exact same
clash). This is precisely why mediapipe was never added to the production
color-engine port. Set up a separate venv:

    python3 -m venv .venv-landmarks
    .venv-landmarks/bin/pip install -r scripts/requirements-landmarks.txt
    .venv-landmarks/bin/python scripts/export_landmarks.py \\
        --input  evaluation_dataset/images \\
        --output evaluation_dataset/landmarks \\
        --model  /path/to/face_landmarker.task

(auramatchgenz/backend/app/color_engine/models/face_landmarker.task is a
ready-made copy if that sibling repo is present; otherwise download it from
MODEL_URL below -- same file, same official Google-hosted asset.)

FORMAT (one JSON file per image, named <stem>.landmarks.json):
    {
      "image": "sample_001.jpg",
      "width": 1200, "height": 1600,
      "faceDetected": true, "faceCount": 1,
      "landmarkCount": 478,
      "landmarks": [[x, y, z], ...],   // normalized [0,1] x/y; z relative
      "faceBox": {"x":.., "y":.., "width":.., "height":..} | null,
      "rollDeg": 1.23                  // same formula as faceGeometry.js's
                                        // alignLandmarks(): atan2 between the
                                        // outer-eye-corner landmarks (33, 263)
    }
A `faceDetected: false` record still gets written (landmarks: null,
faceCount: 0) -- callers must handle it, not skip the file. The real
detection-failure rate on real photos is itself part of what this
evaluation measures (see evaluate_personal_color.py's failure-rate metric).

PRIVACY (see the audit's real-photo evaluation brief, section 28): this
tool makes no network calls, sends nothing to Gemini or any third party,
and only ever reads --input / writes --output on the local filesystem.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def roll_deg(landmarks_xyz, left_idx: int = 33, right_idx: int = 263):
    """Same formula as src/utils/faceGeometry.js's alignLandmarks() -- lets a
    developer sanity-check head tilt straight from the exported JSON without
    loading the JS module.
    """
    if left_idx >= len(landmarks_xyz) or right_idx >= len(landmarks_xyz):
        return None
    lx, ly = landmarks_xyz[left_idx][0], landmarks_xyz[left_idx][1]
    rx, ry = landmarks_xyz[right_idx][0], landmarks_xyz[right_idx][1]
    return math.degrees(math.atan2(ry - ly, rx - lx))


class LandmarkExporter:
    """Thin wrapper over MediaPipe FaceLandmarker (Tasks API, IMAGE mode).

    Deliberately minimal -- mirrors auramatchgenz/backend/app/color_engine/
    face_mesh.py's FaceMesh class, trimmed to exactly what this export tool
    needs (no drawing helpers, no context-manager dependency on a bigger
    engine object).
    """

    def __init__(self, model_path: str):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        self._mp = mp
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, image_bgr) -> dict:
        import cv2

        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        faces = result.face_landmarks

        if not faces:
            return {
                "width": w, "height": h,
                "faceDetected": False, "faceCount": 0, "landmarkCount": 0,
                "landmarks": None, "faceBox": None, "rollDeg": None,
            }

        lm = faces[0]
        coords = [[round(p.x, 6), round(p.y, 6), round(p.z, 6)] for p in lm]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        face_box = {
            "x": min(xs), "y": min(ys),
            "width": max(xs) - min(xs), "height": max(ys) - min(ys),
        }
        return {
            "width": w, "height": h,
            "faceDetected": True, "faceCount": len(faces), "landmarkCount": len(coords),
            "landmarks": coords, "faceBox": face_box, "rollDeg": roll_deg(coords),
        }

    def close(self):
        self._landmarker.close()


def _rotate_in_memory(bgr, degrees: float):
    """Rotate a BGR array by `degrees` around its center, expanding the
    canvas so the face isn't clipped out of frame. Never writes back to the
    source file -- used only for the controlled-rotation robustness check
    (see --rotations).
    """
    import cv2

    h, w = bgr.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), -degrees, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += (new_w - w) / 2
    m[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(bgr, m, (new_w, new_h), borderValue=(128, 128, 128))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="folder of photos")
    ap.add_argument("--output", required=True, type=Path, help="folder to write <stem>.landmarks.json into")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument(
        "--rotations", type=str, default=None,
        help="comma-separated degrees, e.g. '-30,-15,0,15,30' (audit real-photo "
             "brief, section 24: controlled-rotation robustness check). Images are "
             "rotated IN MEMORY only; writes <stem>__rot<deg>.landmarks.json "
             "variants in addition to the base (0deg) export.",
    )
    args = ap.parse_args()

    try:
        import cv2  # noqa: F401
        import mediapipe  # noqa: F401
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("This tool needs its OWN isolated venv -- see the module docstring "
              "(do not run it with backend_auramatch's own .venv).", file=sys.stderr)
        sys.exit(1)

    if not args.model.exists():
        print(f"ERROR: model not found at {args.model}", file=sys.stderr)
        print(f"Download it from:\n  {MODEL_URL}", file=sys.stderr)
        sys.exit(1)

    images = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"No images found in {args.input} (looked for {sorted(IMAGE_EXTS)})")
        return

    args.output.mkdir(parents=True, exist_ok=True)
    exporter = LandmarkExporter(str(args.model))

    rotations = [int(r) for r in args.rotations.split(",")] if args.rotations else [0]

    detected, missing = 0, 0
    for img_path in images:
        import cv2

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  ✗ {img_path.name}: could not decode")
            missing += 1
            continue

        for deg in rotations:
            rotated = _rotate_in_memory(bgr, deg) if deg != 0 else bgr
            record = exporter.detect(rotated)
            record["image"] = img_path.name
            record["rotationAppliedDeg"] = deg
            suffix = f"__rot{deg:+d}" if deg != 0 else ""
            out_path = args.output / f"{img_path.stem}{suffix}.landmarks.json"
            out_path.write_text(json.dumps(record))
            ok = record["faceDetected"]
            detected += 1 if ok else 0
            missing += 0 if ok else 1
            print(f"  {'✓' if ok else '✗'} {img_path.name}{suffix} -> {out_path.name} "
                  f"(faces={record['faceCount']})")

    exporter.close()
    total = len(images) * len(rotations)
    print(f"\nDone: {detected}/{total} detected, {missing}/{total} not detected.")


if __name__ == "__main__":
    main()
