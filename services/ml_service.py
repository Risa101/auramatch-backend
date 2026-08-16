"""
ml_service.py

บริการวิเคราะห์ Seasonal Color ด้วย RandomForest model ที่เทรนเอง
ไม่ใช้ Gemini API ในส่วนนี้

ขั้นตอน:
  1. รับรูปภาพ (bytes)
  2. Extract สีผิวจากบริเวณกลางรูป (Pillow)
  3. คำนวณ features (brightness, warmth, pinkness, saturation)
  4. Predict ด้วย RandomForest model (.pkl)
  5. คืนค่า season + confidence
"""

import os
import io
import numpy as np
import pandas as pd
import joblib
from PIL import Image

# โหลดโมเดลครั้งเดียวตอน import (ไม่โหลดซ้ำทุก request)
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/ml/seasonal_color_model.pkl")
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = joblib.load(_MODEL_PATH)
    return _model


def extract_skin_tone(image_bytes):
    """
    Extract ค่า RGB เฉลี่ยของผิวหน้าจากรูปภาพ

    วิธี:
    - เปิดรูปด้วย Pillow
    - crop บริเวณกลางรูป (สมมติว่าหน้าอยู่กลาง)
    - filter pixel ที่น่าจะเป็นผิว (ไม่เข้มเกินไป ไม่สว่างเกินไป)
    - คืนค่า average RGB

    Returns:
        (R, G, B) เป็น int 0-255
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size

    # crop บริเวณกลางรูป 40% ของ width/height (บริเวณใบหน้า)
    left   = int(w * 0.30)
    right  = int(w * 0.70)
    top    = int(h * 0.15)
    bottom = int(h * 0.55)

    face_region = img.crop((left, top, right, bottom))
    pixels = np.array(face_region).reshape(-1, 3)  # shape: (N, 3)

    # กรอง pixel ที่เป็น background/ผม/เงา ออก
    # เก็บเฉพาะ pixel ที่ brightness อยู่ในช่วง 80-240 (ผิวปกติ)
    brightness = pixels.mean(axis=1)
    skin_mask = (brightness > 80) & (brightness < 240)
    skin_pixels = pixels[skin_mask]

    if len(skin_pixels) == 0:
        # fallback: ใช้ทั้งรูป
        skin_pixels = pixels

    R = int(np.mean(skin_pixels[:, 0]))
    G = int(np.mean(skin_pixels[:, 1]))
    B = int(np.mean(skin_pixels[:, 2]))

    return R, G, B


def build_features(R, G, B):
    """
    คำนวณ features เดียวกับที่ใช้ตอนเทรนโมเดล
    ต้องตรงกับ train_seasonal_color.py ทุก feature
    """
    brightness  = (R + G + B) / 3
    warmth      = R - B          # R - B สูง = warm tone
    pinkness    = R - G          # R - G สูง = pinkish
    max_rgb     = max(R, G, B)
    min_rgb     = min(R, G, B)
    saturation  = (max_rgb - min_rgb) / (max_rgb + 1)

    return pd.DataFrame([{
        "R": R, "G": G, "B": B,
        "brightness": brightness,
        "warmth": warmth,
        "pinkness": pinkness,
        "saturation": saturation,
    }])


def predict_season(image_bytes):
    """
    วิเคราะห์ Seasonal Color จากรูปภาพ

    Args:
        image_bytes: รูปภาพในรูปแบบ bytes

    Returns:
        dict {
            season: "Spring" | "Summer" | "Autumn" | "Winter",
            confidence: float (0-1),
            skin_rgb: { R, G, B },
            all_probabilities: { Spring: float, ... }
        }
    """
    model = _get_model()

    # 1. extract สีผิว
    R, G, B = extract_skin_tone(image_bytes)

    # 2. สร้าง features
    features = build_features(R, G, B)

    # 3. predict
    season = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    classes = model.classes_

    confidence = float(max(probabilities))
    all_probs = {c: round(float(p), 4) for c, p in zip(classes, probabilities)}

    return {
        "season": season,
        "confidence": confidence,
        "skin_rgb": {"R": R, "G": G, "B": B},
        "all_probabilities": all_probs,
        "model": "RandomForest",
    }
