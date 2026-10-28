"""
personal_color_controller.py

THE single Personal Color endpoint (see services/personal_color_service.py
for why this replaces the old three-classifier overwrite chain).

Endpoint:
    POST /api/personal-color/analyze
    - รับ: image (multipart/form-data), landmarks (JSON string: list of
      [x, y] or [x, y, z] normalized [0,1] floats, >= 468 points, from the
      browser's MediaPipe FaceLandmarker -- see src/utils/faceLandmarkerMP.js)
    - คืน: { success, valid, season, sub_tone, confidence, reliability,
             ranked_seasons, axes, undertone, skin_lab, quality,
             recommended_products, ... } เมื่อ valid=true
      หรือ { success, valid: false, reason } เมื่อวัดผลไม่ได้ (เช่น หาผิว
      ไม่พอ) -- ไม่เคยคาดเดา season ทิ้งไปเฉยๆ
"""

import json

from flask import Blueprint, request, jsonify
from services.personal_color_service import analyze_personal_color, InvalidLandmarks
from services.auth_guard import require_auth

personal_color_bp = Blueprint("personal_color", __name__)

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_MIMETYPES = {"image/jpeg", "image/png", "image/webp"}


@personal_color_bp.route("/personal-color/analyze", methods=["POST"])
@require_auth
def analyze():
    image_file = request.files.get("image")
    if not image_file:
        return jsonify({"error": "กรุณาอัปโหลดรูปภาพ"}), 400

    mime = (image_file.mimetype or "").lower()
    if mime not in ALLOWED_MIMETYPES:
        return jsonify({"error": "รองรับเฉพาะ JPEG, PNG, WebP"}), 400

    image_bytes = image_file.read()
    if not image_bytes:
        return jsonify({"error": "ไฟล์รูปภาพว่างเปล่า"}), 400
    if len(image_bytes) > MAX_IMAGE_SIZE:
        return jsonify({"error": "รูปภาพใหญ่เกิน 10MB"}), 400

    landmarks_raw = request.form.get("landmarks")
    if not landmarks_raw:
        return jsonify({"error": "ไม่พบข้อมูล landmarks (ต้องตรวจจับใบหน้าก่อนส่งวิเคราะห์)"}), 400
    try:
        landmarks = json.loads(landmarks_raw)
    except json.JSONDecodeError:
        return jsonify({"error": "รูปแบบ landmarks ไม่ถูกต้อง (ต้องเป็น JSON array)"}), 400

    try:
        result = analyze_personal_color(image_bytes, landmarks)
        return jsonify({"success": True, **result}), 200
    except InvalidLandmarks as exc:
        return jsonify({"error": str(exc)}), 400
    except ValueError as exc:
        # e.g. undecodable image
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        import traceback
        print("❌ personal-color analyze error:", traceback.format_exc())
        return jsonify({"error": str(exc)}), 500
