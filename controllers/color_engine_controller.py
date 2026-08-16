"""
color_engine_controller.py

API endpoint สำหรับวิเคราะห์ Seasonal Color ด้วย color engine ของ auramatchgenz
(MediaPipe 468 landmarks + CIELAB white-balance/skin-segmentation pipeline)
รันเป็น service แยกต่างหาก — endpoint นี้แค่ proxy คำขอไปให้

Endpoint:
    POST /api/color-engine/analyze
    - รับ: image (multipart/form-data)
    - คืน: { success, valid, season, confidence, sub_tone, warm_cool, ranked_seasons, ... }
"""

from flask import Blueprint, request, jsonify
from services.color_engine_service import analyze_with_color_engine, ColorEngineUnavailable
from services.auth_guard import require_auth

color_engine_bp = Blueprint("color_engine", __name__)

MAX_IMAGE_SIZE = 12 * 1024 * 1024  # 12MB, matches the color engine's own cap
ALLOWED_MIMETYPES = {"image/jpeg", "image/png", "image/webp"}


@color_engine_bp.route("/color-engine/analyze", methods=["POST"])
@require_auth
def analyze_color_engine():
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
        return jsonify({"error": "รูปภาพใหญ่เกิน 12MB"}), 400

    try:
        result = analyze_with_color_engine(image_bytes, image_file.filename, mime)
        return jsonify(result), 200
    except ColorEngineUnavailable as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        import traceback
        print("❌ color-engine analyze error:", traceback.format_exc())
        return jsonify({"error": str(exc)}), 500
