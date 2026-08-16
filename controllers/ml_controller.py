"""
ml_controller.py

API endpoint สำหรับวิเคราะห์ Seasonal Color ด้วย ML model ที่เทรนเอง
ไม่ใช้ Gemini API

Endpoint:
    POST /api/ml/analyze-season
    - รับ: image (multipart/form-data)
    - คืน: { season, confidence, skin_rgb, all_probabilities, model }
"""

from flask import Blueprint, request, jsonify
from services.ml_service import predict_season
from services.auth_guard import require_auth

ml_bp = Blueprint("ml", __name__)

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_MIMETYPES = {"image/jpeg", "image/png", "image/webp"}


@ml_bp.route("/ml/analyze-season", methods=["POST"])
@require_auth
def analyze_season():
    """
    วิเคราะห์ Seasonal Color จากรูปใบหน้า
    ใช้ RandomForest model ที่เทรนเองจาก skin tone RGB
    """
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

    try:
        result = predict_season(image_bytes)
        return jsonify({"success": True, **result}), 200
    except Exception as exc:
        import traceback
        print("❌ ml analyze-season error:", traceback.format_exc())
        return jsonify({"error": str(exc)}), 500
