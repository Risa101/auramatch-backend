"""
undertone_controller.py

API endpoint สำหรับวิเคราะห์ Skin Undertone + แนะนำสินค้าบลัชออนจริงจาก DB
พอร์ตมาจาก detection_skin_for_cream(2).ipynb (ดูรายละเอียดใน services/undertone_service.py)

Endpoint:
    POST /api/undertone/analyze
    - รับ: image (multipart/form-data)
    - คืน: { success, undertone, ita, l_star, skin_pixel_count, recommended_products }
"""

from flask import Blueprint, request, jsonify
from services.undertone_service import analyze_undertone
from services.auth_guard import require_auth

undertone_bp = Blueprint("undertone", __name__)

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_MIMETYPES = {"image/jpeg", "image/png", "image/webp"}


@undertone_bp.route("/undertone/analyze", methods=["POST"])
@require_auth
def analyze():
    """
    วิเคราะห์ undertone จากรูปใบหน้า/มือ แล้วแนะนำโทนสีบลัชออน
    ใช้ classical CV pipeline จาก notebook (ไม่มี ML model — ดู undertone_service.py)
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
        result = analyze_undertone(image_bytes)
        return jsonify({"success": True, **result}), 200
    except Exception as exc:
        import traceback
        print("❌ undertone analyze error:", traceback.format_exc())
        return jsonify({"error": str(exc)}), 500
