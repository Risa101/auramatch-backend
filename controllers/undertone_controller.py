"""
undertone_controller.py

API endpoint สำหรับวิเคราะห์ Skin Undertone + Personal Color Season จาก Lab ล้วนๆ
พอร์ตมาจาก detection_skin_for_cream(2).ipynb (ดูรายละเอียดใน services/undertone_service.py)

Endpoints:
    POST /api/undertone/analyze
    - รับ: image (multipart/form-data)
    - คืน: { success, undertone, ita, l_star, skin_pixel_count, recommended_products }

    POST /api/season/analyze
    - รับ: image (multipart/form-data)
    - คืน: { success, season, ita, l_star, skin_pixel_count }
    - แทนที่ RandomForest (services/ml_service.py, ลบแล้ว — ไม่มี dataset เดิมให้
      เทรนใหม่แบบ Lab ได้ จึงเปลี่ยนเป็นกฎ classical ที่ใช้ Lab เหมือนกันทั้งไฟล์แทน)
"""

from flask import Blueprint, request, jsonify
from services.undertone_service import analyze_undertone, analyze_season
from services.auth_guard import require_auth

undertone_bp = Blueprint("undertone", __name__)

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_MIMETYPES = {"image/jpeg", "image/png", "image/webp"}


def _read_validated_image():
    """
    อ่าน + validate ไฟล์รูปจาก request.files['image'] ใช้ร่วมกันทุก endpoint ในไฟล์นี้
    คืนค่า (image_bytes, None) ถ้าผ่าน หรือ (None, (json_response, status_code)) ถ้าไม่ผ่าน
    """
    image_file = request.files.get("image")
    if not image_file:
        return None, (jsonify({"error": "กรุณาอัปโหลดรูปภาพ"}), 400)

    mime = (image_file.mimetype or "").lower()
    if mime not in ALLOWED_MIMETYPES:
        return None, (jsonify({"error": "รองรับเฉพาะ JPEG, PNG, WebP"}), 400)

    image_bytes = image_file.read()
    if not image_bytes:
        return None, (jsonify({"error": "ไฟล์รูปภาพว่างเปล่า"}), 400)
    if len(image_bytes) > MAX_IMAGE_SIZE:
        return None, (jsonify({"error": "รูปภาพใหญ่เกิน 10MB"}), 400)

    return image_bytes, None


@undertone_bp.route("/undertone/analyze", methods=["POST"])
@require_auth
def analyze():
    """
    วิเคราะห์ undertone จากรูปใบหน้า/มือ แล้วแนะนำโทนสีบลัชออน
    ใช้ classical CV pipeline จาก notebook (ไม่มี ML model — ดู undertone_service.py)
    """
    image_bytes, err = _read_validated_image()
    if err:
        return err

    try:
        result = analyze_undertone(image_bytes)
        return jsonify({"success": True, **result}), 200
    except Exception as exc:
        import traceback
        print("❌ undertone analyze error:", traceback.format_exc())
        return jsonify({"error": str(exc)}), 500


@undertone_bp.route("/season/analyze", methods=["POST"])
@require_auth
def analyze_season_route():
    """
    วิเคราะห์ personal color season (Spring/Summer/Autumn/Winter) จากรูปใบหน้า
    ใช้กฎ classical บน Lab ล้วนๆ (ดู estimate_season() ใน undertone_service.py)
    ไม่ใช่ ML model — แทนที่ RandomForest เดิมทั้งหมด
    """
    image_bytes, err = _read_validated_image()
    if err:
        return err

    try:
        result = analyze_season(image_bytes)
        return jsonify({"success": True, **result}), 200
    except Exception as exc:
        import traceback
        print("❌ season analyze error:", traceback.format_exc())
        return jsonify({"error": str(exc)}), 500
