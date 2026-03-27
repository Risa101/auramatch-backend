import os
from flask import Blueprint, request, jsonify
from services.gemini_service import generate_image_with_gemini, analyze_face_with_gemini
from services.auth_guard import require_auth
from extensions import limiter

gemini_bp = Blueprint("gemini", __name__)

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_MIMETYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


def _validate_image(image_file):
    """Returns (image_bytes, error_response) — one of them will be None."""
    if not image_file:
        return None, (jsonify({"error": "กรุณาอัปโหลดรูปภาพ"}), 400)

    mime = (image_file.mimetype or "").lower()
    if mime not in ALLOWED_MIMETYPES:
        return None, (jsonify({"error": "รองรับเฉพาะไฟล์ภาพ (JPEG, PNG, WebP, GIF)"}), 400)

    image_bytes = image_file.read()
    if not image_bytes:
        return None, (jsonify({"error": "ไฟล์รูปภาพว่างเปล่า"}), 400)
    if len(image_bytes) > MAX_IMAGE_SIZE:
        return None, (jsonify({"error": "รูปภาพมีขนาดใหญ่เกินไป (สูงสุด 10MB)"}), 400)

    return image_bytes, None


@gemini_bp.route("/gemini/generate-image", methods=["POST"])
@require_auth
@limiter.limit("20 per hour")
def generate_image():
    image_bytes, err = _validate_image(request.files.get("image"))
    if err:
        return err

    prompt = (request.form.get("prompt") or "").strip()
    if not prompt:
        prompt = os.getenv("GEMINI_DEFAULT_PROMPT", "Describe this image in Thai.")

    try:
        result = generate_image_with_gemini(
            image_bytes=image_bytes,
            image_mime=request.files["image"].mimetype or "image/jpeg",
            prompt=prompt,
        )
        return jsonify({
            "success": True,
            "text": result.get("text"),
            "image": result.get("data_url"),
        }), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@gemini_bp.route("/gemini/analyze-face", methods=["POST"])
@require_auth
@limiter.limit("30 per hour")
def analyze_face():
    image_bytes, err = _validate_image(request.files.get("image"))
    if err:
        return err

    try:
        result = analyze_face_with_gemini(
            image_bytes=image_bytes,
            image_mime=request.files["image"].mimetype or "image/jpeg",
        )
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
