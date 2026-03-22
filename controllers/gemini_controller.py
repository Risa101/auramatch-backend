import os
from flask import Blueprint, request, jsonify
from services.gemini_service import generate_image_with_gemini, analyze_face_with_gemini

gemini_bp = Blueprint("gemini", __name__)

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB


@gemini_bp.route("/gemini/generate-image", methods=["POST"])
def generate_image():
    image_file = request.files.get("image")
    prompt = (request.form.get("prompt") or "").strip()

    if not image_file:
        return jsonify({"error": "กรุณาอัปโหลดรูปภาพ"}), 400

    image_bytes = image_file.read()
    if not image_bytes:
        return jsonify({"error": "ไฟล์รูปภาพว่างเปล่า"}), 400
    if len(image_bytes) > MAX_IMAGE_SIZE:
        return jsonify({"error": "รูปภาพมีขนาดใหญ่เกินไป (สูงสุด 10MB)"}), 400

    if not prompt:
        prompt = os.getenv("GEMINI_DEFAULT_PROMPT", "Describe this image in Thai.")

    try:
        result = generate_image_with_gemini(
            image_bytes=image_bytes,
            image_mime=image_file.mimetype or "image/jpeg",
            prompt=prompt,
        )
        return jsonify({
            "success": True,
            "text": result.get("text"),
            "image": result.get("data_url"),
        }), 200
    except Exception as exc:
        print(f"❌ Gemini generate-image error: {exc}")
        return jsonify({"error": str(exc)}), 500


@gemini_bp.route("/gemini/analyze-face", methods=["POST"])
def analyze_face():
    image_file = request.files.get("image")

    if not image_file:
        return jsonify({"error": "กรุณาอัปโหลดรูปภาพ"}), 400

    image_bytes = image_file.read()
    if not image_bytes:
        return jsonify({"error": "ไฟล์รูปภาพว่างเปล่า"}), 400
    if len(image_bytes) > MAX_IMAGE_SIZE:
        return jsonify({"error": "รูปภาพมีขนาดใหญ่เกินไป (สูงสุด 10MB)"}), 400

    try:
        result = analyze_face_with_gemini(
            image_bytes=image_bytes,
            image_mime=image_file.mimetype or "image/jpeg",
        )
        return jsonify(result), 200
    except Exception as exc:
        print(f"❌ Gemini analyze-face error: {exc}")
        return jsonify({"error": str(exc)}), 500
