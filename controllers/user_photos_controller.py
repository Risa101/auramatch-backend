from flask import Blueprint, request, jsonify, g
from services.user_photos_service import get_all_user_photos, insert_user_photo
from services.auth_guard import require_auth, require_admin

user_photos_bp = Blueprint("user_photos_bp", __name__)

@user_photos_bp.route("/user-photos", methods=["GET"])
@require_admin
def list_user_photos():
    return jsonify(get_all_user_photos()), 200

@user_photos_bp.route("/user-photos", methods=["POST"])
@require_auth
def create_user_photo():
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id")
    photo_url = data.get("photo_url")

    if not user_id or not photo_url:
        return jsonify({"error": "user_id and photo_url required"}), 400

    # Users can only add photos for themselves
    if int(user_id) != g.auth_payload.get("user_id"):
        return jsonify({"error": "forbidden"}), 403

    new_id = insert_user_photo(data)
    return jsonify({"user_photo_id": new_id}), 201
