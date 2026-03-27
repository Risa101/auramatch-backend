from flask import Blueprint, jsonify, g
from services.profiles_service import get_all_profiles, get_profile_by_user_id
from services.auth_guard import require_auth, require_admin

profiles_bp = Blueprint("profiles_bp", __name__)

@profiles_bp.route("/profiles", methods=["GET"])
@require_admin
def list_profiles():
    rows = get_all_profiles()
    return jsonify(rows), 200

@profiles_bp.route("/profiles/<int:user_id>", methods=["GET"])
@require_auth
def get_profile(user_id):
    # Users can only fetch their own profile; admins can fetch any
    if user_id != g.auth_payload.get("user_id") and g.auth_payload.get("role") != "admin":
        return jsonify({"error": "forbidden"}), 403
    row = get_profile_by_user_id(user_id)
    if row:
        return jsonify(row), 200
    return jsonify({"error": "Profile not found"}), 404
