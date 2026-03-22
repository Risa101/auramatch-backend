from flask import Blueprint, jsonify
from services.profiles_service import get_all_profiles, get_profile_by_user_id

profiles_bp = Blueprint("profiles_bp", __name__)

@profiles_bp.route("/profiles", methods=["GET"])
def list_profiles():
    rows = get_all_profiles()
    return jsonify(rows), 200

@profiles_bp.route("/profiles/<int:user_id>", methods=["GET"])
def get_profile(user_id):
    row = get_profile_by_user_id(user_id)
    if row:
        return jsonify(row), 200
    return jsonify({"error": "Profile not found"}), 404
