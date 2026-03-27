from flask import Blueprint, request, jsonify, g
from services.favorite_service import get_favorite_by_user, toggle_favorite
from services.auth_guard import require_auth

favorite_bp = Blueprint("favorite", __name__, url_prefix="/favorites")

@favorite_bp.route("/<int:user_id>", methods=["GET"])
@require_auth
def favorites_by_user(user_id):
    if user_id != g.auth_payload.get("user_id"):
        return jsonify({"status": "error", "message": "forbidden"}), 403
    data = get_favorite_by_user(user_id)
    return jsonify({"status": "success", "data": data})

@favorite_bp.route("/toggle", methods=["POST"])
@require_auth
def toggle():
    body = request.get_json() or {}
    user_id = body.get("user_id")
    product_id = body.get("product_id")
    if not user_id or not product_id:
        return jsonify({"status": "error", "message": "Required fields missing"}), 400
    if int(user_id) != g.auth_payload.get("user_id"):
        return jsonify({"status": "error", "message": "forbidden"}), 403
    action = toggle_favorite(user_id, product_id)
    return jsonify({"status": "success", "action": action})