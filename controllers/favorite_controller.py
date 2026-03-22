from flask import Blueprint, request, jsonify
from services.favorite_service import get_favorite_by_user, toggle_favorite

favorite_bp = Blueprint("favorite", __name__, url_prefix="/favorites")

@favorite_bp.route("/<int:user_id>", methods=["GET"])
def favorites_by_user(user_id):
    data = get_favorite_by_user(user_id)
    return jsonify({"status": "success", "data": data})

@favorite_bp.route("/toggle", methods=["POST"])
def toggle():
    body = request.get_json() or {}
    user_id = body.get("user_id")
    product_id = body.get("product_id")
    if not user_id or not product_id:
        return jsonify({"status": "error", "message": "Required fields missing"}), 400
    action = toggle_favorite(user_id, product_id)
    return jsonify({"status": "success", "action": action})