from flask import Blueprint, request, jsonify, g
from services.review_service import (
    get_all_reviews,
    get_review_by_id,
    insert_review,
    update_review_by_owner,
    delete_review_by_owner,
    admin_update_review,
    admin_delete_review,
)
from services.auth_guard import require_auth, require_admin

review_bp = Blueprint("review_bp", __name__)

# =========================
# ADMIN ROUTES
# =========================
@review_bp.route("/admin/reviews", methods=["GET"])
@require_admin
def admin_list_reviews():
    return jsonify(get_all_reviews()), 200

@review_bp.route("/admin/reviews", methods=["POST"])
@require_admin
def admin_create_review():
    data = request.get_json(silent=True)
    if not data or any(k not in data for k in ["user_id", "product_id", "rating"]):
        return jsonify({"error": "Missing required fields"}), 400
    new_id = insert_review(
        user_id=data["user_id"],
        product_id=data["product_id"],
        rating=data["rating"],
        comment=data.get("comment")
    )
    return jsonify({"review_id": new_id}), 201

@review_bp.route("/admin/reviews/<int:rid>", methods=["PUT", "PATCH"])
@require_admin
def admin_update_review_route(rid):
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400
    if not admin_update_review(rid, data):
        return jsonify({"error": "Review not found"}), 404
    return jsonify({"message": "Review updated"}), 200

@review_bp.route("/admin/reviews/<int:rid>", methods=["DELETE"])
@require_admin
def admin_delete_review_route(rid):
    if not admin_delete_review(rid):
        return jsonify({"error": "Review not found"}), 404
    return jsonify({"message": "Review deleted"}), 200

# =========================
# PUBLIC ROUTES
# =========================
@review_bp.route("/reviews", methods=["GET"])
def list_reviews():
    return jsonify(get_all_reviews()), 200

@review_bp.route("/reviews/<int:rid>", methods=["GET"])
def get_review(rid):
    row = get_review_by_id(rid)
    if not row:
        return jsonify({"error": "Review not found"}), 404
    return jsonify(row), 200

@review_bp.route("/reviews", methods=["POST"])
@require_auth
def create_review():
    data = request.get_json(silent=True)
    if not data or any(k not in data for k in ["user_id", "product_id", "rating"]):
        return jsonify({"error": "Missing required fields"}), 400
    # Enforce: user can only post review as themselves
    if int(data["user_id"]) != g.auth_payload.get("user_id"):
        return jsonify({"error": "forbidden"}), 403
    new_id = insert_review(
        user_id=data["user_id"],
        product_id=data["product_id"],
        rating=data["rating"],
        comment=data.get("comment")
    )
    return jsonify({"review_id": new_id}), 201

@review_bp.route("/reviews/<int:rid>", methods=["PUT"])
@require_auth
def update_review_route(rid):
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400
    result = update_review_by_owner(rid, g.auth_payload.get("user_id"), data)
    if result == "not_found":
        return jsonify({"error": "Review not found"}), 404
    if result == "forbidden":
        return jsonify({"error": "forbidden"}), 403
    return jsonify({"message": "Review updated"}), 200

@review_bp.route("/reviews/<int:rid>", methods=["DELETE"])
@require_auth
def delete_review_route(rid):
    result = delete_review_by_owner(rid, g.auth_payload.get("user_id"))
    if result == "not_found":
        return jsonify({"error": "Review not found"}), 404
    if result == "forbidden":
        return jsonify({"error": "forbidden"}), 403
    return jsonify({"message": "Review deleted"}), 200
