from flask import Blueprint, request, jsonify
from services.review_service import (
    get_all_reviews,
    get_review_by_id,
    insert_review,
    update_review,
    delete_review,
)

review_bp = Blueprint("review_bp", __name__)

# =========================
# ADMIN ROUTES (aliases)
# =========================
@review_bp.route("/admin/reviews", methods=["GET"])
def admin_list_reviews():
    return jsonify(get_all_reviews()), 200

@review_bp.route("/admin/reviews", methods=["POST"])
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
def admin_update_review(rid):
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400
    if not update_review(rid, data):
        return jsonify({"error": "Review not found"}), 404
    return jsonify({"message": "Review updated"}), 200

@review_bp.route("/admin/reviews/<int:rid>", methods=["DELETE"])
def admin_delete_review(rid):
    if not delete_review(rid):
        return jsonify({"error": "Review not found"}), 404
    return jsonify({"message": "Review deleted"}), 200

@review_bp.route("/reviews", methods=["GET"])
def list_reviews():
    return jsonify(get_all_reviews()), 200


@review_bp.route("/reviews", methods=["POST"])
def create_review():
    data = request.get_json(silent=True)
    required = ["user_id", "product_id", "rating"]

    if not data or any(k not in data for k in required):
        return jsonify({"error": "Missing required fields"}), 400

    new_id = insert_review(
        user_id=data["user_id"],
        product_id=data["product_id"],
        rating=data["rating"],
        comment=data.get("comment")
    )
    return jsonify({"review_id": new_id}), 201


@review_bp.route("/reviews/<int:rid>", methods=["GET"])
def get_review(rid):
    row = get_review_by_id(rid)
    if not row:
        return jsonify({"error": "Review not found"}), 404
    return jsonify(row), 200


@review_bp.route("/reviews/<int:rid>", methods=["PUT"])
def update_review_route(rid):
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    if not update_review(rid, data):
        return jsonify({"error": "Review not found"}), 404

    return jsonify({"message": "Review updated"}), 200


@review_bp.route("/reviews/<int:rid>", methods=["DELETE"])
def delete_review_route(rid):
    if not delete_review(rid):
        return jsonify({"error": "Review not found"}), 404
    return jsonify({"message": "Review deleted"}), 200
