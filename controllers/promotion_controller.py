# controllers/promotion_controller.py
from flask import Blueprint, jsonify, request
from services.promotion_service import (
    get_all_promotion,
    get_promotion_by_id,
    insert_promotion,
    update_promotion,
    delete_promotion
)
from services.auth_guard import require_admin

promotion_bp = Blueprint("promotion_bp", __name__)

# =========================
# ADMIN ROUTES
# =========================
@promotion_bp.route("/admin/promotions", methods=["GET"])
@require_admin
def admin_get_promotions():
    data = get_all_promotion()
    return jsonify({"status": "success", "data": data}), 200

@promotion_bp.route("/admin/promotions", methods=["POST"])
@require_admin
def admin_create_promotion():
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "Missing body"}), 400
    new_id = insert_promotion(data)
    if not new_id:
        return jsonify({"status": "error", "message": "Insert failed"}), 500
    return jsonify({"status": "success", "promotion_id": new_id}), 201

@promotion_bp.route("/admin/promotions/<int:pid>", methods=["PUT", "PATCH"])
@require_admin
def admin_update_promotion(pid):
    data = request.get_json()
    success = update_promotion(pid, data)
    if not success:
        return jsonify({"status": "error", "message": "Not found"}), 404
    return jsonify({"status": "success", "message": "Updated"}), 200

@promotion_bp.route("/admin/promotions/<int:pid>", methods=["DELETE"])
@require_admin
def admin_delete_promotion(pid):
    success = delete_promotion(pid)
    if not success:
        return jsonify({"status": "error", "message": "Not found"}), 404
    return jsonify({"status": "success", "message": "Deleted"}), 200

# =========================
# PUBLIC READ ROUTES
# =========================
@promotion_bp.route("/promotions", methods=["GET"])
def get_promotions():
    data = get_all_promotion()
    return jsonify({"status": "success", "data": data}), 200

@promotion_bp.route("/promotion/<int:pid>", methods=["GET"])
def get_promotion(pid):
    data = get_promotion_by_id(pid)
    if not data:
        return jsonify({"status": "error", "message": "Promotion not found"}), 404
    return jsonify({"status": "success", "data": data}), 200
