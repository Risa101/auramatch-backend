# controllers/promotion_controller.py
from flask import Blueprint, jsonify, request
from services.promotion_service import (
    get_all_promotion,
    get_promotion_by_id,
    insert_promotion,
    update_promotion,
    delete_promotion
)

promotion_bp = Blueprint("promotion_bp", __name__)

# =========================
# ADMIN ROUTES (aliases)
# =========================
@promotion_bp.route("/admin/promotions", methods=["GET"])
def admin_get_promotions():
    data = get_all_promotion()
    return jsonify({"status": "success", "data": data}), 200

@promotion_bp.route("/admin/promotions", methods=["POST"])
def admin_create_promotion():
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "Missing body"}), 400
    new_id = insert_promotion(data)
    if not new_id:
        return jsonify({"status": "error", "message": "Insert failed"}), 500
    return jsonify({"status": "success", "promotion_id": new_id}), 201

@promotion_bp.route("/admin/promotions/<int:pid>", methods=["PUT", "PATCH"])
def admin_update_promotion(pid):
    data = request.get_json()
    success = update_promotion(pid, data)
    if not success:
        return jsonify({"status": "error", "message": "Not found"}), 404
    return jsonify({"status": "success", "message": "Updated"}), 200

@promotion_bp.route("/admin/promotions/<int:pid>", methods=["DELETE"])
def admin_delete_promotion(pid):
    success = delete_promotion(pid)
    if not success:
        return jsonify({"status": "error", "message": "Not found"}), 404
    return jsonify({"status": "success", "message": "Deleted"}), 200

# =========================
# GET : โปรโมชั่นทั้งหมด
# =========================
@promotion_bp.route("/promotions", methods=["GET"])
def get_promotions():
    data = get_all_promotion()
    return jsonify({
        "status": "success",
        "data": data
    }), 200


# =========================
# GET : โปรโมชั่นตาม ID
# =========================
@promotion_bp.route("/promotion/<int:pid>", methods=["GET"])
def get_promotion(pid):
    data = get_promotion_by_id(pid)
    if not data:
        return jsonify({
            "status": "error",
            "message": "Promotion not found"
        }), 404

    return jsonify({
        "status": "success",
        "data": data
    }), 200


# =========================
# POST : เพิ่มโปรโมชั่น
# =========================
@promotion_bp.route("/promotion", methods=["POST"])
def create_promotion():
    data = request.get_json()

    required = ["promo_name", "promo_detail", "brand_id", "superadmin_id"]
    for f in required:
        if f not in data:
            return jsonify({
                "status": "error",
                "message": f"Missing field: {f}"
            }), 400

    new_id = insert_promotion(data)

    if not new_id:
        return jsonify({
            "status": "error",
            "message": "Insert failed"
        }), 500

    return jsonify({
        "status": "success",
        "promotion_id": new_id
    }), 201


# =========================
# PUT : แก้ไขโปรโมชั่น
# =========================
@promotion_bp.route("/promotion/<int:pid>", methods=["PUT"])
def update_promotion_controller(pid):
    data = request.get_json()

    success = update_promotion(pid, data)
    if not success:
        return jsonify({
            "status": "error",
            "message": "Promotion not found or no update"
        }), 404

    return jsonify({
        "status": "success",
        "message": "Promotion updated"
    }), 200


# =========================
# DELETE : ลบโปรโมชั่น
# =========================
@promotion_bp.route("/promotion/<int:pid>", methods=["DELETE"])
def delete_promotion_controller(pid):
    success = delete_promotion(pid)
    if not success:
        return jsonify({
            "status": "error",
            "message": "Promotion not found"
        }), 404

    return jsonify({
        "status": "success",
        "message": "Promotion deleted"
    }), 200
