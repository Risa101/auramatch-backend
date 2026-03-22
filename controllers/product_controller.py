from flask import Blueprint, jsonify, request
from services.auth_guard import require_admin
from services.product_service import (
    get_all_products,
    get_all_products_admin,
    get_product_by_id,
    get_best_seller_products,
    get_recommended_products,
    insert_product,
    update_product,
    delete_product
)

products_bp = Blueprint("products_bp", __name__)

# ===============================
# GET ALL PRODUCTS
# ===============================
@products_bp.route("/products", methods=["GET"])
def products():
    try:
        data = get_all_products()
        return jsonify({
            "status": "success",
            "data": data
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



# ===============================
# ADMIN: GET ALL PRODUCTS
# ===============================
@products_bp.route("/admin/products", methods=["GET"])
@require_admin
def admin_products():
    try:
        data = get_all_products_admin()
        return jsonify({
            "status": "success",
            "data": data
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ===============================
# ADMIN: CREATE PRODUCT
# ===============================
@products_bp.route("/admin/products", methods=["POST"])
@require_admin
def admin_create_product():
    data = request.get_json(silent=True) or {}
    if not data.get("product_id") or not data.get("name"):
        return jsonify({"error": "name required"}), 400
    new_id = insert_product(data)
    if not new_id:
        return jsonify({"error": "invalid payload"}), 400
    return jsonify({"product_id": new_id}), 201


# ===============================
# ADMIN: UPDATE PRODUCT
# ===============================
@products_bp.route("/admin/products/<string:product_id>", methods=["PUT"])
@require_admin
def admin_update_product(product_id):
    data = request.get_json(silent=True) or {}
    if not update_product(product_id, data):
        return jsonify({"error": "not found"}), 404
    return jsonify({"message": "updated"}), 200


# ===============================
# ADMIN: DELETE PRODUCT
# ===============================
@products_bp.route("/admin/products/<string:product_id>", methods=["DELETE"])
@require_admin
def admin_delete_product(product_id):
    if not delete_product(product_id):
        return jsonify({"error": "not found"}), 404
    return jsonify({"message": "deleted"}), 200

# ===============================
# GET PRODUCT BY ID
# ===============================
@products_bp.route("/products/<string:product_id>", methods=["GET"])
def product_detail(product_id):
    try:
        data = get_product_by_id(product_id)
        if not data:
            return jsonify({"status": "fail", "message": "Product not found"}), 404
        return jsonify({
            "status": "success",
            "data": data
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ===============================
# 🔥 BEST SELLER
# ===============================
@products_bp.route("/products/stats/best-seller", methods=["GET"])
def best_seller():
    try:
        data = get_best_seller_products()
        return jsonify({
            "status": "success",
            "data": data
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ==========================================
# ✨ RECOMMENDED BY SEASON
# ==========================================
@products_bp.route("/products/recommendations", methods=["GET"])
def recommended_by_season():
    try:
        # รับค่า season จาก Query Parameter (เช่น ?season=Spring)
        # หากไม่ส่งมา จะใช้ค่าเริ่มต้นเป็น 'All'
        season = request.args.get("season", "All")
        
        # ดึงสินค้าแนะนำโดยจำกัด 3 รายการตามที่หน้าเว็บต้องการ
        data = get_recommended_products(season, limit=3)
        
        return jsonify({
            "status": "success",
            "season": season,
            "count": len(data),
            "data": data
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500