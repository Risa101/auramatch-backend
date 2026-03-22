from flask import Blueprint, request, jsonify
from services.eyebrow_service import (
    get_all_eyebrows,
    get_eyebrow_by_id,
    insert_eyebrow,
    update_eyebrow,
    delete_eyebrow,
)

eyebrows_bp = Blueprint("eyebrows_bp", __name__)

# ============================================================
#   GET /eyebrows (list all) + POST /eyebrows (create)
# ============================================================
@eyebrows_bp.route("/eyebrows", methods=["GET"])
def list_eyebrows():
    rows = get_all_eyebrows()
    return jsonify(rows), 200

@eyebrows_bp.route("/eyebrows", methods=["POST"])
def create_eyebrow():
    """เพิ่มสินค้าใหม่ลงฐานข้อมูล"""

    # ดึง JSON จาก body
    data = request.get_json(silent=True)

    # ไม่มี / parse ไม่ได้ → 400
    if not data:
        return jsonify({"error": "Invalid or empty JSON payload"}), 400

    # ฟิลด์ที่ต้องมี
    required = ["name", "price"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    # แปลงราคาให้เป็นตัวเลข
    try:
        price = float(data["price"])
    except (TypeError, ValueError):
        return jsonify({"error": "Field 'price' must be a number"}), 400

    name = data["name"]
    image = data.get("image", "")

    # เรียก service ไป insert
    new_id = insert_eyebrow(
        name=name,
        price=price,
        image=image,
    )

    return jsonify({
        "message": "eyebrow created successfully",
        "eyebrow_id": new_id
    }), 201

# ============================================================
#                  GET /eyebrows/<id>
#                  PUT /eyebrows/<id>
#                DELETE /eyebrows/<id>
# ============================================================
@eyebrows_bp.route("/eyebrows/<int:pid>", methods=["GET"])
def get_eyebrow(pid: int):
    row = get_eyebrow_by_id(pid)
    if not row:
        return jsonify({"error": "eyebrow not found"}), 404
    return jsonify(row), 200


@eyebrows_bp.route("/eyebrows/<int:pid>", methods=["PUT"])
def update_eyebrow_route(pid: int):
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    updated = update_eyebrow(pid, data)

    if not updated:
        return jsonify({"error": "eyebrow not found"}), 404

    return jsonify({
        "message": "eyebrow updated successfully",
        "eyebrow_id": pid
    }), 200


@eyebrows_bp.route("/eyebrows/<int:pid>", methods=["DELETE"])
def delete_eyebrow_route(pid: int):
    deleted = delete_eyebrow(pid)

    if not deleted:
        return jsonify({"error": "eyebrow not found"}), 404

    return jsonify({"message": "eyebrow deleted"}), 200
