from flask import Blueprint, request, jsonify
from services.faceproduct_service import (
    get_all_faceproduct,
    get_faceproduct_by_id,
    insert_faceproduct,
    update_faceproduct,
    delete_faceproduct,
)

faceproduct_bp = Blueprint("faceproduct_bp", __name__)

@faceproduct_bp.route("/faceproducts", methods=["GET"])
def list_faceproducts():
    rows = get_all_faceproduct()
    return jsonify(rows), 200


@faceproduct_bp.route("/faceproducts", methods=["POST"])
def create_faceproduct():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Invalid or empty JSON payload"}), 400

    required = ["face_id", "product_id"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    try:
        face_id = int(data["face_id"])
        product_id = int(data["product_id"])
    except (TypeError, ValueError):
        return jsonify({"error": "face_id and product_id must be integers"}), 400

    new_id = insert_faceproduct(
        face_id=face_id,
        product_id=product_id
    )

    return jsonify({
        "message": "face_product created successfully",
        "face_product_id": new_id
    }), 201


@faceproduct_bp.route("/faceproducts/<int:pid>", methods=["GET"])
def get_faceproduct(pid: int):
    row = get_faceproduct_by_id(pid)
    if not row:
        return jsonify({"error": "face_product not found"}), 404
    return jsonify(row), 200


@faceproduct_bp.route("/faceproducts/<int:pid>", methods=["PUT"])
def update_faceproduct_route(pid: int):
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    updated = update_faceproduct(pid, data)

    if not updated:
        return jsonify({"error": "face_product not found"}), 404

    return jsonify({
        "message": "face_product updated successfully",
        "face_product_id": pid
    }), 200


@faceproduct_bp.route("/faceproducts/<int:pid>", methods=["DELETE"])
def delete_faceproduct_route(pid: int):
    deleted = delete_faceproduct(pid)

    if not deleted:
        return jsonify({"error": "face_product not found"}), 404

    return jsonify({"message": "face_product deleted"}), 200
