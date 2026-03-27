from flask import Blueprint, request, jsonify
from services.producttype_service import (
    get_all_producttype,
    get_producttype_by_id,
    insert_producttype,
    update_producttype,
    delete_producttype,
)
from services.auth_guard import require_admin

producttype_bp = Blueprint("producttype_bp", __name__)

@producttype_bp.route("/producttypes", methods=["GET"])
def list_producttypes():
    return jsonify(get_all_producttype()), 200

@producttype_bp.route("/producttypes/<int:pid>", methods=["GET"])
def get_producttype(pid):
    row = get_producttype_by_id(pid)
    if not row:
        return jsonify({"error": "productType not found"}), 404
    return jsonify(row), 200

@producttype_bp.route("/producttypes", methods=["POST"])
@require_admin
def create_producttype():
    data = request.get_json(silent=True)
    if not data or "type_name" not in data:
        return jsonify({"error": "Missing type_name"}), 400
    new_id = insert_producttype(data["type_name"])
    return jsonify({"message": "productType created", "productType_id": new_id}), 201

@producttype_bp.route("/producttypes/<int:pid>", methods=["PUT"])
@require_admin
def update_producttype_route(pid):
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400
    if not update_producttype(pid, data):
        return jsonify({"error": "productType not found"}), 404
    return jsonify({"message": "productType updated"}), 200

@producttype_bp.route("/producttypes/<int:pid>", methods=["DELETE"])
@require_admin
def delete_producttype_route(pid):
    if not delete_producttype(pid):
        return jsonify({"error": "productType not found"}), 404
    return jsonify({"message": "productType deleted"}), 200
