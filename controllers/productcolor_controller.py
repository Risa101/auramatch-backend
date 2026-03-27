from flask import Blueprint, request, jsonify
from services.productcolor_service import (
    get_all_productcolor,
    get_productcolor_by_id,
    insert_productcolor,
    update_productcolor,
    delete_productcolor,
)
from services.auth_guard import require_admin

productcolor_bp = Blueprint("productcolor_bp", __name__)

@productcolor_bp.route("/productcolors", methods=["GET"])
def list_productcolors():
    return jsonify(get_all_productcolor()), 200

@productcolor_bp.route("/productcolors/<int:pid>", methods=["GET"])
def get_productcolor(pid):
    row = get_productcolor_by_id(pid)
    if not row:
        return jsonify({"error": "Not found"}), 404
    return jsonify(row), 200

@productcolor_bp.route("/productcolors", methods=["POST"])
@require_admin
def create_productcolor():
    data = request.get_json(silent=True)
    if not data or "productColor_name" not in data:
        return jsonify({"error": "Missing productColor_name"}), 400
    new_id = insert_productcolor(data["productColor_name"])
    return jsonify({"message": "productColor created", "productColor_id": new_id}), 201

@productcolor_bp.route("/productcolors/<int:pid>", methods=["PUT"])
@require_admin
def update_productcolor_route(pid):
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400
    if not update_productcolor(pid, data):
        return jsonify({"error": "Not found"}), 404
    return jsonify({"message": "productColor updated"}), 200

@productcolor_bp.route("/productcolors/<int:pid>", methods=["DELETE"])
@require_admin
def delete_productcolor_route(pid):
    if not delete_productcolor(pid):
        return jsonify({"error": "Not found"}), 404
    return jsonify({"message": "productColor deleted"}), 200
