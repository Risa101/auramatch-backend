from flask import Blueprint, request, jsonify
from services.brand_service import (
    get_all_brands,
    get_brand_by_id,
    insert_brand,
    update_brand,
    delete_brand
)
from services.auth_guard import require_admin

brand_bp = Blueprint("brand_bp", __name__)


@brand_bp.route("/brands", methods=["GET"])
def api_get_brands():
    rows = get_all_brands()
    return jsonify(rows), 200


@brand_bp.route("/brands/<int:bid>", methods=["GET"])
def api_get_brand_by_id(bid):
    row = get_brand_by_id(bid)
    if not row:
        return jsonify({"message": "Brand not found"}), 404
    return jsonify(row), 200


@brand_bp.route("/brands", methods=["POST"])
@require_admin
def api_insert_brand():
    data = request.json
    name = data.get("brand_name")
    logo = data.get("logo_path")

    if not name or not logo:
        return jsonify({"message": "brand_name and logo_path required"}), 400

    new_id = insert_brand(name, logo)
    return jsonify({"message": "Brand created", "brand_id": new_id}), 201


@brand_bp.route("/brands/<int:bid>", methods=["PUT"])
@require_admin
def api_update_brand(bid):
    data = request.json
    updated = update_brand(bid, data)

    if not updated:
        return jsonify({"message": "Brand not found"}), 404

    return jsonify({"message": "Brand updated"}), 200


@brand_bp.route("/brands/<int:bid>", methods=["DELETE"])
@require_admin
def api_delete_brand(bid):
    deleted = delete_brand(bid)

    if not deleted:
        return jsonify({"message": "Brand not found"}), 404

    return jsonify({"message": "Brand deleted"}), 200
