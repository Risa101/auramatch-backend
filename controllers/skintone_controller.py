from flask import Blueprint, request, jsonify
from services.skintone_service import (
    get_all_skintone,
    get_skintone_by_id,
    insert_skintone,
    update_skintone,
    delete_skintone,
)

skintone_bp = Blueprint("skintone_bp", __name__)

@skintone_bp.route("/skintones", methods=["GET"])
def list_skintones():
    return jsonify(get_all_skintone()), 200

@skintone_bp.route("/skintones", methods=["POST"])
def create_skintone():
    data = request.get_json(silent=True)
    if not data or "skintone_name" not in data:
        return jsonify({"error": "skintone_name is required"}), 400

    new_id = insert_skintone(data["skintone_name"])
    return jsonify({"skintone_id": new_id}), 201

@skintone_bp.route("/skintones/<int:sid>", methods=["GET"])
def get_skintone(sid):
    row = get_skintone_by_id(sid)
    if not row:
        return jsonify({"error": "not found"}), 404
    return jsonify(row), 200

@skintone_bp.route("/skintones/<int:sid>", methods=["PUT"])
def update_skintone_route(sid):
    data = request.get_json(silent=True)
    if not update_skintone(sid, data):
        return jsonify({"error": "not found"}), 404
    return jsonify({"message": "updated"}), 200

@skintone_bp.route("/skintones/<int:sid>", methods=["DELETE"])
def delete_skintone_route(sid):
    if not delete_skintone(sid):
        return jsonify({"error": "not found"}), 404
    return jsonify({"message": "deleted"}), 200
