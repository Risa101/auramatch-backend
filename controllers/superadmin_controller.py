from flask import Blueprint, request, jsonify
from services.superadmin_service import (
    get_all_superadmin,
    get_superadmin_by_id,
    insert_superadmin,
    update_superadmin,
    delete_superadmin
)
from services.auth_guard import require_admin

superadmin_bp = Blueprint("superadmin_bp", __name__)

@superadmin_bp.route("/superadmin", methods=["GET"])
@require_admin
def list_superadmin():
    return jsonify(get_all_superadmin()), 200

@superadmin_bp.route("/superadmin/<int:sid>", methods=["GET"])
@require_admin
def get_superadmin(sid):
    row = get_superadmin_by_id(sid)
    if not row:
        return jsonify({"error": "not found"}), 404
    return jsonify(row), 200

@superadmin_bp.route("/superadmin", methods=["POST"])
@require_admin
def create_superadmin():
    data = request.get_json()
    new_id = insert_superadmin(data)
    return jsonify({"superadmin_id": new_id}), 201

@superadmin_bp.route("/superadmin/<int:sid>", methods=["PUT"])
@require_admin
def update_superadmin_route(sid):
    data = request.get_json()
    if not update_superadmin(sid, data):
        return jsonify({"error": "not found"}), 404
    return jsonify({"message": "updated"}), 200

@superadmin_bp.route("/superadmin/<int:sid>", methods=["DELETE"])
@require_admin
def delete_superadmin_route(sid):
    if not delete_superadmin(sid):
        return jsonify({"error": "not found"}), 404
    return jsonify({"message": "deleted"}), 200
