from flask import Blueprint, request, jsonify
from services.status_service import (
    get_all_status,
    get_status_by_id,
    insert_status,
    update_status,
    delete_status,
)

status_bp = Blueprint("status_bp", __name__)

@status_bp.route("/status", methods=["GET"])
def list_status():
    return jsonify(get_all_status()), 200

@status_bp.route("/status", methods=["POST"])
def create_status():
    data = request.get_json(silent=True)
    if not data or "status_name" not in data:
        return jsonify({"error": "status_name is required"}), 400

    new_id = insert_status(data["status_name"])
    return jsonify({"status_id": new_id}), 201

@status_bp.route("/status/<int:sid>", methods=["GET"])
def get_status(sid):
    row = get_status_by_id(sid)
    if not row:
        return jsonify({"error": "not found"}), 404
    return jsonify(row), 200

@status_bp.route("/status/<int:sid>", methods=["PUT"])
def update_status_route(sid):
    data = request.get_json(silent=True)
    if not update_status(sid, data):
        return jsonify({"error": "not found"}), 404
    return jsonify({"message": "updated"}), 200

@status_bp.route("/status/<int:sid>", methods=["DELETE"])
def delete_status_route(sid):
    if not delete_status(sid):
        return jsonify({"error": "not found"}), 404
    return jsonify({"message": "deleted"}), 200
