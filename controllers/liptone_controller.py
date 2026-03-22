from flask import Blueprint, request, jsonify
from services.liptone_service import *

liptone_bp = Blueprint("liptone_bp", __name__)

@liptone_bp.route("/liptones", methods=["GET"])
def list_liptones():
    return jsonify(get_all_liptone()), 200

@liptone_bp.route("/liptones/<int:pid>", methods=["GET"])
def get_liptone(pid):
    row = get_liptone_by_id(pid)
    if not row:
        return jsonify({"error": "not found"}), 404
    return jsonify(row), 200

@liptone_bp.route("/liptones", methods=["POST"])
def create_liptone():
    data = request.get_json()
    new_id = insert_liptone(data["liptone_name"])
    return jsonify({"liptone_id": new_id}), 201

@liptone_bp.route("/liptones/<int:pid>", methods=["PUT"])
def update_liptone_route(pid):
    if not update_liptone(pid, request.get_json()):
        return jsonify({"error": "not found"}), 404
    return jsonify({"message": "updated"}), 200

@liptone_bp.route("/liptones/<int:pid>", methods=["DELETE"])
def delete_liptone_route(pid):
    if not delete_liptone(pid):
        return jsonify({"error": "not found"}), 404
    return jsonify({"message": "deleted"}), 200
