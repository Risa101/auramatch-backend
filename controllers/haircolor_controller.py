from flask import Blueprint, request, jsonify
from services.haircolor_service import (
    get_all_haircolor,
    get_haircolor_by_id,
    insert_haircolor,
    update_haircolor,
    delete_haircolor,
)

haircolor_bp = Blueprint("haircolor_bp", __name__)


@haircolor_bp.route("/haircolors", methods=["GET"])
def list_haircolors():
    rows = get_all_haircolor()
    return jsonify(rows), 200


@haircolor_bp.route("/haircolors", methods=["POST"])
def create_haircolor():
    data = request.get_json(silent=True)

    if not data or "haircolor_name" not in data:
        return jsonify({"error": "Missing haircolor_name"}), 400

    new_id = insert_haircolor(data["haircolor_name"])

    return jsonify({
        "message": "haircolor created successfully",
        "haircolor_id": new_id
    }), 201


@haircolor_bp.route("/haircolors/<int:pid>", methods=["GET"])
def get_haircolor(pid: int):
    row = get_haircolor_by_id(pid)
    if not row:
        return jsonify({"error": "haircolor not found"}), 404
    return jsonify(row), 200


@haircolor_bp.route("/haircolors/<int:pid>", methods=["PUT"])
def update_haircolor_route(pid: int):
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    updated = update_haircolor(pid, data)
    if not updated:
        return jsonify({"error": "haircolor not found"}), 404

    return jsonify({
        "message": "haircolor updated successfully",
        "haircolor_id": pid
    }), 200


@haircolor_bp.route("/haircolors/<int:pid>", methods=["DELETE"])
def delete_haircolor_route(pid: int):
    deleted = delete_haircolor(pid)
    if not deleted:
        return jsonify({"error": "haircolor not found"}), 404

    return jsonify({"message": "haircolor deleted"}), 200
