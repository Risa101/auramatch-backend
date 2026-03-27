from flask import Blueprint, request, jsonify
from services.hairstyle_service import (
    get_all_hairstyle,
    get_hairstyle_by_id,
    insert_hairstyle,
    update_hairstyle,
    delete_hairstyle,
)
from services.auth_guard import require_admin

hairstyle_bp = Blueprint("hairstyle_bp", __name__)

@hairstyle_bp.route("/hairstyles", methods=["GET"])
def list_hairstyles():
    rows = get_all_hairstyle()
    return jsonify(rows), 200


@hairstyle_bp.route("/hairstyles", methods=["POST"])
@require_admin
def create_hairstyle():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Invalid or empty JSON payload"}), 400

    if "face_id" not in data or "hairstyle_name" not in data:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        face_id = int(data["face_id"])
    except (TypeError, ValueError):
        return jsonify({"error": "face_id must be integer"}), 400

    hairstyle_name = data["hairstyle_name"]

    new_id = insert_hairstyle(
        face_id=face_id,
        hairstyle_name=hairstyle_name
    )

    return jsonify({
        "message": "hairstyle created successfully",
        "hairstyle_id": new_id
    }), 201


@hairstyle_bp.route("/hairstyles/<int:pid>", methods=["GET"])
def get_hairstyle(pid: int):
    row = get_hairstyle_by_id(pid)
    if not row:
        return jsonify({"error": "hairstyle not found"}), 404
    return jsonify(row), 200


@hairstyle_bp.route("/hairstyles/<int:pid>", methods=["PUT"])
@require_admin
def update_hairstyle_route(pid: int):
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    updated = update_hairstyle(pid, data)

    if not updated:
        return jsonify({"error": "hairstyle not found"}), 404

    return jsonify({
        "message": "hairstyle updated successfully",
        "hairstyle_id": pid
    }), 200


@hairstyle_bp.route("/hairstyles/<int:pid>", methods=["DELETE"])
@require_admin
def delete_hairstyle_route(pid: int):
    deleted = delete_hairstyle(pid)

    if not deleted:
        return jsonify({"error": "hairstyle not found"}), 404

    return jsonify({"message": "hairstyle deleted"}), 200
