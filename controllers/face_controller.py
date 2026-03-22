from flask import Blueprint, request, jsonify
from services.face_service import (
    get_all_face,
    get_face_by_id,
    insert_face,
    update_face,
    delete_face,
)

face_bp = Blueprint("face_bp", __name__)

@face_bp.route("/faces", methods=["GET"])
def list_faces():
    rows = get_all_face()
    return jsonify(rows), 200


@face_bp.route("/faces", methods=["POST"])
def create_face():
    data = request.get_json(silent=True)

    if not data or "user_id" not in data:
        return jsonify({"error": "Missing required field: user_id"}), 400

    try:
        user_id = int(data["user_id"])
    except (TypeError, ValueError):
        return jsonify({"error": "Field 'user_id' must be an integer"}), 400

    new_id = insert_face(user_id)

    return jsonify({
        "message": "face created successfully",
        "face_id": new_id
    }), 201


@face_bp.route("/faces/<int:pid>", methods=["GET"])
def get_face(pid: int):
    row = get_face_by_id(pid)
    if not row:
        return jsonify({"error": "face not found"}), 404
    return jsonify(row), 200


@face_bp.route("/faces/<int:pid>", methods=["PUT"])
def update_face_route(pid: int):
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    updated = update_face(pid, data)

    if not updated:
        return jsonify({"error": "face not found or no data to update"}), 404

    return jsonify({
        "message": "face updated successfully",
        "face_id": pid
    }), 200


@face_bp.route("/faces/<int:pid>", methods=["DELETE"])
def delete_face_route(pid: int):
    deleted = delete_face(pid)

    if not deleted:
        return jsonify({"error": "face not found"}), 404

    return jsonify({"message": "face deleted"}), 200
