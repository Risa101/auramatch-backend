from flask import Blueprint, request, jsonify
from services.facetype_service import (
    get_all_facetype,
    get_facetype_by_id,
    insert_facetype,
    update_facetype,
    delete_facetype,
)

facetype_bp = Blueprint("facetype_bp", __name__)

@facetype_bp.route("/facetypes", methods=["GET"])
def list_facetypes():
    rows = get_all_facetype()
    return jsonify(rows), 200


@facetype_bp.route("/facetypes", methods=["POST"])
def create_facetype():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    if "face_id" not in data or "facetype_name" not in data:
        return jsonify({"error": "Missing required fields"}), 400

    new_id = insert_facetype(
        face_id=int(data["face_id"]),
        facetype_name=data["facetype_name"]
    )

    return jsonify({
        "message": "facetype created",
        "facetype_id": new_id
    }), 201


@facetype_bp.route("/facetypes/<int:fid>", methods=["GET"])
def get_facetype(fid):
    row = get_facetype_by_id(fid)
    if not row:
        return jsonify({"error": "facetype not found"}), 404
    return jsonify(row), 200


@facetype_bp.route("/facetypes/<int:fid>", methods=["PUT"])
def update_facetype_route(fid):
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    updated = update_facetype(fid, data)
    if not updated:
        return jsonify({"error": "facetype not found"}), 404

    return jsonify({"message": "facetype updated"}), 200


@facetype_bp.route("/facetypes/<int:fid>", methods=["DELETE"])
def delete_facetype_route(fid):
    deleted = delete_facetype(fid)
    if not deleted:
        return jsonify({"error": "facetype not found"}), 404

    return jsonify({"message": "facetype deleted"}), 200
