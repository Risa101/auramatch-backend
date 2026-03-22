from flask import Blueprint, request, jsonify
from services.eyeshape_service import (
    get_all_eyeshape,
    get_eyeshape_by_id,
    insert_eyeshape,
    update_eyeshape,
    delete_eyeshape,
)

eyeshape_bp = Blueprint("eyeshape_bp", __name__)

# ============================================================
#   GET /eyeshapes (list all) + POST /eyeshapes (create)
# ============================================================
@eyeshape_bp.route("/eyeshapes", methods=["GET"])
def list_eyeshapes():
    rows = get_all_eyeshape()
    return jsonify(rows), 200


@eyeshape_bp.route("/eyeshapes", methods=["POST"])
def create_eyeshape():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Invalid or empty JSON payload"}), 400

    required = ["face_id", "shape_name"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    try:
        face_id = int(data["face_id"])
    except (TypeError, ValueError):
        return jsonify({"error": "Field 'face_id' must be an integer"}), 400

    shape_name = data["shape_name"]

    new_id = insert_eyeshape(
        face_id=face_id,
        shape_name=shape_name
    )

    return jsonify({
        "message": "eyeshape created successfully",
        "eyeshape_id": new_id
    }), 201


# ============================================================
#                  GET /eyeshapes/<id>
#                  PUT /eyeshapes/<id>
#                DELETE /eyeshapes/<id>
# ============================================================
@eyeshape_bp.route("/eyeshapes/<int:pid>", methods=["GET"])
def get_eyeshape(pid: int):
    row = get_eyeshape_by_id(pid)
    if not row:
        return jsonify({"error": "eyeshape not found"}), 404
    return jsonify(row), 200


@eyeshape_bp.route("/eyeshapes/<int:pid>", methods=["PUT"])
def update_eyeshape_route(pid: int):
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    updated = update_eyeshape(pid, data)

    if not updated:
        return jsonify({"error": "eyeshape not found"}), 404

    return jsonify({
        "message": "eyeshape updated successfully",
        "eyeshape_id": pid
    }), 200


@eyeshape_bp.route("/eyeshapes/<int:pid>", methods=["DELETE"])
def delete_eyeshape_route(pid: int):
    deleted = delete_eyeshape(pid)

    if not deleted:
        return jsonify({"error": "eyeshape not found"}), 404

    return jsonify({"message": "eyeshape deleted"}), 200
