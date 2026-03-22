from flask import Blueprint, request, jsonify
from services.stock_service import (
    get_all_stock,
    get_stock_by_id,
    insert_stock,
    update_stock,
    delete_stock,
)

stock_bp = Blueprint("stock_bp", __name__)

@stock_bp.route("/stock", methods=["GET"])
def list_stock():
    return jsonify(get_all_stock()), 200

@stock_bp.route("/stock", methods=["POST"])
def create_stock():
    data = request.get_json(silent=True)
    if not data or "product_id" not in data or "quantity" not in data:
        return jsonify({"error": "product_id and quantity are required"}), 400

    new_id = insert_stock(data["product_id"], data["quantity"])
    return jsonify({"stock_id": new_id}), 201

@stock_bp.route("/stock/<int:sid>", methods=["GET"])
def get_stock(sid):
    row = get_stock_by_id(sid)
    if not row:
        return jsonify({"error": "not found"}), 404
    return jsonify(row), 200

@stock_bp.route("/stock/<int:sid>", methods=["PUT"])
def update_stock_route(sid):
    data = request.get_json(silent=True)
    if not update_stock(sid, data):
        return jsonify({"error": "not found"}), 404
    return jsonify({"message": "updated"}), 200

@stock_bp.route("/stock/<int:sid>", methods=["DELETE"])
def delete_stock_route(sid):
    if not delete_stock(sid):
        return jsonify({"error": "not found"}), 404
    return jsonify({"message": "deleted"}), 200
