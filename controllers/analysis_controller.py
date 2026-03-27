from flask import Blueprint, request, jsonify, g
from services.analysis_service import save_analysis, get_history_by_user, delete_analysis_by_id
from services.auth_guard import require_auth

analysis_bp = Blueprint('analysis', __name__)

@analysis_bp.route('/save-analysis', methods=['POST'])
@require_auth
def create_analysis():
    data = request.json
    if not data or 'user_id' not in data:
        return jsonify({"error": "Missing required data"}), 400
    # ป้องกัน user บันทึกในนาม user อื่น
    if int(data.get("user_id", 0)) != g.auth_payload.get("user_id"):
        return jsonify({"error": "forbidden"}), 403
    result = save_analysis(data)
    if result.get("success"):
        return jsonify(result), 201
    return jsonify(result), 500

@analysis_bp.route('/analysis-history/<int:user_id>', methods=['GET'])
@require_auth
def get_user_history(user_id):
    if user_id != g.auth_payload.get("user_id"):
        return jsonify({"error": "forbidden"}), 403
    try:
        history = get_history_by_user(user_id)
        return jsonify(history if history is not None else []), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@analysis_bp.route('/analysis-history/<int:analysis_id>', methods=['DELETE'])
@require_auth
def delete_analysis(analysis_id):
    result = delete_analysis_by_id(analysis_id, user_id=g.auth_payload.get("user_id"))
    if result.get("success"):
        return jsonify(result), 200
    return jsonify(result), 404 if "not found" in result.get("message", "").lower() else 500