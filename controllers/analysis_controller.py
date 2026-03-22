from flask import Blueprint, request, jsonify
from services.analysis_service import save_analysis, get_history_by_user

# ตั้งชื่อ Blueprint
analysis_bp = Blueprint('analysis', __name__)

# 1. สำหรับบันทึกข้อมูล (POST)
@analysis_bp.route('/save-analysis', methods=['POST'])
def create_analysis():
    data = request.json
    
    if not data or 'user_id' not in data:
        return jsonify({"error": "Missing required data"}), 400
        
    result = save_analysis(data)
    
    if result.get("success"):
        return jsonify(result), 201
    else:
        return jsonify(result), 500

# 2. สำหรับดึงประวัติ (GET) 
# ปรับ Path ให้ตรงกับ Frontend: /api/analysis-history/<user_id>
@analysis_bp.route('/analysis-history/<int:user_id>', methods=['GET'])
def get_user_history(user_id):
    try:
        history = get_history_by_user(user_id)
        # ตรวจสอบว่า history เป็น None หรือไม่ ถ้าใช่ให้ส่ง []
        return jsonify(history if history is not None else []), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500