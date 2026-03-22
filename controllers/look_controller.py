from flask import Blueprint, jsonify, request
from services.look_service import get_looks_by_color

looks_bp = Blueprint("looks_bp", __name__)

@looks_bp.route("/looks", methods=["GET"])
def get_looks():
    # ดึงค่า 'Spring' หรือ 'Summer' จาก URL
    color_param = request.args.get('personal_color')
    
    if not color_param:
        return jsonify({"status": "success", "data": []})

    # ปรับเป็นตัวพิมพ์ใหญ่ตัวแรกให้ตรงกับ ENUM ในฐานข้อมูล
    formatted_color = color_param.capitalize()
    
    try:
        looks = get_looks_by_color(formatted_color)
        return jsonify({
            "status": "success",
            "data": looks
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500