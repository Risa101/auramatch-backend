# controllers/youtube_controller.py
from flask import Blueprint, jsonify
from services.youtube_service import get_youtube_videos

youtube_bp = Blueprint("youtube_bp", __name__)

@youtube_bp.route("/youtube", methods=["GET"])
def get_youtube():
    data = get_youtube_videos()
    return jsonify({
        "status": "success",
        "data": data
    })
