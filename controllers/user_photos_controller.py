from flask import Blueprint, request, jsonify
from services.user_photos_service import *

user_photos_bp = Blueprint("user_photos_bp", __name__)

@user_photos_bp.route("/user-photos", methods=["GET"])
def list_user_photos():
    return jsonify(get_all_user_photos()), 200

@user_photos_bp.route("/user-photos", methods=["POST"])
def create_user_photo():
    data = request.get_json()
    new_id = insert_user_photo(data)
    return jsonify({"user_photo_id": new_id}), 201
