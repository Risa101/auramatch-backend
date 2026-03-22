from flask import Blueprint, request, jsonify
import os
import re
from services.user_service import *
from services.auth_service import generate_auth_token, verify_auth_token
from services.auth_guard import require_admin
from services.password_reset_service import (
    create_password_reset,
    consume_password_reset,
    send_reset_email,
)

user_bp = Blueprint("user_bp", __name__)

@user_bp.route("/users", methods=["GET"])
def list_users():
    return jsonify(get_all_user()), 200

@user_bp.route("/users/<int:uid>", methods=["GET"])
def get_user(uid):
    row = get_user_by_id(uid)
    if not row:
        return jsonify({"error": "not found"}), 404
    return jsonify(row), 200

@user_bp.route("/api/users", methods=["GET"])
def list_users_api():
    return jsonify(get_all_user()), 200

@user_bp.route("/api/users/<int:uid>", methods=["GET"])
def get_user_api(uid):
    row = get_user_by_id(uid)
    if not row:
        return jsonify({"error": "not found"}), 404
    return jsonify(row), 200

@user_bp.route("/users", methods=["POST"])
def create_user():
    data = request.get_json()
    new_id = insert_user(data)
    return jsonify({"user_id": new_id}), 201

@user_bp.route("/users/<int:uid>", methods=["PUT"])
def update_user_route(uid):
    data = request.get_json()
    if not update_user(uid, data):
        return jsonify({"error": "not found"}), 404
    return jsonify({"message": "updated"}), 200

@user_bp.route("/users/<int:uid>", methods=["DELETE"])
def delete_user_route(uid):
    if not delete_user(uid):
        return jsonify({"error": "not found"}), 404
    return jsonify({"message": "deleted"}), 200

@user_bp.route("/api/users", methods=["POST"])
def create_user_api():
    data = request.get_json()
    new_id = insert_user(data)
    return jsonify({"user_id": new_id}), 201

@user_bp.route("/api/users/<int:uid>", methods=["PUT"])
def update_user_api(uid):
    data = request.get_json()
    if not update_user(uid, data):
        return jsonify({"error": "not found"}), 404
    return jsonify({"message": "updated"}), 200

@user_bp.route("/api/users/<int:uid>", methods=["DELETE"])
def delete_user_api(uid):
    if not delete_user(uid):
        return jsonify({"error": "not found"}), 404
    return jsonify({"message": "deleted"}), 200

@user_bp.route("/admin/users", methods=["GET"])
@require_admin
def list_users_admin():
    return jsonify(get_all_user()), 200

@user_bp.route("/admin/users/<int:uid>", methods=["GET"])
@require_admin
def get_user_admin(uid):
    row = get_user_by_id(uid)
    if not row:
        return jsonify({"error": "not found"}), 404
    return jsonify(row), 200

@user_bp.route("/admin/users", methods=["POST"])
@require_admin
def create_user_admin():
    data = request.get_json()
    new_id = insert_user(data)
    return jsonify({"user_id": new_id}), 201

@user_bp.route("/admin/users/<int:uid>", methods=["PUT"])
@require_admin
def update_user_admin(uid):
    data = request.get_json()
    if not update_user(uid, data):
        return jsonify({"error": "not found"}), 404
    return jsonify({"message": "updated"}), 200

@user_bp.route("/admin/users/<int:uid>", methods=["DELETE"])
@require_admin
def delete_user_admin(uid):
    if not delete_user(uid):
        return jsonify({"error": "not found"}), 404
    return jsonify({"message": "deleted"}), 200


def _login_impl():
    data = request.get_json(silent=True) or {}
    identifier = data.get("email") or data.get("username")
    password = data.get("password")
    if not identifier or not password:
        return jsonify({"error": "email/username and password required"}), 400

    user = authenticate_user(identifier, password)
    if not user:
        return jsonify({"error": "invalid credentials"}), 401
    token = generate_auth_token({
        "user_id": user.get("user_id"),
        "email": user.get("email"),
        "username": user.get("username"),
        "role": user.get("role") or "user",
    })
    return jsonify({"token": token, "user": user}), 200


@user_bp.route("/login", methods=["POST"])
def login_user():
    return _login_impl()


@user_bp.route("/api/login", methods=["POST"])
def login_user_api():
    return _login_impl()


def _password_error(password):
    if len(password) < 8:
        return "password too short"
    if not re.search(r"[A-Za-z]", password) or not re.search(r"[0-9]", password):
        return "password must include letters and numbers"
    return ""


@user_bp.route("/register", methods=["POST"])
def register_user():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password")
    username = (data.get("username") or data.get("name") or "").strip()

    if not email or not password:
        return jsonify({"error": "email and password required"}), 400
    pw_error = _password_error(password)
    if pw_error:
        return jsonify({"error": pw_error}), 400
    if get_user_by_email(email):
        return jsonify({"error": "email already exists"}), 409

    new_id = insert_user({
        "username": username or None,
        "email": email,
        "password": password,
    })
    row = get_user_by_id(new_id)
    if row:
        row.pop("password", None)
    payload = row or {"user_id": new_id, "email": email, "username": username, "role": "user"}
    token = generate_auth_token({
        "user_id": payload.get("user_id"),
        "email": payload.get("email"),
        "username": payload.get("username"),
        "role": payload.get("role") or "user",
    })
    return jsonify({"token": token, "user": payload}), 201


@user_bp.route("/me", methods=["GET"])
def get_me():
    auth_header = request.headers.get("Authorization", "")
    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return jsonify({"error": "missing token"}), 401
    payload = verify_auth_token(parts[1])
    if not payload:
        return jsonify({"error": "invalid token"}), 401
    return jsonify(payload), 200


@user_bp.route("/api/user", methods=["GET"])
def get_me_api():
    return get_me()


@user_bp.route("/password/forgot", methods=["POST"])
def forgot_password():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    if not email:
        return jsonify({"error": "email required"}), 400

    user = get_user_by_email(email)
    reset_link = None
    email_sent = False

    if user:
        token, _ = create_password_reset(user["user_id"])
        frontend_base = os.getenv("FRONTEND_BASE_URL", "http://127.0.0.1:5174/AURAMATCH-VER2")
        reset_link = f"{frontend_base.rstrip('/')}/reset-password?token={token}"
        try:
            email_sent = send_reset_email(email, reset_link)
        except Exception:
            email_sent = False

    response = {"message": "if email exists, a reset link will be sent", "email_sent": email_sent}
    if reset_link and (email_sent is False or os.getenv("RESET_DEBUG_RETURN_LINK") == "1"):
        response["reset_link"] = reset_link
    return jsonify(response), 200


@user_bp.route("/password/reset", methods=["POST"])
def reset_password():
    data = request.get_json(silent=True) or {}
    token = (data.get("token") or "").strip()
    password = data.get("password") or ""
    if not token or not password:
        return jsonify({"error": "token and password required"}), 400
    pw_error = _password_error(password)
    if pw_error:
        return jsonify({"error": pw_error}), 400

    user_id = consume_password_reset(token)
    if not user_id:
        return jsonify({"error": "invalid or expired token"}), 400

    if not set_user_password(user_id, password):
        return jsonify({"error": "failed to update password"}), 500
    return jsonify({"message": "password updated"}), 200
