from flask import Blueprint, request, jsonify, g
import os
import re
from extensions import limiter
from services.user_service import (
    get_all_user, get_user_by_id, get_user_by_email,
    insert_user, update_user, delete_user,
    authenticate_user, set_user_password,
)
from services.auth_service import generate_auth_token, verify_auth_token
from services.auth_guard import require_auth, require_admin
from services.password_reset_service import (
    create_password_reset,
    consume_password_reset,
    send_reset_email,
)

user_bp = Blueprint("user_bp", __name__)

# =========================
# ADMIN: User CRUD
# =========================

@user_bp.route("/admin/users", methods=["GET"])
@require_admin
def list_users():
    return jsonify(get_all_user()), 200


@user_bp.route("/admin/users/<int:uid>", methods=["GET"])
@require_admin
def get_user(uid):
    row = get_user_by_id(uid)
    if not row:
        return jsonify({"error": "not found"}), 404
    return jsonify(row), 200


@user_bp.route("/admin/users", methods=["POST"])
@require_admin
def create_user():
    data = request.get_json()
    new_id = insert_user(data)
    return jsonify({"user_id": new_id}), 201


@user_bp.route("/admin/users/<int:uid>", methods=["PUT", "PATCH"])
@require_admin
def update_user_route(uid):
    data = request.get_json()
    if not update_user(uid, data):
        return jsonify({"error": "not found"}), 404
    return jsonify({"message": "updated"}), 200


@user_bp.route("/admin/users/<int:uid>", methods=["DELETE"])
@require_admin
def delete_user_route(uid):
    if not delete_user(uid):
        return jsonify({"error": "not found"}), 404
    return jsonify({"message": "deleted"}), 200


# =========================
# AUTH: login / register / me
# =========================

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
@user_bp.route("/api/login", methods=["POST"])
@limiter.limit("10 per minute")
def login_user():
    return _login_impl()


def _password_error(password):
    if len(password) < 8:
        return "password too short"
    if not re.search(r"[A-Za-z]", password) or not re.search(r"[0-9]", password):
        return "password must include letters and numbers"
    return ""


@user_bp.route("/register", methods=["POST"])
@user_bp.route("/api/register", methods=["POST"])
@limiter.limit("5 per minute")
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
@user_bp.route("/api/user", methods=["GET"])
@require_auth
def get_me():
    return jsonify(g.auth_payload), 200


@user_bp.route("/api/user/me", methods=["PUT", "PATCH"])
@require_auth
def update_me():
    data = request.get_json(silent=True) or {}
    uid = g.auth_payload.get("user_id")
    allowed = {}
    if "username" in data and data["username"]:
        allowed["username"] = str(data["username"]).strip()
    if "avatar" in data:
        allowed["avatar"] = str(data["avatar"]).strip()
    if not allowed:
        return jsonify({"error": "No valid fields to update"}), 400
    update_user(uid, allowed)
    row = get_user_by_id(uid)
    if row:
        row.pop("password", None)
    return jsonify({"message": "Profile updated", "user": row}), 200


@user_bp.route("/api/upload/avatar", methods=["POST"])
@require_auth
def upload_avatar():
    import uuid
    from werkzeug.utils import secure_filename
    ALLOWED = {"image/jpeg", "image/png", "image/webp", "image/gif"}
    MAX_SIZE = 5 * 1024 * 1024  # 5MB

    file = request.files.get("avatar")
    if not file:
        return jsonify({"error": "No file provided"}), 400
    mime = (file.mimetype or "").lower()
    if mime not in ALLOWED:
        return jsonify({"error": "Only JPEG, PNG, WebP, GIF allowed"}), 400
    data = file.read()
    if len(data) > MAX_SIZE:
        return jsonify({"error": "File too large (max 5MB)"}), 400

    ext = mime.split("/")[-1].replace("jpeg", "jpg")
    filename = f"{g.auth_payload.get('user_id')}_{uuid.uuid4().hex[:8]}.{ext}"
    save_dir = os.path.join(os.path.dirname(__file__), "..", "static", "images", "avatars")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, filename), "wb") as f:
        f.write(data)

    avatar_url = f"/avatars/{filename}"
    return jsonify({"avatar_url": avatar_url}), 200


@user_bp.route("/api/firebase-sync", methods=["POST"])
def firebase_sync():
    import secrets
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    name = (data.get("name") or "").strip()
    photo_url = (data.get("photo_url") or "").strip()

    if not email:
        return jsonify({"error": "email required"}), 400

    user = get_user_by_email(email)
    if not user:
        new_id = insert_user({
            "username": name or email.split("@")[0],
            "email": email,
            "password": secrets.token_hex(16),
            "avatar": photo_url,
        })
        user = get_user_by_id(new_id)

    if user:
        user.pop("password", None)

    payload = user or {"user_id": None, "email": email, "username": name, "role": "user"}
    token = generate_auth_token({
        "user_id": payload.get("user_id"),
        "email": payload.get("email"),
        "username": payload.get("username"),
        "role": payload.get("role") or "user",
    })
    return jsonify({"token": token, "user": payload}), 200


# =========================
# Password reset
# =========================

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
