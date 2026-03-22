from functools import wraps

from flask import jsonify, request, g

from services.auth_service import verify_auth_token


def _get_bearer_token():
    auth_header = request.headers.get("Authorization", "")
    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1]


def require_auth(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        token = _get_bearer_token()
        if not token:
            return jsonify({"error": "missing token"}), 401
        payload = verify_auth_token(token)
        if not payload:
            return jsonify({"error": "invalid token"}), 401
        g.auth_payload = payload
        return fn(*args, **kwargs)
    return wrapper


def require_admin(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        token = _get_bearer_token()
        if not token:
            return jsonify({"error": "missing token"}), 401
        payload = verify_auth_token(token)
        if not payload:
            return jsonify({"error": "invalid token"}), 401
        if payload.get("role") != "admin":
            return jsonify({"error": "forbidden"}), 403
        g.auth_payload = payload
        return fn(*args, **kwargs)
    return wrapper
