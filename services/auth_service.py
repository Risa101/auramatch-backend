import os
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired


def _get_serializer():
    secret = os.getenv("SECRET_KEY", "dev-secret")
    return URLSafeTimedSerializer(secret, salt="auramatch-auth")


def generate_auth_token(payload):
    serializer = _get_serializer()
    return serializer.dumps(payload)


def verify_auth_token(token, max_age_seconds=60 * 60 * 24 * 7):
    serializer = _get_serializer()
    try:
        return serializer.loads(token, max_age=max_age_seconds)
    except (BadSignature, SignatureExpired):
        return None
