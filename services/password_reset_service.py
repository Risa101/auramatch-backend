import hashlib
import os
import secrets
import smtplib
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage

from db import get_conn


def _hash_token(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _reset_ttl_seconds():
    return int(os.getenv("RESET_TOKEN_TTL_SECONDS", "3600"))


def create_password_reset(user_id):
    token = secrets.token_urlsafe(32)
    token_hash = _hash_token(token)
    ttl_seconds = _reset_ttl_seconds()
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE password_resets SET used_at=UTC_TIMESTAMP() WHERE user_id=%s AND used_at IS NULL",
        (user_id,),
    )
    cur.execute(
        """
        INSERT INTO password_resets (user_id, token_hash, expires_at)
        VALUES (%s, %s, DATE_ADD(UTC_TIMESTAMP(), INTERVAL %s SECOND))
        """,
        (user_id, token_hash, ttl_seconds),
    )
    conn.commit()
    cur.close()
    conn.close()
    return token, expires_at


def consume_password_reset(token):
    token_hash = _hash_token(token)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT reset_id, user_id
        FROM password_resets
        WHERE token_hash=%s AND used_at IS NULL AND expires_at > UTC_TIMESTAMP()
        LIMIT 1
        """,
        (token_hash,),
    )
    row = cur.fetchone()
    if not row:
        cur.close()
        conn.close()
        return None

    cur.execute(
        "UPDATE password_resets SET used_at=UTC_TIMESTAMP() WHERE reset_id=%s",
        (row["reset_id"],),
    )
    conn.commit()
    cur.close()
    conn.close()
    return row["user_id"]


def send_reset_email(to_email, reset_link):
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASS")
    sender = os.getenv("SMTP_FROM", user or "")
    use_tls = os.getenv("SMTP_USE_TLS", "true").lower() == "true"

    if not host or not sender:
        return False

    msg = EmailMessage()
    msg["Subject"] = "Reset your AuraMatch password"
    msg["From"] = sender
    msg["To"] = to_email
    msg.set_content(
        "You requested a password reset.\n\n"
        f"Reset link: {reset_link}\n\n"
        "If you did not request this, you can ignore this email."
    )

    with smtplib.SMTP(host, port) as smtp:
        if use_tls:
            smtp.starttls()
        if user and password:
            smtp.login(user, password)
        smtp.send_message(msg)
    return True
