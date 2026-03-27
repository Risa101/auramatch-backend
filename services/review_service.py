from db import get_conn
import pymysql

# Fields that users are allowed to update on their own reviews
_ALLOWED_UPDATE_FIELDS = {"rating", "comment"}


def get_all_reviews():
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT * FROM review")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_review_by_id(rid: int):
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT * FROM review WHERE review_id=%s", (rid,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def insert_review(user_id, product_id, rating, comment=None):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO review (user_id, product_id, rating, comment) VALUES (%s, %s, %s, %s)",
        (user_id, product_id, rating, comment)
    )
    conn.commit()
    new_id = cur.lastrowid
    cur.close()
    conn.close()
    return new_id


def update_review_by_owner(rid: int, user_id: int, data: dict):
    """Update a review only if the requester owns it. Returns 'ok', 'not_found', or 'forbidden'."""
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT user_id FROM review WHERE review_id=%s", (rid,))
    row = cur.fetchone()
    if not row:
        cur.close()
        conn.close()
        return "not_found"
    if row["user_id"] != user_id:
        cur.close()
        conn.close()
        return "forbidden"

    # Only allow whitelisted fields to prevent overwriting user_id etc.
    safe = {k: v for k, v in data.items() if k in _ALLOWED_UPDATE_FIELDS}
    if not safe:
        cur.close()
        conn.close()
        return "ok"

    fields = [f"{k}=%s" for k in safe]
    values = list(safe.values()) + [rid]
    cur.execute(f"UPDATE review SET {', '.join(fields)} WHERE review_id=%s", values)
    conn.commit()
    cur.close()
    conn.close()
    return "ok"


def delete_review_by_owner(rid: int, user_id: int):
    """Delete a review only if the requester owns it. Returns 'ok', 'not_found', or 'forbidden'."""
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT user_id FROM review WHERE review_id=%s", (rid,))
    row = cur.fetchone()
    if not row:
        cur.close()
        conn.close()
        return "not_found"
    if row["user_id"] != user_id:
        cur.close()
        conn.close()
        return "forbidden"

    cur.execute("DELETE FROM review WHERE review_id=%s", (rid,))
    conn.commit()
    cur.close()
    conn.close()
    return "ok"


def admin_update_review(rid: int, data: dict):
    """Admin: update any review field (whitelist still applied for safety)."""
    safe = {k: v for k, v in data.items() if k in _ALLOWED_UPDATE_FIELDS}
    if not safe:
        return False
    fields = [f"{k}=%s" for k in safe]
    values = list(safe.values()) + [rid]
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"UPDATE review SET {', '.join(fields)} WHERE review_id=%s", values)
    conn.commit()
    affected = cur.rowcount
    cur.close()
    conn.close()
    return affected > 0


def admin_delete_review(rid: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM review WHERE review_id=%s", (rid,))
    conn.commit()
    affected = cur.rowcount
    cur.close()
    conn.close()
    return affected > 0
