import pymysql
from db import get_conn
from werkzeug.security import check_password_hash, generate_password_hash

def get_all_user():
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(
        """
        SELECT user_id, username, email, created_at, avatar, deleted_at, role
        FROM user
        WHERE deleted_at IS NULL
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def get_user_by_id(uid):
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(
        """
        SELECT user_id, username, email, created_at, avatar, deleted_at, role
        FROM user
        WHERE user_id=%s
        """,
        (uid,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row

def insert_user(data):
    conn = get_conn()
    cur = conn.cursor()
    hashed_password = generate_password_hash(data["password"])
    columns = ["email", "password"]
    values = [data["email"], hashed_password]
    if data.get("username"):
        columns.append("username")
        values.append(data["username"])
    if data.get("avatar"):
        columns.append("avatar")
        values.append(data["avatar"])
    if data.get("role"):
        columns.append("role")
        values.append(data["role"])

    placeholders = ", ".join(["%s"] * len(values))
    sql = f"INSERT INTO user ({', '.join(columns)}) VALUES ({placeholders})"
    cur.execute(sql, values)
    conn.commit()
    new_id = cur.lastrowid
    cur.close()
    conn.close()
    return new_id

def update_user(uid, data):
    conn = get_conn()
    cur = conn.cursor()
    fields = []
    values = []
    if "username" in data:
        fields.append("username=%s")
        values.append(data["username"])
    if "email" in data:
        fields.append("email=%s")
        values.append(data["email"])
    if "avatar" in data:
        fields.append("avatar=%s")
        values.append(data["avatar"])
    if "role" in data:
        fields.append("role=%s")
        values.append(data["role"])
    if "password" in data and data["password"]:
        fields.append("password=%s")
        values.append(generate_password_hash(data["password"]))

    if not fields:
        cur.close()
        conn.close()
        return False

    values.append(uid)
    sql = f"UPDATE user SET {', '.join(fields)} WHERE user_id=%s"
    cur.execute(sql, values)
    conn.commit()
    updated = cur.rowcount
    cur.close()
    conn.close()
    return updated > 0

def delete_user(uid):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE user SET deleted_at=NOW() WHERE user_id=%s", (uid,))
    conn.commit()
    deleted = cur.rowcount
    cur.close()
    conn.close()
    return deleted > 0

def set_user_password(user_id, password):
    conn = get_conn()
    cur = conn.cursor()
    hashed_password = generate_password_hash(password)
    cur.execute(
        "UPDATE user SET password=%s WHERE user_id=%s",
        (hashed_password, user_id),
    )
    conn.commit()
    updated = cur.rowcount
    cur.close()
    conn.close()
    return updated > 0

def get_user_by_email(email):
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(
        "SELECT * FROM user WHERE email=%s AND deleted_at IS NULL LIMIT 1",
        (email,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row

def _is_password_hash(value):
    return isinstance(value, str) and value.startswith(("pbkdf2:", "scrypt:", "argon2:"))

def authenticate_user(identifier, password):
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(
        """
        SELECT user_id, username, email, password, created_at, avatar, role
        FROM user
        WHERE deleted_at IS NULL AND (email=%s OR username=%s)
        LIMIT 1
        """,
        (identifier, identifier),
    )
    row = cur.fetchone()
    if not row:
        cur.close()
        conn.close()
        return None

    stored_password = row.get("password") or ""
    if _is_password_hash(stored_password):
        if not check_password_hash(stored_password, password):
            cur.close()
            conn.close()
            return None
    else:
        if stored_password != password:
            cur.close()
            conn.close()
            return None
        # upgrade legacy plain text password to a hash
        new_hash = generate_password_hash(password)
        cur.execute(
            "UPDATE user SET password=%s WHERE user_id=%s",
            (new_hash, row["user_id"]),
        )
        conn.commit()

    cur.close()
    conn.close()
    row.pop("password", None)
    return row
