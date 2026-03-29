import pymysql
from db import get_conn

def get_all_hairstyle():
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(
        """
        SELECT *
        FROM hairstyle
        WHERE deleted_at IS NULL
        ORDER BY hairstyle_id ASC
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_hairstyle_by_id(pid: int):
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(
        """
        SELECT *
        FROM hairstyle
        WHERE hairstyle_id = %s
          AND deleted_at IS NULL
        """,
        (pid,)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def insert_hairstyle(data: dict):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO hairstyle (hairstyle_name, category, face_shape, image_path)
        VALUES (%s, %s, %s, %s)
        """,
        (
            data["hairstyle_name"],
            data.get("category"),
            data.get("face_shape"),
            data.get("image_path"),
        )
    )
    conn.commit()
    new_id = cur.lastrowid
    cur.close()
    conn.close()
    return new_id


def update_hairstyle(pid: int, data: dict):
    fields = []
    values = []

    if "hairstyle_name" in data:
        fields.append("hairstyle_name = %s")
        values.append(data["hairstyle_name"])
    if "category" in data:
        fields.append("category = %s")
        values.append(data["category"])
    if "face_shape" in data:
        fields.append("face_shape = %s")
        values.append(data["face_shape"])
    if "image_path" in data:
        fields.append("image_path = %s")
        values.append(data["image_path"])

    if not fields:
        return False

    values.append(pid)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        f"UPDATE hairstyle SET {', '.join(fields)} WHERE hairstyle_id = %s",
        values
    )
    conn.commit()
    updated = cur.rowcount > 0
    cur.close()
    conn.close()
    return updated


def delete_hairstyle(pid: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE hairstyle SET deleted_at = NOW() WHERE hairstyle_id = %s AND deleted_at IS NULL",
        (pid,)
    )
    conn.commit()
    deleted = cur.rowcount > 0
    cur.close()
    conn.close()
    return deleted
