import pymysql
from db import get_conn

def get_all_hairstyle():
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT * FROM hairstyle")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_hairstyle_by_id(pid: int):
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(
        "SELECT * FROM hairstyle WHERE hairstyle_id = %s",
        (pid,)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def insert_hairstyle(face_id: int, hairstyle_name: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO hairstyle (face_id, hairstyle_name) VALUES (%s, %s)",
        (face_id, hairstyle_name)
    )
    conn.commit()
    new_id = cur.lastrowid
    cur.close()
    conn.close()
    return new_id


def update_hairstyle(pid: int, data: dict):
    fields = []
    values = []

    if "face_id" in data:
        fields.append("face_id = %s")
        values.append(data["face_id"])

    if "hairstyle_name" in data:
        fields.append("hairstyle_name = %s")
        values.append(data["hairstyle_name"])

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
        "DELETE FROM hairstyle WHERE hairstyle_id = %s",
        (pid,)
    )
    conn.commit()
    deleted = cur.rowcount > 0
    cur.close()
    conn.close()
    return deleted
