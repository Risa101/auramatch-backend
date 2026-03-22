import pymysql
from db import get_conn


def get_all_haircolor():
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT * FROM haircolor")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_haircolor_by_id(haircolor_id: int):
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(
        "SELECT * FROM haircolor WHERE haircolor_id = %s",
        (haircolor_id,)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def insert_haircolor(haircolor_name: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO haircolor (haircolor_name) VALUES (%s)",
        (haircolor_name,)
    )
    conn.commit()
    new_id = cur.lastrowid
    cur.close()
    conn.close()
    return new_id


def update_haircolor(haircolor_id: int, data: dict):
    if "haircolor_name" not in data:
        return False

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE haircolor SET haircolor_name = %s WHERE haircolor_id = %s",
        (data["haircolor_name"], haircolor_id)
    )
    conn.commit()
    updated = cur.rowcount > 0
    cur.close()
    conn.close()
    return updated


def delete_haircolor(haircolor_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM haircolor WHERE haircolor_id = %s",
        (haircolor_id,)
    )
    conn.commit()
    deleted = cur.rowcount > 0
    cur.close()
    conn.close()
    return deleted
