from db import get_conn
import pymysql

def get_all_status():
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT * FROM status")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def get_status_by_id(sid: int):
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT * FROM status WHERE status_id=%s", (sid,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row

def insert_status(name: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO status (status_name) VALUES (%s)",
        (name,)
    )
    conn.commit()
    new_id = cur.lastrowid
    cur.close()
    conn.close()
    return new_id

def update_status(sid: int, data: dict):
    if "status_name" not in data:
        return False

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE status SET status_name=%s WHERE status_id=%s",
        (data["status_name"], sid)
    )
    conn.commit()
    affected = cur.rowcount
    cur.close()
    conn.close()
    return affected > 0

def delete_status(sid: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM status WHERE status_id=%s", (sid,))
    conn.commit()
    affected = cur.rowcount
    cur.close()
    conn.close()
    return affected > 0
