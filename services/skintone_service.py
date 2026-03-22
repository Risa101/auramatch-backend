from db import get_conn
import pymysql

def get_all_skintone():
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT * FROM skintone")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def get_skintone_by_id(sid: int):
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT * FROM skintone WHERE skintone_id=%s", (sid,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row

def insert_skintone(name: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO skintone (skintone_name) VALUES (%s)",
        (name,)
    )
    conn.commit()
    new_id = cur.lastrowid
    cur.close()
    conn.close()
    return new_id

def update_skintone(sid: int, data: dict):
    if "skintone_name" not in data:
        return False

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE skintone SET skintone_name=%s WHERE skintone_id=%s",
        (data["skintone_name"], sid)
    )
    conn.commit()
    affected = cur.rowcount
    cur.close()
    conn.close()
    return affected > 0

def delete_skintone(sid: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM skintone WHERE skintone_id=%s", (sid,))
    conn.commit()
    affected = cur.rowcount
    cur.close()
    conn.close()
    return affected > 0
