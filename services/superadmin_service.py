import pymysql
from db import get_conn

def get_all_superadmin():
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT * FROM superadmin")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def get_superadmin_by_id(sid):
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT * FROM superadmin WHERE superadmin_id=%s", (sid,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row

def insert_superadmin(data):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO superadmin (username, password) VALUES (%s, %s)",
        (data["username"], data["password"])
    )
    conn.commit()
    new_id = cur.lastrowid
    cur.close()
    conn.close()
    return new_id

def update_superadmin(sid, data):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE superadmin SET username=%s, password=%s WHERE superadmin_id=%s",
        (data["username"], data["password"], sid)
    )
    conn.commit()
    updated = cur.rowcount
    cur.close()
    conn.close()
    return updated > 0

def delete_superadmin(sid):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM superadmin WHERE superadmin_id=%s", (sid,))
    conn.commit()
    deleted = cur.rowcount
    cur.close()
    conn.close()
    return deleted > 0
