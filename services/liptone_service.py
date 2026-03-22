from db import get_conn

def get_all_liptone():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM liptone")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def get_liptone_by_id(liptone_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM liptone WHERE liptone_id=%s", (liptone_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row

def insert_liptone(liptone_name):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO liptone (liptone_name) VALUES (%s)",
        (liptone_name,)
    )
    new_id = cur.lastrowid
    cur.close()
    conn.close()
    return new_id

def update_liptone(liptone_id, data):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE liptone SET liptone_name=%s WHERE liptone_id=%s",
        (data.get("liptone_name"), liptone_id)
    )
    updated = cur.rowcount
    cur.close()
    conn.close()
    return updated > 0

def delete_liptone(liptone_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM liptone WHERE liptone_id=%s",
        (liptone_id,)
    )
    deleted = cur.rowcount
    cur.close()
    conn.close()
    return deleted > 0
