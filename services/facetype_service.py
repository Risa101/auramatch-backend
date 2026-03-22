from db import get_conn
import pymysql

def get_all_facetype():
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT * FROM facetype")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_facetype_by_id(fid: int):
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT * FROM facetype WHERE facetype_id = %s", (fid,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def insert_facetype(face_id: int, facetype_name: str):
    conn = get_conn()
    cur = conn.cursor()
    sql = """
        INSERT INTO facetype (face_id, facetype_name)
        VALUES (%s, %s)
    """
    cur.execute(sql, (face_id, facetype_name))
    conn.commit()
    new_id = cur.lastrowid
    cur.close()
    conn.close()
    return new_id


def update_facetype(fid: int, data: dict):
    if not data:
        return False

    fields = []
    values = []

    for key, value in data.items():
        fields.append(f"{key} = %s")
        values.append(value)

    values.append(fid)

    sql = f"""
        UPDATE facetype
        SET {', '.join(fields)}
        WHERE facetype_id = %s
    """

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(sql, values)
    conn.commit()
    updated = cur.rowcount
    cur.close()
    conn.close()
    return updated > 0


def delete_facetype(fid: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM facetype WHERE facetype_id = %s", (fid,))
    conn.commit()
    deleted = cur.rowcount
    cur.close()
    conn.close()
    return deleted > 0
