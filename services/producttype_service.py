from db import get_conn
import pymysql

def get_all_producttype():
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT * FROM productType")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def get_producttype_by_id(pid: int):
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(
        "SELECT * FROM productType WHERE productType_id = %s",
        (pid,)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row

def insert_producttype(type_name):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO productType (type_name) VALUES (%s)",
        (type_name,)
    )
    conn.commit()
    new_id = cur.lastrowid
    cur.close()
    conn.close()
    return new_id

def update_producttype(pid: int, data: dict):
    if not data:
        return False

    fields = []
    values = []

    for k, v in data.items():
        fields.append(f"{k}=%s")
        values.append(v)

    values.append(pid)

    sql = f"""
        UPDATE productType
        SET {', '.join(fields)}
        WHERE productType_id = %s
    """

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(sql, values)
    conn.commit()
    affected = cur.rowcount
    cur.close()
    conn.close()
    return affected > 0

def delete_producttype(pid: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM productType WHERE productType_id = %s",
        (pid,)
    )
    conn.commit()
    affected = cur.rowcount
    cur.close()
    conn.close()
    return affected > 0
