from db import get_conn
import pymysql


def get_all_productcolor():
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT * FROM productColor")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_productcolor_by_id(pid: int):
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(
        "SELECT * FROM productColor WHERE productColor_id = %s",
        (pid,)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def insert_productcolor(product_id, color_name):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO productColor (product_id, color_name) VALUES (%s, %s)",
        (product_id, color_name)
    )
    conn.commit()
    new_id = cur.lastrowid
    cur.close()
    conn.close()
    return new_id


def update_productcolor(pid: int, data: dict):
    if not data:
        return False

    fields = []
    values = []

    for key, value in data.items():
        fields.append(f"{key}=%s")
        values.append(value)

    values.append(pid)

    sql = f"""
        UPDATE productColor
        SET {', '.join(fields)}
        WHERE productColor_id = %s
    """

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(sql, values)
    conn.commit()
    affected = cur.rowcount
    cur.close()
    conn.close()
    return affected > 0


def delete_productcolor(pid: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM productColor WHERE productColor_id = %s",
        (pid,)
    )
    conn.commit()
    affected = cur.rowcount
    cur.close()
    conn.close()
    return affected > 0
