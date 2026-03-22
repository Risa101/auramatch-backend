import pymysql
from db import get_conn

def get_all_faceproduct():
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT * FROM face_product")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_faceproduct_by_id(pid: int):
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(
        "SELECT * FROM face_product WHERE face_product_id = %s",
        (pid,)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def insert_faceproduct(face_id: int, product_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO face_product (face_id, product_id)
        VALUES (%s, %s)
        """,
        (face_id, product_id)
    )
    conn.commit()
    new_id = cur.lastrowid
    cur.close()
    conn.close()
    return new_id


def update_faceproduct(pid: int, data: dict):
    if not data:
        return False

    fields = []
    values = []

    for key, value in data.items():
        fields.append(f"{key} = %s")
        values.append(value)

    values.append(pid)

    sql = f"""
        UPDATE face_product
        SET {", ".join(fields)}
        WHERE face_product_id = %s
    """

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(sql, values)
    conn.commit()
    affected = cur.rowcount
    cur.close()
    conn.close()
    return affected > 0


def delete_faceproduct(pid: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM face_product WHERE face_product_id = %s",
        (pid,)
    )
    conn.commit()
    affected = cur.rowcount
    cur.close()
    conn.close()
    return affected > 0
