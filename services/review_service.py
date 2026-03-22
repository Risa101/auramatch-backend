from db import get_conn
import pymysql

def get_all_reviews():
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT * FROM review")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_review_by_id(rid: int):
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT * FROM review WHERE review_id=%s", (rid,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def insert_review(user_id, product_id, rating, comment=None):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO review (user_id, product_id, rating, comment)
        VALUES (%s, %s, %s, %s)
        """,
        (user_id, product_id, rating, comment)
    )
    conn.commit()
    new_id = cur.lastrowid
    cur.close()
    conn.close()
    return new_id


def update_review(rid: int, data: dict):
    fields = []
    values = []

    for k, v in data.items():
        fields.append(f"{k}=%s")
        values.append(v)

    if not fields:
        return False

    values.append(rid)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        f"UPDATE review SET {', '.join(fields)} WHERE review_id=%s",
        values
    )
    conn.commit()
    affected = cur.rowcount
    cur.close()
    conn.close()
    return affected > 0


def delete_review(rid: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM review WHERE review_id=%s", (rid,))
    conn.commit()
    affected = cur.rowcount
    cur.close()
    conn.close()
    return affected > 0
