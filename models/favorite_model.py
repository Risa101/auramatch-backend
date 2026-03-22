from db import get_conn
import pymysql.cursors

def get_favorite_by_user_db(user_id: int):
    conn = get_conn()
    try:
        cur = conn.cursor(pymysql.cursors.DictCursor)
        cur.execute("""
            SELECT f.favorite_id, f.product_id, p.name, p.price, p.image_url
            FROM favorite f
            JOIN products p ON f.product_id = p.product_id
            WHERE f.user_id = %s AND f.deleted_at IS NULL
              AND p.deleted_at IS NULL AND p.status = 'active'
        """, (user_id,))
        return cur.fetchall()
    finally:
        cur.close()
        conn.close()

def toggle_favorite_db(user_id: int, product_id: str):
    conn = get_conn()
    try:
        cur = conn.cursor(pymysql.cursors.DictCursor)
        # ตรวจสอบว่าเคยมีข้อมูลหรือไม่ (ป้องกัน Duplicate Entry)
        cur.execute("SELECT favorite_id, deleted_at FROM favorite WHERE user_id = %s AND product_id = %s LIMIT 1", (user_id, product_id))
        row = cur.fetchone()

        if row:
            if row["deleted_at"] is None:
                cur.execute("UPDATE favorite SET deleted_at = NOW() WHERE favorite_id = %s", (row["favorite_id"],))
                action = "removed"
            else:
                cur.execute("UPDATE favorite SET deleted_at = NULL WHERE favorite_id = %s", (row["favorite_id"],))
                action = "added"
        else:
            cur.execute("INSERT INTO favorite (user_id, product_id) VALUES (%s, %s)", (user_id, product_id))
            action = "added"
        conn.commit()
        return action
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()