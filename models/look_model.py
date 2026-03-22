from db import get_conn
import pymysql.cursors

def get_looks_from_db(color):
    conn = get_conn()
    if not conn: return []
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            # ใช้พารามิเตอร์ %s เพื่อดึงข้อมูลตามกลุ่มสี (Spring, Summer, etc.)
            sql = "SELECT * FROM looks WHERE personal_color = %s AND status = 'active'"
            cur.execute(sql, (color,))
            return cur.fetchall()
    finally:
        conn.close()