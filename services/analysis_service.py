from db import get_conn
from datetime import datetime
import pymysql

def save_analysis(data: dict):
    try:
        conn = get_conn()
    except Exception as e:
        return {"success": False, "message": f"DB connection failed: {str(e)}"}

    cur = conn.cursor()
    try:
        sql = """
            INSERT INTO analysis_history
            (user_id, season, face_shape, eyebrows, eyes, nose, lips, image_path, score)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            data.get('user_id'),
            data.get('season'),
            data.get('face_shape'),
            data.get('eyebrows'),
            data.get('eyes'),
            data.get('nose'),
            data.get('lips'),
            data.get('image_path'),
            data.get('score', 100)
        )
        cur.execute(sql, values)
        conn.commit()
        return {"success": True, "id": cur.lastrowid, "message": "Analysis saved successfully"}
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        return {"success": False, "message": str(e)}
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass

def get_history_by_user(user_id: int):
    conn = get_conn()
    # ใช้ DictCursor เพื่อให้ผลลัพธ์เป็น dict
    cur = conn.cursor(pymysql.cursors.DictCursor)
    try:
        sql = "SELECT *, history_id AS analysis_id FROM analysis_history WHERE user_id = %s ORDER BY analysis_date DESC"
        cur.execute(sql, (user_id,))
        rows = cur.fetchall()
        
        # ปรับแก้: แปลง datetime เป็น string เพื่อให้ส่ง JSON ได้
        for row in rows:
            if isinstance(row.get('analysis_date'), datetime):
                row['analysis_date'] = row['analysis_date'].strftime('%Y-%m-%d %H:%M:%S')
                
        return rows
    except Exception as e:
        print(f"Error fetching history: {e}")
        return []
    finally:
        cur.close()
        conn.close()
