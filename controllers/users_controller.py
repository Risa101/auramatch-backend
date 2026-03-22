"""
Service สำหรับงานที่เกี่ยวกับผู้ใช้งาน (users)

หน้าที่ของ service คือเป็นชั้นกลางที่ติดต่อกับฐานข้อมูล
และแยก logic การเข้าถึงข้อมูลออกจาก controller
คอมเมนต์ทั้งหมดเป็นภาษาไทยตามคำขอ
"""

from db import get_conn


def add_user(name: str, email: str) -> None:
    """เพิ่มผู้ใช้ใหม่ลงตาราง users

    ฟังก์ชันนี้จะเปิดการเชื่อมต่อแบบชั่วคราว แล้วปิดเมื่อทำงานเสร็จ
    """
    conn = get_conn()
    cursor = conn.cursor()
    try:
        sql = "INSERT INTO users(name, email) VALUES (%s, %s)"
        cursor.execute(sql, (name, email))
    finally:
        # ปิด cursor และ connection เสมอเพื่อป้องกัน resource leak
        cursor.close()
        conn.close()


def get_user_by_id(uid: int):
    """ดึงข้อมูลผู้ใช้ตาม id ถ้าไม่มีคืน None"""
    conn = get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM users WHERE id=%s", (uid,))
        return cursor.fetchone()
    finally:
        cursor.close()
        conn.close()