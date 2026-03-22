"""
ส่วนจัดการการเชื่อมต่อฐานข้อมูล (DB)

ฟังก์ชันที่นี่จะคืน connection ใหม่ทุกครั้งที่เรียก เพื่อให้แต่ละคำขอสามารถแยกการเชื่อมต่อได้
ค่า config จะอ่านจาก environment variables (เช่น .env)
คอมเมนต์ทั้งหมดเป็นภาษาไทยตามคำขอ
"""

import os
import pymysql
from pymysql.cursors import DictCursor


def _get_db_config():
    """เตรียม dictionary ของการตั้งค่า DB โดยอ่านจาก environment variables

    ค่าพื้นฐานจะถูกใช้เป็นค่าเริ่มต้นถ้าไม่ได้กำหนดใน env
    """
    return {
        "host": os.getenv("DB_HOST", "127.0.0.1"),
        "port": int(os.getenv("DB_PORT", "3306")),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASS"),
        "database": os.getenv("DB_NAME"),
        "cursorclass": DictCursor,
        "charset": "utf8mb4",
        "autocommit": True,
    }


def get_conn():
    """สร้างและคืนค่าการเชื่อมต่อฐานข้อมูลใหม่

    ผู้เรียกต้องรับผิดชอบในการปิด cursor และ connection เมื่อเสร็จ
    """
    return pymysql.connect(**_get_db_config())