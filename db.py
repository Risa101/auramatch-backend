"""
ส่วนจัดการการเชื่อมต่อฐานข้อมูล (DB)

ฟังก์ชันที่นี่จะคืน connection ใหม่ทุกครั้งที่เรียก เพื่อให้แต่ละคำขอสามารถแยกการเชื่อมต่อได้
ค่า config จะอ่านจาก environment variables (เช่น .env)
คอมเมนต์ทั้งหมดเป็นภาษาไทยตามคำขอ
"""

import os
from urllib.parse import unquote, urlparse

import pymysql
from pymysql.cursors import DictCursor


def _pick_env(*keys, default=None):
    """คืนค่าตัวแปร env ตัวแรกที่มีค่า"""
    for key in keys:
        value = os.getenv(key)
        if value not in (None, ""):
            return value
    return default


def _parse_database_url(raw_url):
    """รองรับ URL รูปแบบ mysql://user:pass@host:port/dbname"""
    if not raw_url:
        return {}

    parsed = urlparse(raw_url)
    if parsed.scheme not in {"mysql", "mysql+pymysql"}:
        return {}

    return {
        "host": parsed.hostname,
        "port": parsed.port or 3306,
        "user": unquote(parsed.username) if parsed.username else None,
        "password": unquote(parsed.password) if parsed.password else None,
        "database": parsed.path.lstrip("/") or None,
    }


def _get_db_config():
    """เตรียม dictionary ของการตั้งค่า DB โดยอ่านจาก environment variables

    ค่าพื้นฐานจะถูกใช้เป็นค่าเริ่มต้นถ้าไม่ได้กำหนดใน env
    """
    parsed_url = _parse_database_url(
        _pick_env("DATABASE_URL", "MYSQL_URL", "MYSQL_PUBLIC_URL")
    )

    config = {
        "host": _pick_env(
            "DB_HOST",
            "MYSQLHOST",
            default=parsed_url.get("host", "127.0.0.1"),
        ),
        "port": int(
            _pick_env(
                "DB_PORT",
                "MYSQLPORT",
                default=str(parsed_url.get("port", 3306)),
            )
        ),
        "user": _pick_env("DB_USER", "MYSQLUSER", default=parsed_url.get("user")),
        "password": _pick_env(
            "DB_PASS",
            "MYSQLPASSWORD",
            default=parsed_url.get("password"),
        ),
        "database": _pick_env(
            "DB_NAME",
            "MYSQLDATABASE",
            default=parsed_url.get("database"),
        ),
        "cursorclass": DictCursor,
        "charset": "utf8mb4",
        "autocommit": True,
    }

    missing = [key for key in ("user", "database") if not config.get(key)]
    if missing:
        raise RuntimeError(
            "Database configuration incomplete. "
            "Set DB_HOST/DB_PORT/DB_USER/DB_PASS/DB_NAME "
            "or Railway MYSQLHOST/MYSQLPORT/MYSQLUSER/MYSQLPASSWORD/MYSQLDATABASE."
        )

    return config


def get_conn():
    """สร้างและคืนค่าการเชื่อมต่อฐานข้อมูลใหม่

    ผู้เรียกต้องรับผิดชอบในการปิด cursor และ connection เมื่อเสร็จ
    """
    return pymysql.connect(**_get_db_config())
