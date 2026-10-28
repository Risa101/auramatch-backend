"""
บริการจัดการช่องทางการซื้อสินค้าหลายแพลตฟอร์ม (Shopee / Lazada / TikTok Shop ฯลฯ)

หนึ่งสินค้า (products.product_id) สามารถมีลิงก์ได้หลายแพลตฟอร์ม เก็บไว้ใน
ตาราง `product_platforms` (ดู migrations/2026-08-30_add_product_platforms.sql)
คอลัมน์ `platform` เป็น VARCHAR ไม่ใช่ ENUM เพื่อให้เพิ่มแพลตฟอร์มใหม่ในอนาคตได้
โดยไม่ต้องแก้ schema — ฝั่งนี้จึงตรวจสอบแบบ "รูปแบบข้อมูลถูกต้องหรือไม่"
(ไม่ว่าง, ตัวพิมพ์เล็ก, ความยาวพอดี) แทนการล็อกรายชื่อแพลตฟอร์มตายตัว

ไฟล์นี้ไม่แตะ services/color_engine/ หรือ services/personal_color_service.py —
เป็นเพียง data access layer ของสินค้า เหมือน product_service.py เดิม
"""

import re

from db import get_conn
import pymysql.cursors

# รายการแพลตฟอร์มที่ระบบรู้จักในตอนนี้ (ใช้เพื่อ validate เบาๆ ไม่ใช่ hard block —
# แพลตฟอร์มใหม่ที่ยังไม่อยู่ในนี้ยังคง insert ได้ ค่อยเพิ่มชื่อเข้ามาทีหลังได้เสมอ)
KNOWN_PLATFORMS = {"shopee", "lazada", "tiktok_shop"}

_PLATFORM_KEY_RE = re.compile(r"^[a-z0-9_]{2,30}$")


def _normalize_platform_key(raw):
    key = str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    return key


def is_valid_platform_key(raw):
    key = _normalize_platform_key(raw)
    return bool(_PLATFORM_KEY_RE.match(key))


def is_valid_url(raw):
    """ตรวจสอบคร่าวๆ ว่าเป็น http(s) URL จริง ก่อนจะเก็บลง DB —
    ป้องกัน javascript:/data: หรือค่าว่างที่ปนเข้ามาจาก payload ของ admin"""
    url = str(raw or "").strip()
    if not url:
        return False
    return bool(re.match(r"^https?://[^\s]+$", url, re.IGNORECASE))


def get_platforms_for_product(product_id, include_inactive=False):
    conn = get_conn()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            sql = (
                "SELECT id, product_id, platform, product_url, affiliate_url, "
                "is_active, created_at, updated_at "
                "FROM product_platforms WHERE product_id = %s"
            )
            params = [product_id]
            if not include_inactive:
                sql += " AND is_active = 1"
            sql += " ORDER BY platform ASC"
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        conn.close()


def get_platforms_for_products(product_ids, include_inactive=False):
    """ดึงลิงก์แพลตฟอร์มของสินค้าหลายชิ้นในครั้งเดียว (กัน N+1 query เวลาแสดงผลรายการสินค้า)

    คืนค่าเป็น dict: { product_id: [platform_row, ...] }
    """
    product_ids = [pid for pid in dict.fromkeys(product_ids) if pid]
    if not product_ids:
        return {}

    conn = get_conn()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            placeholders = ", ".join(["%s"] * len(product_ids))
            sql = (
                "SELECT id, product_id, platform, product_url, affiliate_url, "
                "is_active, created_at, updated_at "
                f"FROM product_platforms WHERE product_id IN ({placeholders})"
            )
            params = list(product_ids)
            if not include_inactive:
                sql += " AND is_active = 1"
            sql += " ORDER BY platform ASC"
            cur.execute(sql, params)
            rows = cur.fetchall()
    finally:
        conn.close()

    grouped = {}
    for row in rows:
        grouped.setdefault(row["product_id"], []).append(row)
    return grouped


def replace_platforms_for_product(product_id, platforms):
    """อัปเดต/เพิ่มแพลตฟอร์มที่ส่งมาแบบ bulk upsert -- ไม่ใช่การแทนที่ทั้งชุด

    `platforms` คือ list ของ dict: {platform, product_url, affiliate_url?, is_active?}
    ใช้ ON DUPLICATE KEY UPDATE บน (product_id, platform) — ไม่ลบแถวของ platform อื่น
    ที่ไม่ได้ส่งมาในรอบนี้ (เผื่อ admin ส่งมาเฉพาะบางแพลตฟอร์มที่แก้ไข) ชื่อฟังก์ชัน
    ยังคงไว้เพื่อ backward-compat กับโค้ด/เทสต์ที่เรียกอยู่ — ถ้าต้องการลบแพลตฟอร์ม
    ใดออกจริงๆ ให้ใช้ delete_platform() ด้านล่างแทน

    คืนค่า (ok: bool, error: str|None)
    """
    if not product_id:
        return False, "product_id required"
    if not isinstance(platforms, list) or not platforms:
        return False, "platforms must be a non-empty list"

    rows_to_upsert = []
    for entry in platforms:
        platform_key = _normalize_platform_key(entry.get("platform"))
        url = str(entry.get("product_url") or "").strip()
        if not is_valid_platform_key(platform_key):
            return False, f"invalid platform key: {entry.get('platform')!r}"
        if not is_valid_url(url):
            return False, f"invalid product_url for platform {platform_key!r}"
        affiliate_url = entry.get("affiliate_url")
        affiliate_url = str(affiliate_url).strip() if affiliate_url else None
        if affiliate_url and not is_valid_url(affiliate_url):
            return False, f"invalid affiliate_url for platform {platform_key!r}"
        is_active = 1 if entry.get("is_active", True) else 0
        rows_to_upsert.append((product_id, platform_key, url, affiliate_url, is_active))

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM products WHERE product_id=%s AND deleted_at IS NULL",
                (product_id,),
            )
            if not cur.fetchone():
                return False, "product not found"

            sql = (
                "INSERT INTO product_platforms "
                "(product_id, platform, product_url, affiliate_url, is_active) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE "
                "product_url=VALUES(product_url), "
                "affiliate_url=VALUES(affiliate_url), "
                "is_active=VALUES(is_active)"
            )
            cur.executemany(sql, rows_to_upsert)
        conn.commit()
        return True, None
    finally:
        conn.close()


def delete_platform(product_id, platform):
    platform_key = _normalize_platform_key(platform)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM product_platforms WHERE product_id=%s AND platform=%s",
                (product_id, platform_key),
            )
            affected = cur.rowcount
        conn.commit()
        return affected > 0
    finally:
        conn.close()


def serialize_platform_row(row):
    return {
        "id": row["id"],
        "platform": row["platform"],
        "product_url": row["product_url"],
        "affiliate_url": row.get("affiliate_url"),
        "is_active": bool(row["is_active"]),
    }
