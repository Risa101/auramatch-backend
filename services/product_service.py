from db import get_conn
import pymysql.cursors
from services.product_platform_service import (
    get_platforms_for_products,
    serialize_platform_row,
)


def _attach_platforms(rows):
    """เติมฟิลด์ `platforms` (list ของ {platform, product_url, ...}) ให้แต่ละแถวสินค้า

    Additive เท่านั้น — ไม่แก้ไขฟิลด์เดิมของ products/`shop_url` เลย เพื่อไม่ให้
    หน้าเว็บที่ยังใช้ p.shop_url แบบเดิม (ลิงก์เดียว) พังไป ส่วนหน้าที่รองรับ
    หลายแพลตฟอร์มแล้วจะอ่านจาก p.platforms แทน
    """
    if not rows:
        return rows
    product_ids = [r.get("product_id") for r in rows if r.get("product_id")]
    grouped = get_platforms_for_products(product_ids)
    for row in rows:
        pid = row.get("product_id")
        row["platforms"] = [serialize_platform_row(p) for p in grouped.get(pid, [])]
    return rows


# ===============================
# GET ALL PRODUCTS
# ===============================
def get_all_products():
    conn = get_conn()
    # ใช้ DictCursor เพื่อให้เรียกข้อมูลผ่านชื่อคอลัมน์ r['name'] ได้เลย
    cur = conn.cursor(pymysql.cursors.DictCursor)

    sql = """
        SELECT *
        FROM products
        WHERE deleted_at IS NULL
          AND status = 'active'
        ORDER BY created_at DESC
    """
    cur.execute(sql)
    rows = cur.fetchall()

    cur.close()
    conn.close()
    return _attach_platforms(rows)

# ===============================
# GET ALL PRODUCTS (ADMIN)
# ===============================
def get_all_products_admin():
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)

    sql = """
        SELECT *
        FROM products
        WHERE deleted_at IS NULL
        ORDER BY created_at DESC
    """
    cur.execute(sql)
    rows = cur.fetchall()

    cur.close()
    conn.close()
    return _attach_platforms(rows)


# ===============================
# GET PRODUCT BY ID
# ===============================
def get_product_by_id(product_id):
    conn = get_conn()
    try:
        # ใช้ DictCursor เพื่อให้เข้าถึงข้อมูลด้วยชื่อคอลัมน์ได้ เช่น row['name']
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            sql = """
                SELECT *
                FROM products
                WHERE product_id = %s
                  AND deleted_at IS NULL
            """
            cur.execute(sql, (product_id,))
            row = cur.fetchone()
            
            # หากไม่พบข้อมูลให้คืนค่า None
            if not row:
                return None
                
            # แปลงค่า Decimal (เช่น ราคา) เป็น float เพื่อให้ส่งออก JSON ได้ง่าย
            if 'price' in row and row['price'] is not None:
                row['price'] = float(row['price'])

            _attach_platforms([row])
            return row
    finally:
        # มั่นใจว่าการเชื่อมต่อจะถูกปิดเสมอแม้เกิด Error
        conn.close()


# ===============================
# 🔥 BEST SELLER PRODUCTS
# ===============================
def get_best_seller_products():
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)

    sql = """
        SELECT
            p.product_id,
            p.name,
            p.image_url,
            p.price,
            COALESCE(SUM(oi.qty), 0) AS total_sold
        FROM products p
        LEFT JOIN order_items oi
            ON p.product_id = oi.product_id
        WHERE p.deleted_at IS NULL 
          AND p.status = 'active'
        GROUP BY
            p.product_id,
            p.name,
            p.image_url,
            p.price
        HAVING total_sold > 0
        ORDER BY total_sold DESC
        LIMIT 6
    """
    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return _attach_platforms([
        {
            "product_id": r["product_id"],
            "name": r["name"],
            "image_url": r["image_url"],
            "price": float(r["price"]),
            "total_sold": int(r["total_sold"]),
        }
        for r in rows
    ])


# ==========================================
# ✨ RECOMMENDED PRODUCTS (แก้ไขให้ตรงตามตารางจริง)
# ==========================================
def get_recommended_products(season, limit=3, category=None):
    """
    ดึงสินค้าที่ตรงกับ personal_color_tags เช่น 'Spring'
    อ้างอิงข้อมูลจริงจาก DB ที่ใช้รูปแบบ 'Warm,Spring'

    category: กรองตามหมวด (เช่น 'lip') ถ้าไม่ส่งมาจะดึงทุกหมวดเหมือนเดิม —
    เพิ่มเพื่อให้หน้าเว็บใช้ endpoint เดียวกันดึงสินค้าจริงจาก DB แยกตามหมวดได้
    (เช่น ลิปที่มีลิงก์ร้านค้าจริง) แทนของ hardcode
    """
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)

    # แก้ไข seasonTags -> personal_color_tags
    # แก้ไข RANDOM() -> RAND() สำหรับ MySQL
    sql = """
        SELECT
            product_id,
            name,
            image_url,
            price,
            category,
            suitable_for_color,
            personal_color_tags,
            shades,
            links
        FROM products
        WHERE (personal_color_tags LIKE %s OR personal_color_tags LIKE '%%All%%')
          AND status = 'active'
          AND deleted_at IS NULL
    """
    params = [f"%{season}%"]
    if category:
        sql += " AND category = %s"
        params.append(category)
    sql += " ORDER BY RAND() LIMIT %s"
    params.append(limit)

    cur.execute(sql, tuple(params))
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return _attach_platforms([
        {
            "product_id": r["product_id"],
            "name": r["name"],
            "image_url": r["image_url"],
            "price": float(r["price"]) if r["price"] is not None else None,
            "category": r["category"],
            "suitable_for_color": r["suitable_for_color"],
            "personal_color_tags": r["personal_color_tags"],
            "shades": r["shades"],
            "shop_url": r["links"],
        }
        for r in rows
    ])

# ==========================================
# 💄 BLUSH PRODUCTS BY UNDERTONE (ใช้โดย undertone_service.py — แทนที่ hardcode palette)
# ==========================================
def get_blush_products_by_undertone(undertone, limit=4):
    """
    ดึงสินค้าบลัชออนจริงจาก DB ที่เหมาะกับ undertone ("cool" | "warm" | "neutral")
    อ้างอิงคอลัมน์ suitable_for_color (ค่าจริงใน DB ไม่สม่ำเสมอเรื่องตัวพิมพ์ เช่น
    "Cool" กับ "warm" ปนกัน จึงเทียบแบบ case-insensitive)

    ถ้า undertone เป็น "unknown" (หรือไม่ตรงกับ cool/warm/neutral) จะดึงบลัชออนทั่วไป
    แบบสุ่มแทน ไม่ใช้ค่าตายตัวใดๆ ทั้งสิ้น
    """
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)

    tone = (undertone or "").strip().lower()
    if tone in ("cool", "warm", "neutral"):
        sql = """
            SELECT product_id, name, image_url, price, category,
                   suitable_for_color, personal_color_tags, shades, links, brand_id
            FROM products
            WHERE category = 'blush'
              AND LOWER(suitable_for_color) = %s
              AND status = 'active'
              AND deleted_at IS NULL
            ORDER BY RAND()
            LIMIT %s
        """
        cur.execute(sql, (tone, limit))
    else:
        # undertone ตรวจไม่ได้ (unknown) — ดึงบลัชออนทั่วไปแทน ไม่เดาโทนสีเอง
        sql = """
            SELECT product_id, name, image_url, price, category,
                   suitable_for_color, personal_color_tags, shades, links, brand_id
            FROM products
            WHERE category = 'blush'
              AND status = 'active'
              AND deleted_at IS NULL
            ORDER BY RAND()
            LIMIT %s
        """
        cur.execute(sql, (limit,))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return _attach_platforms([
        {
            "product_id": r["product_id"],
            "name": r["name"],
            "image_url": r["image_url"],
            "price": float(r["price"]) if r["price"] is not None else None,
            "category": r["category"],
            "suitable_for_color": r["suitable_for_color"],
            "personal_color_tags": r["personal_color_tags"],
            "shades": r["shades"],
            "shop_url": r["links"],
        }
        for r in rows
    ])


# ===============================
# ADMIN CRUD
# ===============================
_PRODUCT_FIELDS = {
    "product_id": "product_id",
    "name": "name",
    "image_url": "image_url",
    "price": "price",
    "category": "category",
    "personal_color_tags": "personal_color_tags",
    "status": "status",
    "stock": "stock",
}


def _normalize_product_payload(data: dict):
    payload = {}
    for key, column in _PRODUCT_FIELDS.items():
        if key in data and data[key] is not None:
            payload[column] = data[key]
    if "price" in payload:
        try:
            payload["price"] = float(payload["price"])
        except (TypeError, ValueError):
            payload["price"] = None
    return payload


def insert_product(data: dict):
    payload = _normalize_product_payload(data)
    if not payload or not payload.get("product_id"):
        return None
    if "status" not in payload:
        payload["status"] = "active"

    columns = list(payload.keys())
    values = [payload[c] for c in columns]
    placeholders = ", ".join(["%s"] * len(columns))
    sql = f"INSERT INTO products ({', '.join(columns)}) VALUES ({placeholders})"

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(sql, values)
    conn.commit()
    cur.close()
    conn.close()
    return payload["product_id"]


def update_product(product_id: int, data: dict):
    payload = _normalize_product_payload(data)
    if not payload:
        return False

    conn = get_conn()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                "SELECT 1 FROM products WHERE product_id=%s AND deleted_at IS NULL",
                (product_id,)
            )
            if not cur.fetchone():
                return False

            set_clause = ", ".join([f"{col}=%s" for col in payload.keys()])
            values = list(payload.values()) + [product_id]
            cur.execute(
                f"UPDATE products SET {set_clause} WHERE product_id=%s AND deleted_at IS NULL",
                values
            )
            conn.commit()
            return True
    finally:
        conn.close()


def delete_product(product_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE products SET deleted_at=NOW() WHERE product_id=%s AND deleted_at IS NULL",
        (product_id,)
    )
    conn.commit()
    affected = cur.rowcount
    cur.close()
    conn.close()
    return affected > 0
