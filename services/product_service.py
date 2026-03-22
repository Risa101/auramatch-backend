from db import get_conn
import pymysql.cursors

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
    return rows

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
    return rows


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

    return [
        {
            "product_id": r["product_id"],
            "name": r["name"],
            "image_url": r["image_url"],
            "price": float(r["price"]),
            "total_sold": int(r["total_sold"]),
        }
        for r in rows
    ]


# ==========================================
# ✨ RECOMMENDED PRODUCTS (แก้ไขให้ตรงตามตารางจริง)
# ==========================================
def get_recommended_products(season, limit=3):
    """
    ดึงสินค้าที่ตรงกับ personal_color_tags เช่น 'Spring'
    อ้างอิงข้อมูลจริงจาก DB ที่ใช้รูปแบบ 'Warm,Spring'
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
            category
        FROM products 
        WHERE (personal_color_tags LIKE %s OR personal_color_tags LIKE '%%All%%')
          AND status = 'active' 
          AND deleted_at IS NULL
        ORDER BY RAND() 
        LIMIT %s
    """
    
    search_term = f"%{season}%"
    cur.execute(sql, (search_term, limit))
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return [
        {
            "product_id": r["product_id"],
            "name": r["name"],
            "image_url": r["image_url"],
            "price": float(r["price"]),
            "category": r["category"]
        }
        for r in rows
    ]

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
    new_id = cur.lastrowid
    cur.close()
    conn.close()
    return new_id


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
