from db import get_conn
import pymysql.cursors

# ==================================================
# GET ALL PRODUCTS
# ==================================================
def model_get_all_products(filters):
    conn = get_conn()
    # ใช้ DictCursor เพื่อให้ได้ผลลัพธ์เป็น Dictionary
    cur = conn.cursor(pymysql.cursors.DictCursor)

    sql = """
        SELECT *
        FROM products 
        WHERE deleted_at IS NULL
          AND status = 'active'
    """
    params = []

    # กรองตาม Category
    if "category" in filters:
        sql += " AND category = %s"
        params.append(filters["category"])

    # กรองตาม Personal Color (Season)
    if "season" in filters:
        sql += " AND (personal_color_tags LIKE %s OR personal_color_tags LIKE '%%All%%')"
        params.append(f"%{filters['season']}%")

    # การเรียงลำดับ
    if "sort" in filters and filters["sort"] == "price_desc":
        sql += " ORDER BY price DESC"
    else:
        sql += " ORDER BY created_at DESC"

    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

# ==================================================
# GET PRODUCT BY ID
# ==================================================
def model_get_product_by_id(pid):
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)

    cur.execute("""
        SELECT *
        FROM products
        WHERE product_id = %s
          AND deleted_at IS NULL
          AND status = 'active'
    """, (pid,))

    row = cur.fetchone()
    cur.close()
    conn.close()
    return row

# ==================================================
# 🔥 BEST SELLER
# ==================================================
def model_get_best_seller_products(limit):
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)

    cur.execute("""
        SELECT
            p.product_id,
            p.name,
            p.image_url,
            p.price,
            COALESCE(SUM(oi.qty), 0) AS total_sold
        FROM products p
        LEFT JOIN order_items oi ON p.product_id = oi.product_id
        LEFT JOIN orders o ON oi.order_id = o.order_id
        WHERE (o.status = 'completed' OR o.status IS NULL)
          AND p.deleted_at IS NULL
          AND p.status = 'active'
        GROUP BY p.product_id, p.name, p.image_url, p.price
        ORDER BY total_sold DESC
        LIMIT %s
    """, (limit,))

    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

# ==================================================
# ✨ RECOMMENDED BY SEASON (เพิ่มฟังก์ชันนี้ให้ตรงกับหน้าบ้าน)
# ==================================================
def model_get_recommended_products(season, limit=3):
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)

    sql = """
        SELECT product_id, name, image_url, price, category
        FROM products 
        WHERE (personal_color_tags LIKE %s OR personal_color_tags LIKE '%%All%%')
          AND status = 'active' 
          AND deleted_at IS NULL
        ORDER BY RAND() 
        LIMIT %s
    """
    cur.execute(sql, (f"%{season}%", limit))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

# ==================================================
# INSERT PRODUCT (เพิ่มคอลัมน์ตามโครงสร้างจริง)
# ==================================================
def model_insert_product(data):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO products (
            product_id, name, price, category, brand_id, 
            image_url, rating, status, personal_color_tags,
            finish_type, coverage_level, suitable_for_skin_type,
            created_at
        ) VALUES (
            %s,%s,%s,%s,%s,%s,%s,'active',%s,%s,%s,%s,NOW()
        )
    """, (
        data["product_id"],
        data["name"],
        data["price"],
        data.get("category"),
        data.get("brand_id"),
        data.get("image_url"),
        data.get("rating", 0),
        data.get("personal_color_tags"),
        data.get("finish_type"),
        data.get("coverage_level"),
        data.get("suitable_for_skin_type")
    ))

    conn.commit()
    cur.close()
    conn.close()

# ==================================================
# UPDATE PRODUCT
# ==================================================
def model_update_product(pid, data):
    conn = get_conn()
    cur = conn.cursor()

    fields = []
    values = []

    for k, v in data.items():
        fields.append(f"{k}=%s")
        values.append(v)

    if not fields:
        return False

    values.append(pid)

    cur.execute(f"""
        UPDATE products
        SET {",".join(fields)}, updated_at = NOW()
        WHERE product_id = %s
          AND deleted_at IS NULL
    """, values)

    updated = cur.rowcount > 0
    conn.commit()
    cur.close()
    conn.close()
    return updated

# ==================================================
# DELETE PRODUCT (SOFT)
# ==================================================
def model_delete_product(pid):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        UPDATE products
        SET deleted_at = NOW(), status='inactive'
        WHERE product_id = %s
    """, (pid,))

    deleted = cur.rowcount > 0
    conn.commit()
    cur.close()
    conn.close()
    return deleted