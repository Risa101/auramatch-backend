# services/promotion_service.py
from db import get_conn

# =========================
# GET : โปรโมชั่นทั้งหมด
# =========================
def get_all_promotion():
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:  # DictCursor
                sql = """
                    SELECT
                        p.promotion_id,
                        p.promo_name,
                        p.promo_detail,
                        p.brand_id,
                        p.discount_percent,
                        p.coupon_code,
                        p.min_price,
                        p.max_discount,
                        p.promo_type,
                        p.season,
                        p.start_date,
                        p.end_date,
                        p.status,
                        p.logo_url,
                        p.superadmin_id,
                        b.brand_name
                    FROM promotion p
                    LEFT JOIN brand b ON p.brand_id = b.brand_id
                    WHERE p.deleted_at IS NULL
                    ORDER BY p.promotion_id DESC
                """
                cur.execute(sql)
                return cur.fetchall()   # ✅ return dict list
    except Exception as e:
        print("[get_all_promotion] Error:", e)
        return []


# =========================
# GET : โปรโมชั่นตาม ID
# =========================
def get_promotion_by_id(pid):
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                sql = """
                    SELECT 
                        p.*,
                        b.brand_name
                    FROM promotion p
                    LEFT JOIN brand b ON p.brand_id = b.brand_id
                    WHERE p.promotion_id = %s
                      AND p.deleted_at IS NULL
                """
                cur.execute(sql, (pid,))
                return cur.fetchone()
    except Exception as e:
        print("[get_promotion_by_id] Error:", e)
        return None


# =========================
# POST : เพิ่มโปรโมชั่น
# =========================
def insert_promotion(data):
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                sql = """
                    INSERT INTO promotion (
                        promo_name,
                        promo_detail,
                        brand_id,
                        discount_percent,
                        coupon_code,
                        min_price,
                        max_discount,
                        promo_type,
                        season,
                        start_date,
                        end_date,
                        status,
                        logo_url,
                        superadmin_id
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """
                cur.execute(sql, (
                    data["promo_name"],
                    data["promo_detail"],
                    data["brand_id"],
                    data.get("discount_percent"),
                    data.get("coupon_code"),
                    data.get("min_price"),
                    data.get("max_discount"),
                    data.get("promo_type"),
                    data.get("season"),
                    data.get("start_date"),
                    data.get("end_date"),
                    data.get("status"),
                    data.get("logo_url"),
                    data["superadmin_id"]
                ))
                conn.commit()
                return cur.lastrowid
    except Exception as e:
        print("[insert_promotion] Error:", e)
        return None


# =========================
# PUT : แก้ไขโปรโมชั่น
# =========================
def update_promotion(pid, data):
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                sql = """
                    UPDATE promotion
                    SET promo_name=%s,
                        promo_detail=%s,
                        brand_id=%s,
                        discount_percent=%s,
                        coupon_code=%s,
                        min_price=%s,
                        max_discount=%s,
                        promo_type=%s,
                        season=%s,
                        start_date=%s,
                        end_date=%s,
                        status=%s,
                        logo_url=%s
                    WHERE promotion_id=%s
                      AND deleted_at IS NULL
                """
                cur.execute(sql, (
                    data.get("promo_name"),
                    data.get("promo_detail"),
                    data.get("brand_id"),
                    data.get("discount_percent"),
                    data.get("coupon_code"),
                    data.get("min_price"),
                    data.get("max_discount"),
                    data.get("promo_type"),
                    data.get("season"),
                    data.get("start_date"),
                    data.get("end_date"),
                    data.get("status"),
                    data.get("logo_url"),
                    pid
                ))
                conn.commit()
                return cur.rowcount > 0
    except Exception as e:
        print("[update_promotion] Error:", e)
        return False


# =========================
# DELETE : Soft delete
# =========================
def delete_promotion(pid):
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                sql = """
                    UPDATE promotion
                    SET deleted_at = NOW()
                    WHERE promotion_id=%s
                      AND deleted_at IS NULL
                """
                cur.execute(sql, (pid,))
                conn.commit()
                return cur.rowcount > 0
    except Exception as e:
        print("[delete_promotion] Error:", e)
        return False
