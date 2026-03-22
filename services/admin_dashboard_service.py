from db import get_conn
from services.product_service import get_best_seller_products


def get_admin_overview():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) AS total_users FROM `user` WHERE deleted_at IS NULL")
    total_users = cur.fetchone()["total_users"]

    cur.execute("SELECT COUNT(*) AS total_products FROM products WHERE deleted_at IS NULL")
    total_products = cur.fetchone()["total_products"]

    cur.execute("SELECT COUNT(*) AS total_promotions FROM promotion WHERE deleted_at IS NULL")
    total_promotions = cur.fetchone()["total_promotions"]

    cur.execute("SELECT COUNT(*) AS total_reviews FROM review")
    total_reviews = cur.fetchone()["total_reviews"]

    cur.execute(
        """
        SELECT user_id, username, email, created_at
        FROM `user`
        WHERE deleted_at IS NULL
        ORDER BY created_at DESC
        LIMIT 5
        """
    )
    recent_users = cur.fetchall()

    low_stock = []
    for column in ("quantity", "qty", "stock_qty"):
        try:
            cur.execute(
                f"""
                SELECT s.product_id, s.{column} AS quantity, p.name
                FROM stock s
                LEFT JOIN products p ON s.product_id = p.product_id
                ORDER BY s.{column} ASC
                LIMIT 5
                """
            )
            low_stock = cur.fetchall()
            break
        except Exception:
            low_stock = []

    cur.close()
    conn.close()

    return {
        "kpis": {
            "users": total_users,
            "products": total_products,
            "promotions": total_promotions,
            "reviews": total_reviews,
        },
        "best_sellers": get_best_seller_products(),
        "recent_users": recent_users,
        "low_stock": low_stock,
    }
