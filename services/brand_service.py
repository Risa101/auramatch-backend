from db import get_conn


def get_all_brands():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM brand WHERE deleted_at IS NULL")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_brand_by_id(bid: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM brand WHERE brand_id=%s AND deleted_at IS NULL",
        (bid,)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def insert_brand(name: str, logo_path: str):
    conn = get_conn()
    cur = conn.cursor()

    sql = """
        INSERT INTO brand (brand_name, logo_path)
        VALUES (%s, %s)
    """
    cur.execute(sql, (name, logo_path))
    new_id = cur.lastrowid

    conn.commit()
    cur.close()
    conn.close()
    return new_id


def update_brand(bid: int, data: dict):
    conn = get_conn()
    cur = conn.cursor()

    sql = """
        UPDATE brand
        SET brand_name=%s,
            logo_path=%s
        WHERE brand_id=%s
    """

    cur.execute(
        sql,
        (
            data.get("brand_name"),
            data.get("logo_path"),
            bid
        )
    )

    conn.commit()
    updated = cur.rowcount > 0

    cur.close()
    conn.close()
    return updated


def delete_brand(bid: int):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "UPDATE brand SET deleted_at=NOW() WHERE brand_id=%s",
        (bid,)
    )
    conn.commit()

    deleted = cur.rowcount > 0

    cur.close()
    conn.close()
    return deleted
