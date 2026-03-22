from db import get_conn

def get_all_eyebrows():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM eyebrow WHERE deleted_at IS NULL")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_eyebrow_by_id(pid: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM eyebrow WHERE eyebrow_id=%s", (pid,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def insert_eyebrow(name: str, price: float, image: str):
    conn = get_conn()
    cur = conn.cursor()

    sql = """
        INSERT INTO eyebrow (name, price, image)
        VALUES (%s, %s, %s)
    """
    cur.execute(sql, (name, price, image))
    new_id = cur.lastrowid

    conn.commit()
    cur.close()
    conn.close()
    return new_id


def update_eyebrow(pid: int, data: dict):
    conn = get_conn()
    cur = conn.cursor()

    sql = """
        UPDATE eyebrow
        SET name=%s, price=%s, image=%s
        WHERE eyebrow_id=%s
    """

    cur.execute(
        sql,
        (
            data.get("name"),
            data.get("price"),
            data.get("image", ""),
            pid
        )
    )

    conn.commit()
    updated = cur.rowcount > 0

    cur.close()
    conn.close()
    return updated


def delete_eyebrow(pid: int):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "UPDATE eyebrow SET deleted_at=NOW() WHERE eyebrow_id=%s",
        (pid,)
    )
    conn.commit()

    deleted = cur.rowcount > 0

    cur.close()
    conn.close()
    return deleted
