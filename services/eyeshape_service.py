from db import get_conn

def get_all_eyeshape():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM eyeshape")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_eyeshape_by_id(pid: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM eyeshape WHERE eyeshape_id=%s",
        (pid,)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def insert_eyeshape(face_id: int, shape_name: str):
    conn = get_conn()
    cur = conn.cursor()

    sql = """
        INSERT INTO eyeshape (face_id, shape_name)
        VALUES (%s, %s)
    """
    cur.execute(sql, (face_id, shape_name))
    new_id = cur.lastrowid

    conn.commit()
    cur.close()
    conn.close()
    return new_id


def update_eyeshape(pid: int, data: dict):
    conn = get_conn()
    cur = conn.cursor()

    sql = """
        UPDATE eyeshape
        SET face_id=%s, shape_name=%s
        WHERE eyeshape_id=%s
    """

    cur.execute(
        sql,
        (
            data.get("face_id"),
            data.get("shape_name"),
            pid
        )
    )

    conn.commit()
    updated = cur.rowcount > 0

    cur.close()
    conn.close()
    return updated


def delete_eyeshape(pid: int):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "DELETE FROM eyeshape WHERE eyeshape_id=%s",
        (pid,)
    )
    conn.commit()

    deleted = cur.rowcount > 0

    cur.close()
    conn.close()
    return deleted
