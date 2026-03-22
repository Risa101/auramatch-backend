from db import get_conn

def get_all_face():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM face")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_face_by_id(face_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM face WHERE face_id = %s",
        (face_id,)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def insert_face(user_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO face (user_id) VALUES (%s)",
        (user_id,)
    )
    conn.commit()
    new_id = cur.lastrowid
    cur.close()
    conn.close()
    return new_id


def update_face(face_id: int, data: dict):
    if not data:
        return False

    fields = []
    values = []

    for k, v in data.items():
        fields.append(f"{k}=%s")
        values.append(v)

    values.append(face_id)

    sql = f"""
        UPDATE face
        SET {', '.join(fields)}
        WHERE face_id = %s
    """

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(sql, tuple(values))
    conn.commit()
    affected = cur.rowcount
    cur.close()
    conn.close()
    return affected > 0


def delete_face(face_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM face WHERE face_id = %s",
        (face_id,)
    )
    conn.commit()
    affected = cur.rowcount
    cur.close()
    conn.close()
    return affected > 0
