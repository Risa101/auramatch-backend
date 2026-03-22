import pymysql
from db import get_conn

def get_all_user_photos():
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT * FROM user_photos")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def insert_user_photo(data):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO user_photos (user_id, photo_url) VALUES (%s, %s)",
        (data["user_id"], data["photo_url"])
    )
    conn.commit()
    new_id = cur.lastrowid
    cur.close()
    conn.close()
    return new_id
