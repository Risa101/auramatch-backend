# services/youtube_service.py
from db import get_conn

def get_youtube_videos():
    conn = get_conn()
    cur = conn.cursor()  # ใช้ DictCursor

    sql = """
        SELECT
            youtube_id,
            video_url
        FROM youtube
        WHERE deleted_at IS NULL
        ORDER BY youtube_id DESC
    """

    cur.execute(sql)
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return rows
