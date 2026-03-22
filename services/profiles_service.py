from db import get_conn

def get_all_profiles():
    conn = get_conn()
    cur = conn.cursor()
    sql = """
    SELECT 
        p.user_id,
        p.display_name,
        p.gender,
        p.birthdate,
        p.bio,
        p.aura_color,
        p.skin_tone,
        p.location,
        p.updated_at
    FROM profiles p
    WHERE p.deleted_at IS NULL
    """
    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def get_profile_by_user_id(user_id: int):
    conn = get_conn()
    cur = conn.cursor()
    sql = """
    SELECT 
        p.user_id,
        p.display_name,
        p.gender,
        p.birthdate,
        p.bio,
        p.aura_color,
        p.skin_tone,
        p.location,
        p.updated_at
    FROM profiles p
    WHERE p.user_id = %s AND p.deleted_at IS NULL
    """
    cur.execute(sql, (user_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row
