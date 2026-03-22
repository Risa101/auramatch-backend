import sys
from werkzeug.security import generate_password_hash

from db import get_conn

def _is_password_hash(value):
    return isinstance(value, str) and value.startswith(("pbkdf2:", "scrypt:", "argon2:"))


def main():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT user_id, password FROM user WHERE password IS NOT NULL")
    rows = cur.fetchall()

    updated = 0
    for row in rows:
        user_id = row["user_id"]
        password = row["password"] or ""
        if _is_password_hash(password):
            continue
        new_hash = generate_password_hash(password)
        cur.execute(
            "UPDATE user SET password=%s WHERE user_id=%s",
            (new_hash, user_id),
        )
        updated += 1

    conn.commit()
    cur.close()
    conn.close()

    print(f"Updated {updated} users.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
