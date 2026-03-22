from db import get_conn
import pymysql



def _get_qty_column(cur):
    cur.execute(
        """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'stock'
          AND COLUMN_NAME IN ('quantity', 'qty', 'stock_qty')
        ORDER BY FIELD(COLUMN_NAME, 'quantity', 'qty', 'stock_qty')
        LIMIT 1
        """
    )
    row = cur.fetchone()
    if not row:
        return 'quantity'
    return row['COLUMN_NAME'] if isinstance(row, dict) else row[0]


def get_all_stock():
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT * FROM stock")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def get_stock_by_id(sid: int):
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT * FROM stock WHERE stock_id=%s", (sid,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row

def insert_stock(product_id: int, qty: int):
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    qty_column = _get_qty_column(cur)
    cur.execute(
        f"INSERT INTO stock (product_id, {qty_column}) VALUES (%s, %s)",
        (product_id, qty)
    )
    conn.commit()
    new_id = cur.lastrowid
    cur.close()
    conn.close()
    return new_id

def update_stock(sid: int, data: dict):
    if "quantity" not in data:
        return False

    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    qty_column = _get_qty_column(cur)
    cur.execute(
        f"UPDATE stock SET {qty_column}=%s WHERE stock_id=%s",
        (data["quantity"], sid)
    )
    conn.commit()
    affected = cur.rowcount
    cur.close()
    conn.close()
    return affected > 0

def delete_stock(sid: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM stock WHERE stock_id=%s", (sid,))
    conn.commit()
    affected = cur.rowcount
    cur.close()
    conn.close()
    return affected > 0
