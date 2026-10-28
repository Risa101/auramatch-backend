"""
!! DEVELOPMENT / DEMO DATA ONLY !!

Seeds a handful of `product_platforms` rows (Shopee / Lazada / TikTok Shop)
for a few existing products, purely so the new multi-platform shopping UI has
something real to render in local dev.

Every URL below is a placeholder search-style link, NOT a real product page —
do not treat these as production data and do not run this against the
production database. It targets whatever DB the local `db.get_conn()` config
(.env) points at, same as every other script in this folder.

Usage:
    python3 scripts/seed_demo_product_platforms.py [--dry-run]

Requires migrations/2026-08-30_add_product_platforms.sql to already be
applied (creates the `product_platforms` table).
"""

import argparse
import sys

sys.path.insert(0, ".")

from dotenv import load_dotenv

load_dotenv()

from db import get_conn  # noqa: E402
from services.product_platform_service import replace_platforms_for_product  # noqa: E402

# Marked test/demo per project instructions (section 11): fabricated search
# URLs, not real per-product deep links. Swap for real product_url values via
# the admin API before shipping to production.
DEMO_SEED = [
    {
        "product_id": "BLU002",
        "platforms": [
            {"platform": "shopee", "product_url": "https://shopee.co.th/search?keyword=Rare+Beauty+Soft+Pinch+DEMO"},
            {"platform": "lazada", "product_url": "https://www.lazada.co.th/catalog/?q=Rare+Beauty+Soft+Pinch+DEMO"},
            {"platform": "tiktok_shop", "product_url": "https://www.tiktok.com/search/video?q=Rare+Beauty+Soft+Pinch+DEMO"},
        ],
    },
    {
        "product_id": "BLU003",
        "platforms": [
            {"platform": "shopee", "product_url": "https://shopee.co.th/search?keyword=3CE+Face+Blush+DEMO"},
            {"platform": "tiktok_shop", "product_url": "https://www.tiktok.com/search/video?q=3CE+Face+Blush+DEMO", "is_active": False},
        ],
    },
]


def _product_exists(conn, product_id):
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM products WHERE product_id=%s AND deleted_at IS NULL", (product_id,))
        return cur.fetchone() is not None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print what would be seeded without writing")
    args = parser.parse_args()

    conn = get_conn()
    try:
        for entry in DEMO_SEED:
            pid = entry["product_id"]
            if not _product_exists(conn, pid):
                print(f"skip {pid}: product not found in this DB")
                continue
            if args.dry_run:
                print(f"[dry-run] would upsert {len(entry['platforms'])} platform(s) for {pid}")
                continue
            ok, err = replace_platforms_for_product(pid, entry["platforms"])
            print(f"{pid}: {'ok' if ok else 'FAILED - ' + str(err)}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
