"""Unit tests for services.product_platform_service -- the multi-platform
(Shopee / Lazada / TikTok Shop) shopping-link layer added on top of the
existing `products` table.

Follows the same convention as tests/test_personal_color_service.py: no live
MySQL connection required. `db.get_conn` is monkeypatched with a tiny fake so
these tests run anywhere, including CI without a database.
"""

import pytest

import services.product_platform_service as pps


# ---------------------------------------------------------------------------
# Pure validation helpers -- no DB involved at all
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("shopee", True),
    ("Lazada", True),          # normalized to lowercase before matching
    ("tiktok_shop", True),
    ("tiktok shop", True),     # spaces normalized to underscores
    ("new-platform-2027", True),  # not in KNOWN_PLATFORMS but still a valid key shape
    ("", False),
    ("a", False),              # too short
    ("x" * 31, False),         # too long
    ("shopee!", False),        # disallowed character
])
def test_is_valid_platform_key(raw, expected):
    assert pps.is_valid_platform_key(raw) is expected


@pytest.mark.parametrize("raw,expected", [
    ("https://shopee.co.th/product/123", True),
    ("http://lazada.co.th/x", True),
    ("javascript:alert(1)", False),
    ("data:text/html,<script>", False),
    ("", False),
    (None, False),
    ("shopee.co.th/no-scheme", False),
])
def test_is_valid_url(raw, expected):
    assert pps.is_valid_url(raw) is expected


def test_serialize_platform_row_shape():
    row = {
        "id": 1,
        "product_id": "BLU002",
        "platform": "shopee",
        "product_url": "https://shopee.co.th/x",
        "affiliate_url": None,
        "is_active": 1,
        "created_at": "irrelevant",
        "updated_at": "irrelevant",
    }
    out = pps.serialize_platform_row(row)
    assert out == {
        "id": 1,
        "platform": "shopee",
        "product_url": "https://shopee.co.th/x",
        "affiliate_url": None,
        "is_active": True,
    }


# ---------------------------------------------------------------------------
# replace_platforms_for_product: invalid input must be rejected WITHOUT ever
# touching the DB (get_conn should not be called at all in these cases).
# ---------------------------------------------------------------------------

def _forbid_db(monkeypatch):
    def _boom():
        raise AssertionError("get_conn() must not be called for invalid input")
    monkeypatch.setattr(pps, "get_conn", _boom)


def test_replace_platforms_rejects_missing_product_id(monkeypatch):
    _forbid_db(monkeypatch)
    ok, err = pps.replace_platforms_for_product(None, [{"platform": "shopee", "product_url": "https://x"}])
    assert ok is False
    assert "product_id" in err


def test_replace_platforms_rejects_empty_list(monkeypatch):
    _forbid_db(monkeypatch)
    ok, err = pps.replace_platforms_for_product("BLU002", [])
    assert ok is False


def test_replace_platforms_rejects_bad_platform_key(monkeypatch):
    _forbid_db(monkeypatch)
    ok, err = pps.replace_platforms_for_product(
        "BLU002", [{"platform": "!!!", "product_url": "https://shopee.co.th/x"}]
    )
    assert ok is False
    assert "invalid platform key" in err


def test_replace_platforms_rejects_unsafe_url(monkeypatch):
    _forbid_db(monkeypatch)
    ok, err = pps.replace_platforms_for_product(
        "BLU002", [{"platform": "shopee", "product_url": "javascript:alert(1)"}]
    )
    assert ok is False
    assert "invalid product_url" in err


# ---------------------------------------------------------------------------
# replace_platforms_for_product: happy path against a fake connection, to
# check the upsert SQL is shaped correctly (executemany with the normalized
# rows) without needing a live MySQL instance.
# ---------------------------------------------------------------------------

class _FakeCursor:
    def __init__(self, existing_product=True):
        self.existing_product = existing_product
        self.executed = []
        self.executemany_calls = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def executemany(self, sql, rows):
        self.executemany_calls.append((sql, rows))

    def fetchone(self):
        return (1,) if self.existing_product else None


class _FakeConn:
    def __init__(self, existing_product=True):
        self.cursor_obj = _FakeCursor(existing_product)
        self.committed = False

    def cursor(self, *a, **kw):
        return self.cursor_obj

    def commit(self):
        self.committed = True

    def close(self):
        pass


def test_replace_platforms_happy_path_upserts_normalized_rows(monkeypatch):
    fake_conn = _FakeConn(existing_product=True)
    monkeypatch.setattr(pps, "get_conn", lambda: fake_conn)

    ok, err = pps.replace_platforms_for_product("BLU002", [
        {"platform": "Shopee", "product_url": "https://shopee.co.th/x", "is_active": True},
        {"platform": "lazada", "product_url": "https://lazada.co.th/y", "is_active": False},
    ])

    assert ok is True
    assert err is None
    assert fake_conn.committed is True
    assert len(fake_conn.cursor_obj.executemany_calls) == 1
    _, rows = fake_conn.cursor_obj.executemany_calls[0]
    assert rows == [
        ("BLU002", "shopee", "https://shopee.co.th/x", None, 1),
        ("BLU002", "lazada", "https://lazada.co.th/y", None, 0),
    ]


def test_replace_platforms_product_not_found(monkeypatch):
    fake_conn = _FakeConn(existing_product=False)
    monkeypatch.setattr(pps, "get_conn", lambda: fake_conn)

    ok, err = pps.replace_platforms_for_product("NOPE", [
        {"platform": "shopee", "product_url": "https://shopee.co.th/x"},
    ])
    assert ok is False
    assert err == "product not found"
    assert fake_conn.committed is False
