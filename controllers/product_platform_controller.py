from flask import Blueprint, request, jsonify

from services.auth_guard import require_admin
from services.product_platform_service import (
    get_platforms_for_product,
    replace_platforms_for_product,
    delete_platform,
    serialize_platform_row,
)

product_platform_bp = Blueprint("product_platform_bp", __name__)

# NOTE: there is no public `/products/<id>/platforms` route here on purpose.
# app.py already registers `/products/<path:filename>` directly on the Flask
# app (to serve product images), and werkzeug's `path` converter matches
# slashes too — so a route here would collide with/be shadowed by that static
# file route. Public consumers get `platforms` embedded inline instead, via
# product_service._attach_platforms() on /products, /products/<id>, and
# /products/recommendations. Only the /admin/products/... namespace is free
# of that collision, so admin lookup/write endpoints live there.


# ===============================
# ADMIN: GET ALL PLATFORMS FOR A PRODUCT (includes inactive)
# ===============================
@product_platform_bp.route("/admin/products/<string:product_id>/platforms", methods=["GET"])
@require_admin
def admin_list_product_platforms(product_id):
    try:
        rows = get_platforms_for_product(product_id, include_inactive=True)
        return jsonify({
            "status": "success",
            "data": [serialize_platform_row(r) for r in rows],
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ===============================
# ADMIN: BULK UPSERT PLATFORMS FOR A PRODUCT (not a full replace -- platform
# rows omitted from the payload are left untouched; see
# replace_platforms_for_product()'s docstring. Use the DELETE route below to
# remove a platform entirely.)
# body: { "platforms": [{ "platform": "shopee", "product_url": "https://...",
#                          "affiliate_url": null, "is_active": true }, ...] }
# ===============================
@product_platform_bp.route("/admin/products/<string:product_id>/platforms", methods=["PUT"])
@require_admin
def admin_set_product_platforms(product_id):
    data = request.get_json(silent=True) or {}
    platforms = data.get("platforms")
    ok, error = replace_platforms_for_product(product_id, platforms)
    if not ok:
        status = 404 if error == "product not found" else 400
        return jsonify({"error": error}), status

    rows = get_platforms_for_product(product_id, include_inactive=True)
    return jsonify({
        "status": "success",
        "data": [serialize_platform_row(r) for r in rows],
    }), 200


# ===============================
# ADMIN: DELETE ONE PLATFORM FROM A PRODUCT
# ===============================
@product_platform_bp.route("/admin/products/<string:product_id>/platforms/<string:platform>", methods=["DELETE"])
@require_admin
def admin_delete_product_platform(product_id, platform):
    if not delete_platform(product_id, platform):
        return jsonify({"error": "not found"}), 404
    return jsonify({"message": "deleted"}), 200
