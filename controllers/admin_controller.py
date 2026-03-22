from flask import Blueprint, jsonify

from services.admin_dashboard_service import get_admin_overview
from services.auth_guard import require_admin

admin_bp = Blueprint("admin_bp", __name__)


@admin_bp.route("/admin/overview", methods=["GET"])
@require_admin
def admin_overview():
    return jsonify(get_admin_overview()), 200
