from dotenv import load_dotenv
from flask import Flask, send_from_directory
from flask_cors import CORS
import os

load_dotenv()

def create_app():
    app = Flask(__name__, static_folder='static', static_url_path='/static')
    frontend_url = os.getenv("FRONTEND_BASE_URL", "http://localhost:5173").rstrip("/")
    allowed_origins = [
        frontend_url,
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ]
    CORS(app, resources={r"/*": {"origins": allowed_origins}})

    try:
        # นำเข้า Blueprint จาก Controller ต่างๆ
        from controllers.product_controller import products_bp
        from controllers.eyebrow_controller import eyebrows_bp
        from controllers.eyeshape_controller import eyeshape_bp
        from controllers.face_controller import face_bp
        from controllers.facetype_controller import facetype_bp
        from controllers.faceproduct_controller import faceproduct_bp
        from controllers.favorite_controller import favorite_bp
        from controllers.haircolor_controller import haircolor_bp
        from controllers.hairstyle_controller import hairstyle_bp
        from controllers.productcolor_controller import productcolor_bp
        from controllers.producttype_controller import producttype_bp
        from controllers.liptone_controller import liptone_bp
        from controllers.profiles_controller import profiles_bp
        from controllers.promotion_controller import promotion_bp
        from controllers.review_controller import review_bp
        from controllers.skintone_controller import skintone_bp
        from controllers.status_controller import status_bp
        from controllers.stock_controller import stock_bp
        from controllers.superadmin_controller import superadmin_bp
        from controllers.user_controller import user_bp
        from controllers.user_photos_controller import user_photos_bp
        from controllers.youtube_controller import youtube_bp
        from controllers.look_controller import looks_bp
        from controllers.brand_controller import brand_bp
        from controllers.analysis_controller import analysis_bp
        from controllers.admin_controller import admin_bp
        from controllers.gemini_controller import gemini_bp

        # ลงทะเบียน Blueprints
        app.register_blueprint(products_bp)
        app.register_blueprint(eyebrows_bp)
        app.register_blueprint(eyeshape_bp)
        app.register_blueprint(face_bp)
        app.register_blueprint(facetype_bp)
        app.register_blueprint(faceproduct_bp)
        app.register_blueprint(favorite_bp)
        app.register_blueprint(haircolor_bp)
        app.register_blueprint(hairstyle_bp)
        app.register_blueprint(productcolor_bp)
        app.register_blueprint(producttype_bp)
        app.register_blueprint(liptone_bp)
        app.register_blueprint(profiles_bp)
        app.register_blueprint(promotion_bp)
        app.register_blueprint(review_bp)
        app.register_blueprint(skintone_bp)
        app.register_blueprint(status_bp)
        app.register_blueprint(stock_bp)
        app.register_blueprint(superadmin_bp)
        app.register_blueprint(user_bp)
        app.register_blueprint(user_photos_bp)
        app.register_blueprint(youtube_bp)
        app.register_blueprint(looks_bp)
        app.register_blueprint(brand_bp)
        app.register_blueprint(admin_bp)
        # ต้องลงทะเบียนแบบนี้เพื่อให้เป็น /api/analysis-history
        app.register_blueprint(analysis_bp, url_prefix='/api')
        app.register_blueprint(gemini_bp, url_prefix='/api')

        print("✅ All controllers loaded successfully")

        
    except Exception as e:
        print("❌ controller load failed:", e)
        raise

    @app.route("/")
    def home():
        return {"message": "Auramatch API Ready"}

    # สำหรับจัดการ Error 404 ที่คุณเจอ:
    # เมื่อเรียก /brands/xxx.png ให้ไปดึงจาก static/images/xxx.png
    @app.route('/brands/<path:filename>')
    def custom_static(filename):
        return send_from_directory(os.path.join(app.root_path, 'static', 'images'), filename)

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.getenv("PORT", 5010))
    debug = os.getenv("FLASK_ENV", "development") == "development"
    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug,
        use_reloader=False
    )
