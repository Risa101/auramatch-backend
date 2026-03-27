from dotenv import load_dotenv
from flask import Flask, send_from_directory
from flask_cors import CORS
from extensions import limiter
import os

load_dotenv()

def create_app():
    app = Flask(__name__, static_folder='static', static_url_path='/static')
    limiter.init_app(app)
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
        from controllers.favorite_controller import favorite_bp
        from controllers.hairstyle_controller import hairstyle_bp
        from controllers.productcolor_controller import productcolor_bp
        from controllers.producttype_controller import producttype_bp
        from controllers.profiles_controller import profiles_bp
        from controllers.promotion_controller import promotion_bp
        from controllers.review_controller import review_bp
        from controllers.status_controller import status_bp
        from controllers.stock_controller import stock_bp
        from controllers.user_controller import user_bp
        from controllers.user_photos_controller import user_photos_bp
        from controllers.look_controller import looks_bp
        from controllers.brand_controller import brand_bp
        from controllers.analysis_controller import analysis_bp
        from controllers.admin_controller import admin_bp
        from controllers.gemini_controller import gemini_bp

        # ลงทะเบียน Blueprints
        app.register_blueprint(products_bp)
        app.register_blueprint(favorite_bp)
        app.register_blueprint(hairstyle_bp)
        app.register_blueprint(productcolor_bp)
        app.register_blueprint(producttype_bp)
        app.register_blueprint(profiles_bp)
        app.register_blueprint(promotion_bp)
        app.register_blueprint(review_bp)
        app.register_blueprint(status_bp)
        app.register_blueprint(stock_bp)
        app.register_blueprint(user_bp)
        app.register_blueprint(user_photos_bp)
        app.register_blueprint(looks_bp)
        app.register_blueprint(brand_bp)
        app.register_blueprint(admin_bp)
        app.register_blueprint(analysis_bp, url_prefix='/api')
        app.register_blueprint(gemini_bp, url_prefix='/api')

        print("✅ All controllers loaded successfully")

        
    except Exception as e:
        print("❌ controller load failed:", e)
        raise

    @app.route("/")
    def home():
        return {"message": "Auramatch API Ready"}

    images_dir = os.path.join(app.root_path, 'static', 'images')

    @app.route('/brands/<path:filename>')
    def serve_brands(filename):
        return send_from_directory(os.path.join(images_dir, 'brands'), filename)

    @app.route('/avatars/<path:filename>')
    def serve_avatars(filename):
        return send_from_directory(os.path.join(images_dir, 'avatars'), filename)

    @app.route('/assets/<path:filename>')
    def serve_assets(filename):
        return send_from_directory(os.path.join(images_dir, 'assets'), filename)

    @app.route('/makeup/<path:filename>')
    def serve_makeup(filename):
        return send_from_directory(os.path.join(images_dir, 'makeup'), filename)

    @app.route('/hair/<path:filename>')
    def serve_hair(filename):
        return send_from_directory(os.path.join(images_dir, 'hair'), filename)

    @app.route('/faceshape/<path:filename>')
    def serve_faceshape(filename):
        return send_from_directory(os.path.join(images_dir, 'faceshape'), filename)

    @app.route('/overlays/<path:filename>')
    def serve_overlays(filename):
        return send_from_directory(os.path.join(images_dir, 'overlays'), filename)

    @app.route('/eye/<path:filename>')
    def serve_eye(filename):
        return send_from_directory(os.path.join(images_dir, 'products'), filename)

    @app.route('/product/<path:filename>')
    def serve_product(filename):
        return send_from_directory(os.path.join(images_dir, 'products'), filename)

    @app.route('/products/<path:filename>')
    def serve_products(filename):
        return send_from_directory(os.path.join(images_dir, 'products'), filename)

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
