from flask import Flask
from controllers.profiles_controller import profiles_bp

app = Flask(__name__)
app.register_blueprint(profiles_bp)

if __name__ == "__main__":
    app.run(debug=True)
