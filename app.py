"""
Vehicle Identity System - Main Flask Application

This application provides a complete vehicle identity management system
with features for:
- Vehicle detection and plate OCR
- Identity clustering (plate → face → visual features)
- Manual verification and correction
- Admin dashboard with statistics
- Session-based authentication for admin routes
"""

import sys
import os
from datetime import timedelta

# ---------------------------------------------------------------------
# IMPORT FIX (CRITICAL)
# ---------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
# ---------------------------------------------------------------------

from flask import Flask

# Create Flask app
app = Flask(__name__, template_folder='Views', static_folder='static')

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///vehicle_identity.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
app.secret_key = 'vehicle-identity-system-secret-key-change-in-production'

# Session Configuration
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)  # Remember me duration

# Initialize Database
from Models import db
db.init_app(app)

# Create tables
with app.app_context():
    db.create_all()
    print("[APP] Database tables created/verified")

# ---------------------------------------------------------------------
# REGISTER BLUEPRINTS
# ---------------------------------------------------------------------

# Auth routes (login/logout) - Register FIRST
from Routes.auth_routes import auth_bp
app.register_blueprint(auth_bp)

# Main API routes (upload, video, webcam)
from Routes.api import api
app.register_blueprint(api)

# Admin UI routes (protected with login)
from Routes.admin_routes import admin_bp
app.register_blueprint(admin_bp)

# Vehicle API (JSON endpoints)
from Routes.vehicle_api import vehicle_api
app.register_blueprint(vehicle_api)

# ---------------------------------------------------------------------
# ENSURE DIRECTORIES
# ---------------------------------------------------------------------
from Libs.config import ensure_directories
ensure_directories()

# ---------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------
if __name__ == '__main__':
    print(f"[APP] Starting Vehicle Identity System from {current_dir}")
    print("[APP] Available routes:")
    print("  - / (Guest Home)")
    print("  - /login (Login Page)")
    print("  - /logout (Logout)")
    print("  - /admin (Admin Dashboard) [Protected]")
    print("  - /admin/vehicles (Vehicle Identities) [Protected]")
    print("  - /admin/observations (All Observations) [Protected]")
    print("  - /admin/gallery (Gallery) [Protected]")
    print("  - /admin/merge (Merge & Split) [Protected]")
    print("  - /admin/settings (Settings) [Protected]")
    print("  - /api/... (JSON API)")
    print("[APP] Default login: admin / admin123")
    app.run(host='0.0.0.0', port=5000, debug=True)
