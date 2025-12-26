"""
API Routes - Main application routes for upload, video, and webcam.
"""

from flask import Blueprint, request, render_template, jsonify, Response
from Controllers.api_controller import ApiController
from Models import db, VehicleIdentity, VehicleObservation
from Libs.config import VEHICLE_UI_PER_PAGE

api = Blueprint('api', __name__)


# ========================================
# UI ROUTES
# ========================================

@api.route('/', methods=['GET'])
def index():
    """Guest home page with statistics"""
    from collections import Counter
    from sqlalchemy import func
    
    # Get statistics from database
    type_counter = Counter()
    color_counter = Counter()
    letter_counter = Counter()
    total_detections = 0
    
    # Query all identities
    identities = VehicleIdentity.query.all()
    
    for identity in identities:
        # Vehicle type
        if identity.vehicle_type:
            type_counter[identity.vehicle_type] += identity.detection_count or 1
        
        # Vehicle color
        if identity.vehicle_color:
            color_counter[identity.vehicle_color] += identity.detection_count or 1
        
        # Plate letter (first character)
        if identity.plate_text and len(identity.plate_text) > 0:
            letter = identity.plate_text[0].upper()
            if letter.isalpha():
                letter_counter[letter] += 1
        
        total_detections += identity.detection_count or 1
    
    # Prepare stats for template
    type_stats = [{'label': k, 'count': v} for k, v in type_counter.most_common()]
    color_stats = [{'label': k, 'count': v} for k, v in color_counter.most_common()]
    letter_stats = [{'label': k, 'count': v} for k, v in sorted(letter_counter.items(), key=lambda x: (-x[1], x[0]))]
    
    total_vehicles = len(identities)
    
    return render_template('index.html',
                           type_stats=type_stats,
                           color_stats=color_stats,
                           letter_stats=letter_stats,
                           total_vehicles=total_vehicles,
                           total_detections=total_detections)


@api.route('/video_ui', methods=['GET'])
def video_ui():
    """Video streaming UI (webcam/ipcam)"""
    source = request.args.get('source', 'webcam')
    url = request.args.get('url', '')
    return render_template('video.html', source=source, url=url)


# ========================================
# API ENDPOINTS
# ========================================

@api.route('/api/image', methods=['POST'])
def process_image():
    """
    Process uploaded image through the pipeline.
    Returns rendered HTML result.
    """
    return ApiController.upload_image(request)


@api.route('/api/webcam')
def webcam_feed():
    """MJPEG stream from webcam"""
    return ApiController.video_feed_webcam()


@api.route('/api/ipcam')
def ipcam_feed():
    """MJPEG stream from IP camera"""
    url = request.args.get('url')
    if not url:
        return "Missing URL", 400
    return ApiController.video_feed_ipcam(url)


# ========================================
# LEGACY CLUSTER ROUTES (redirect to new admin)
# ========================================

@api.route('/clusters')
def clusters_ui():
    """Redirect to new vehicles page"""
    from flask import redirect, url_for
    return redirect(url_for('admin.vehicles_list'))


@api.route('/clusters/<cluster_id>')
def cluster_detail(cluster_id):
    """Redirect to new vehicle detail page"""
    from flask import redirect, url_for
    # Try to find by ID
    try:
        identity_id = int(cluster_id)
        return redirect(url_for('admin.vehicle_detail', id=identity_id))
    except ValueError:
        # If it's not a numeric ID, try to find by plate
        identity = VehicleIdentity.query.filter_by(plate_text=cluster_id).first()
        if identity:
            return redirect(url_for('admin.vehicle_detail', id=identity.id))
        return "Not found", 404
