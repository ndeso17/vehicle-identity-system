"""
Admin Routes - UI pages for vehicle identity management.
All routes protected with @login_required decorator.
"""

from flask import Blueprint, render_template, request, redirect, url_for
from Models import db, VehicleIdentity, VehicleObservation, AuditLog
from Libs.config import VEHICLE_UI_PER_PAGE, OBSERVATIONS_PER_PAGE
from Libs.auth import login_required

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')


# Apply login_required to ALL routes in this blueprint
@admin_bp.before_request
def require_login():
    """Protect all admin routes with login requirement."""
    from flask import session
    if 'user' not in session:
        from flask import flash
        flash('Silakan login terlebih dahulu', 'warning')
        session['next'] = request.url
        return redirect(url_for('auth.login'))


@admin_bp.route('/')
@admin_bp.route('/dashboard')
def dashboard():
    """Admin dashboard with statistics"""
    
    # Get statistics
    total_identities = VehicleIdentity.query.count()
    verified_count = VehicleIdentity.query.filter_by(verified=True).count()
    unverified_count = total_identities - verified_count
    total_observations = VehicleObservation.query.count()
    
    # Recent observations
    recent_observations = VehicleObservation.query.order_by(
        VehicleObservation.timestamp.desc()
    ).limit(10).all()
    
    # Identity method breakdown
    plate_based = VehicleIdentity.query.filter_by(identity_method='plate').count()
    face_based = VehicleIdentity.query.filter_by(identity_method='face').count()
    visual_based = VehicleIdentity.query.filter_by(identity_method='visual').count()
    
    # Vehicle type breakdown
    from sqlalchemy import func
    type_stats = db.session.query(
        VehicleIdentity.vehicle_type,
        func.count(VehicleIdentity.id)
    ).group_by(VehicleIdentity.vehicle_type).all()
    
    return render_template('admin/dashboard.html',
        total_identities=total_identities,
        verified_count=verified_count,
        unverified_count=unverified_count,
        total_observations=total_observations,
        recent_observations=recent_observations,
        plate_based=plate_based,
        face_based=face_based,
        visual_based=visual_based,
        type_stats=type_stats
    )



@admin_bp.route('/vehicles')
def vehicles_list():
    """Vehicle identity gallery view"""
    
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '').strip()
    status_filter = request.args.get('status', 'all')
    method_filter = request.args.get('method', 'all')
    type_filter = request.args.get('type', 'all')
    
    # Base query
    query = VehicleIdentity.query
    
    # Apply filters
    if search:
        query = query.filter(VehicleIdentity.plate_text.ilike(f'%{search}%'))
    
    if status_filter == 'verified':
        query = query.filter_by(verified=True)
    elif status_filter == 'unverified':
        query = query.filter_by(verified=False)
    
    if method_filter in ['plate', 'face', 'visual']:
        query = query.filter_by(identity_method=method_filter)
    
    if type_filter in ['car', 'motorcycle', 'bus', 'truck']:
        query = query.filter_by(vehicle_type=type_filter)
    
    # Order by last seen
    query = query.order_by(VehicleIdentity.last_seen.desc())
    
    # Paginate
    pagination = query.paginate(page=page, per_page=VEHICLE_UI_PER_PAGE, error_out=False)
    
    return render_template('admin/vehicles.html',
        vehicles=pagination.items,
        pagination=pagination,
        search=search,
        status_filter=status_filter,
        method_filter=method_filter,
        type_filter=type_filter
    )


@admin_bp.route('/vehicles/<int:id>')
def vehicle_detail(id):
    """Vehicle identity detail page"""
    
    identity = VehicleIdentity.query.get_or_404(id)
    
    # Get observations with pagination
    page = request.args.get('page', 1, type=int)
    observations = identity.observations.order_by(
        VehicleObservation.timestamp.desc()
    ).paginate(page=page, per_page=20, error_out=False)
    
    # Get merge history
    merge_history = identity.merge_history
    if merge_history:
        import json
        merge_history = json.loads(merge_history)
    else:
        merge_history = []
    
    return render_template('admin/vehicle_detail.html',
        vehicle=identity,
        observations=observations,
        merge_history=merge_history
    )


@admin_bp.route('/observations')
def observations_list():
    """All observations table view"""
    
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '').strip()
    source_filter = request.args.get('source', 'all')
    ocr_filter = request.args.get('ocr', 'all')
    
    # Base query
    query = VehicleObservation.query
    
    # Apply filters
    if search:
        query = query.filter(VehicleObservation.plate_text.ilike(f'%{search}%'))
    
    if source_filter in ['image', 'video', 'webcam', 'ipcam']:
        query = query.filter_by(source_type=source_filter)
    
    if ocr_filter == 'success':
        query = query.filter_by(ocr_success=True)
    elif ocr_filter == 'failed':
        query = query.filter_by(ocr_success=False)
    
    # Order by timestamp
    query = query.order_by(VehicleObservation.timestamp.desc())
    
    # Paginate
    pagination = query.paginate(page=page, per_page=OBSERVATIONS_PER_PAGE, error_out=False)
    
    return render_template('admin/observations.html',
        observations=pagination.items,
        pagination=pagination,
        search=search,
        source_filter=source_filter,
        ocr_filter=ocr_filter
    )


@admin_bp.route('/observations/<int:id>')
def observation_detail(id):
    """Single observation detail"""
    
    observation = VehicleObservation.query.get_or_404(id)
    identity = observation.identity
    
    return render_template('admin/observation_detail.html',
        observation=observation,
        vehicle=identity
    )


@admin_bp.route('/merge')
def merge_page():
    """Merge & Split page"""
    
    # Get all identities for selection
    identities = VehicleIdentity.query.order_by(VehicleIdentity.last_seen.desc()).all()
    
    return render_template('admin/merge_split.html',
        identities=identities
    )


@admin_bp.route('/settings')
def settings():
    """Settings page"""
    
    from Libs.config import (
        PLATE_PRIMARY_CONF, FACE_SIM_THRESHOLD, 
        CLUSTER_MATCH_THRESHOLD, WEIGHT_PLATE, WEIGHT_FACE
    )
    
    # Get audit log
    recent_actions = AuditLog.query.order_by(
        AuditLog.timestamp.desc()
    ).limit(20).all()
    
    return render_template('admin/settings.html',
        plate_threshold=PLATE_PRIMARY_CONF,
        face_threshold=FACE_SIM_THRESHOLD,
        cluster_threshold=CLUSTER_MATCH_THRESHOLD,
        weight_plate=WEIGHT_PLATE,
        weight_face=WEIGHT_FACE,
        recent_actions=recent_actions
    )


@admin_bp.route('/gallery')
def gallery():
    """
    Gallery page - Google Photos style grouping.
    
    Groups vehicles by:
    1. Plate text (primary) - if OCR successful
    2. Driver face (fallback) - if plate not available
    """
    
    page = request.args.get('page', 1, type=int)
    group_by = request.args.get('group_by', 'plate')  # 'plate' or 'driver'
    
    # Get all identities from database
    identities = VehicleIdentity.query.order_by(VehicleIdentity.last_seen.desc()).all()
    
    # Build gallery groups
    gallery_groups = []
    
    if identities:
        # Real data from database
        for identity in identities:
            # Determine group label based on priority
            if identity.plate_text:
                group_label = identity.plate_text
                group_type = 'plate'
            elif identity.face_embedding:
                group_label = f"Driver #{identity.id}"
                group_type = 'driver'
            else:
                group_label = f"Vehicle #{identity.id}"
                group_type = 'visual'
            
            # Get sample images (up to 4 for preview)
            sample_observations = identity.observations.order_by(
                VehicleObservation.timestamp.desc()
            ).limit(4).all()
            
            gallery_groups.append({
                'id': identity.id,
                'label': group_label,
                'type': group_type,
                'plate_text': identity.plate_text,
                'vehicle_type': identity.vehicle_type or 'Unknown',
                'vehicle_color': identity.vehicle_color,
                'count': identity.detection_count or len(sample_observations),
                'representative_image': identity.representative_image,
                'sample_images': [obs.image_path for obs in sample_observations if obs.image_path],
                'first_seen': identity.first_seen,
                'last_seen': identity.last_seen,
                'verified': identity.verified,
                'identity_method': identity.identity_method or 'visual',
                'has_driver_face': identity.face_embedding is not None
            })
    else:
        # Dummy data for demonstration when no real data exists
        import datetime
        dummy_data = [
            {
                'id': 1,
                'label': 'B 1234 XYZ',
                'type': 'plate',
                'plate_text': 'B 1234 XYZ',
                'vehicle_type': 'car',
                'vehicle_color': 'white',
                'count': 5,
                'representative_image': None,
                'sample_images': [],
                'first_seen': datetime.datetime.now() - datetime.timedelta(days=7),
                'last_seen': datetime.datetime.now(),
                'verified': True,
                'identity_method': 'plate',
                'has_driver_face': True
            },
            {
                'id': 2,
                'label': 'DK 5678 ABC',
                'type': 'plate',
                'plate_text': 'DK 5678 ABC',
                'vehicle_type': 'motorcycle',
                'vehicle_color': 'black',
                'count': 3,
                'representative_image': None,
                'sample_images': [],
                'first_seen': datetime.datetime.now() - datetime.timedelta(days=3),
                'last_seen': datetime.datetime.now() - datetime.timedelta(hours=2),
                'verified': False,
                'identity_method': 'plate',
                'has_driver_face': True
            },
            {
                'id': 3,
                'label': 'Driver #3',
                'type': 'driver',
                'plate_text': None,
                'vehicle_type': 'car',
                'vehicle_color': 'silver',
                'count': 2,
                'representative_image': None,
                'sample_images': [],
                'first_seen': datetime.datetime.now() - datetime.timedelta(days=1),
                'last_seen': datetime.datetime.now() - datetime.timedelta(hours=5),
                'verified': False,
                'identity_method': 'face',
                'has_driver_face': True
            },
            {
                'id': 4,
                'label': 'L 9999 DEF',
                'type': 'plate',
                'plate_text': 'L 9999 DEF',
                'vehicle_type': 'truck',
                'vehicle_color': 'red',
                'count': 8,
                'representative_image': None,
                'sample_images': [],
                'first_seen': datetime.datetime.now() - datetime.timedelta(days=14),
                'last_seen': datetime.datetime.now() - datetime.timedelta(days=1),
                'verified': True,
                'identity_method': 'plate',
                'has_driver_face': False
            },
            {
                'id': 5,
                'label': 'Vehicle #5',
                'type': 'visual',
                'plate_text': None,
                'vehicle_type': 'motorcycle',
                'vehicle_color': 'blue',
                'count': 1,
                'representative_image': None,
                'sample_images': [],
                'first_seen': datetime.datetime.now() - datetime.timedelta(hours=3),
                'last_seen': datetime.datetime.now() - datetime.timedelta(hours=3),
                'verified': False,
                'identity_method': 'visual',
                'has_driver_face': False
            },
            {
                'id': 6,
                'label': 'Driver #6',
                'type': 'driver',
                'plate_text': None,
                'vehicle_type': 'car',
                'vehicle_color': 'black',
                'count': 4,
                'representative_image': None,
                'sample_images': [],
                'first_seen': datetime.datetime.now() - datetime.timedelta(days=5),
                'last_seen': datetime.datetime.now() - datetime.timedelta(hours=1),
                'verified': False,
                'identity_method': 'face',
                'has_driver_face': True
            }
        ]
        gallery_groups = dummy_data
    
    # Filter by group type if requested
    if group_by == 'plate':
        gallery_groups = [g for g in gallery_groups if g['type'] == 'plate']
    elif group_by == 'driver':
        gallery_groups = [g for g in gallery_groups if g['type'] == 'driver']
    
    # Statistics for the header
    total_groups = len(gallery_groups)
    plate_groups = sum(1 for g in gallery_groups if g['type'] == 'plate')
    driver_groups = sum(1 for g in gallery_groups if g['type'] == 'driver')
    visual_groups = sum(1 for g in gallery_groups if g['type'] == 'visual')
    
    return render_template('admin/gallery.html',
        groups=gallery_groups,
        group_by=group_by,
        total_groups=total_groups,
        plate_groups=plate_groups,
        driver_groups=driver_groups,
        visual_groups=visual_groups,
        is_dummy_data=len(identities) == 0
    )


@admin_bp.route('/gallery/<int:id>')
def gallery_detail(id):
    """Gallery group detail - shows all photos in a group"""
    
    identity = VehicleIdentity.query.get(id)
    
    if identity:
        # Real data
        observations = identity.observations.order_by(
            VehicleObservation.timestamp.desc()
        ).all()
        
        group_data = {
            'id': identity.id,
            'label': identity.plate_text or f"Vehicle #{identity.id}",
            'type': identity.identity_method or 'visual',
            'plate_text': identity.plate_text,
            'vehicle_type': identity.vehicle_type,
            'vehicle_color': identity.vehicle_color,
            'verified': identity.verified,
            'first_seen': identity.first_seen,
            'last_seen': identity.last_seen,
            'photos': [{
                'id': obs.id,
                'image_path': obs.image_path,
                'annotated_path': obs.annotated_image_path,
                'plate_image': obs.plate_image_path,
                'driver_image': obs.driver_image_path,
                'timestamp': obs.timestamp,
                'source_type': obs.source_type or 'image',
                'plate_text': obs.plate_text,
                'plate_confidence': obs.plate_confidence,
                'ocr_success': obs.ocr_success
            } for obs in observations]
        }
    else:
        # Dummy data for demo
        import datetime
        group_data = {
            'id': id,
            'label': f"Demo Group #{id}",
            'type': 'plate',
            'plate_text': 'B 1234 XYZ',
            'vehicle_type': 'car',
            'vehicle_color': 'white',
            'verified': False,
            'first_seen': datetime.datetime.now() - datetime.timedelta(days=7),
            'last_seen': datetime.datetime.now(),
            'photos': [
                {
                    'id': 1,
                    'image_path': None,
                    'annotated_path': None,
                    'plate_image': None,
                    'driver_image': None,
                    'timestamp': datetime.datetime.now() - datetime.timedelta(hours=i),
                    'source_type': ['image', 'webcam', 'video', 'ipcam'][i % 4],
                    'plate_text': 'B 1234 XYZ',
                    'plate_confidence': 0.85 - (i * 0.05),
                    'ocr_success': True
                } for i in range(5)
            ]
        }
    
    return render_template('admin/gallery_detail.html',
        group=group_data
    )
