"""
Vehicle API - JSON API endpoints for vehicle identity management.
"""

from flask import Blueprint, request, jsonify
from Models import db, VehicleIdentity, VehicleObservation, AuditLog
from Libs.identity_manager import get_identity_manager
from Libs.config import VEHICLE_UI_PER_PAGE, OBSERVATIONS_PER_PAGE

vehicle_api = Blueprint('vehicle_api', __name__, url_prefix='/api')


# ========================================
# IDENTITY ENDPOINTS
# ========================================

@vehicle_api.route('/identities', methods=['GET'])
def get_identities():
    """Get paginated list of identities"""
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', VEHICLE_UI_PER_PAGE, type=int)
    search = request.args.get('search', '').strip()
    status = request.args.get('status', 'all')
    method = request.args.get('method', 'all')
    
    query = VehicleIdentity.query
    
    if search:
        query = query.filter(VehicleIdentity.plate_text.ilike(f'%{search}%'))
    
    if status == 'verified':
        query = query.filter_by(verified=True)
    elif status == 'unverified':
        query = query.filter_by(verified=False)
    
    if method in ['plate', 'face', 'visual']:
        query = query.filter_by(identity_method=method)
    
    query = query.order_by(VehicleIdentity.last_seen.desc())
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'success': True,
        'data': [v.to_dict() for v in pagination.items],
        'pagination': {
            'page': pagination.page,
            'pages': pagination.pages,
            'total': pagination.total,
            'per_page': per_page,
            'has_next': pagination.has_next,
            'has_prev': pagination.has_prev
        }
    })


@vehicle_api.route('/identities/<int:id>', methods=['GET'])
def get_identity(id):
    """Get single identity with observations"""
    
    identity = VehicleIdentity.query.get(id)
    if not identity:
        return jsonify({'success': False, 'error': 'Identity not found'}), 404
    
    include_obs = request.args.get('include_observations', 'true').lower() == 'true'
    
    return jsonify({
        'success': True,
        'data': identity.to_dict(include_observations=include_obs)
    })


@vehicle_api.route('/identities/<int:id>/verify', methods=['POST'])
def verify_identity(id):
    """Mark identity as verified"""
    
    manager = get_identity_manager()
    result = manager.verify_identity(id, performed_by='api')
    
    if result['success']:
        return jsonify(result)
    return jsonify(result), 400


@vehicle_api.route('/identities/<int:id>/unverify', methods=['POST'])
def unverify_identity(id):
    """Mark identity as unverified"""
    
    manager = get_identity_manager()
    result = manager.unverify_identity(id, performed_by='api')
    
    if result['success']:
        return jsonify(result)
    return jsonify(result), 400


@vehicle_api.route('/identities/<int:id>/plate', methods=['PUT'])
def update_plate(id):
    """Update plate text manually"""
    
    data = request.get_json() or {}
    new_plate = data.get('plate_text', '')
    
    manager = get_identity_manager()
    result = manager.update_plate_text(id, new_plate, performed_by='api')
    
    if result['success']:
        return jsonify(result)
    return jsonify(result), 400


@vehicle_api.route('/identities/<int:id>', methods=['DELETE'])
def delete_identity(id):
    """Delete identity and all observations"""
    
    manager = get_identity_manager()
    result = manager.delete_identity(id, performed_by='api')
    
    if result['success']:
        return jsonify(result)
    return jsonify(result), 400


# ========================================
# MERGE & SPLIT
# ========================================

@vehicle_api.route('/identities/merge', methods=['POST'])
def merge_identities():
    """
    Merge multiple identities into one.
    
    Request body:
    {
        "primary_id": 1,
        "secondary_ids": [2, 3, 4]
    }
    """
    
    data = request.get_json() or {}
    primary_id = data.get('primary_id')
    secondary_ids = data.get('secondary_ids', [])
    
    if not primary_id:
        return jsonify({'success': False, 'error': 'primary_id required'}), 400
    
    if not secondary_ids:
        return jsonify({'success': False, 'error': 'secondary_ids required'}), 400
    
    manager = get_identity_manager()
    result = manager.merge_identities(primary_id, secondary_ids, performed_by='api')
    
    if result['success']:
        return jsonify(result)
    return jsonify(result), 400


@vehicle_api.route('/identities/split', methods=['POST'])
def split_identity():
    """
    Split observations from identity to new identity.
    
    Request body:
    {
        "identity_id": 1,
        "observation_ids": [5, 6, 7]
    }
    """
    
    data = request.get_json() or {}
    identity_id = data.get('identity_id')
    observation_ids = data.get('observation_ids', [])
    
    if not identity_id:
        return jsonify({'success': False, 'error': 'identity_id required'}), 400
    
    if not observation_ids:
        return jsonify({'success': False, 'error': 'observation_ids required'}), 400
    
    manager = get_identity_manager()
    result = manager.split_identity(identity_id, observation_ids, performed_by='api')
    
    if result['success']:
        return jsonify(result)
    return jsonify(result), 400


# ========================================
# OBSERVATION ENDPOINTS
# ========================================

@vehicle_api.route('/observations', methods=['GET'])
def get_observations():
    """Get paginated list of observations"""
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', OBSERVATIONS_PER_PAGE, type=int)
    identity_id = request.args.get('identity_id', type=int)
    source = request.args.get('source', 'all')
    ocr = request.args.get('ocr', 'all')
    
    query = VehicleObservation.query
    
    if identity_id:
        query = query.filter_by(vehicle_id=identity_id)
    
    if source in ['image', 'video', 'webcam', 'ipcam']:
        query = query.filter_by(source_type=source)
    
    if ocr == 'success':
        query = query.filter_by(ocr_success=True)
    elif ocr == 'failed':
        query = query.filter_by(ocr_success=False)
    
    query = query.order_by(VehicleObservation.timestamp.desc())
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'success': True,
        'data': [o.to_dict() for o in pagination.items],
        'pagination': {
            'page': pagination.page,
            'pages': pagination.pages,
            'total': pagination.total,
            'per_page': per_page,
            'has_next': pagination.has_next,
            'has_prev': pagination.has_prev
        }
    })


@vehicle_api.route('/observations/<int:id>', methods=['GET'])
def get_observation(id):
    """Get single observation detail"""
    
    observation = VehicleObservation.query.get(id)
    if not observation:
        return jsonify({'success': False, 'error': 'Observation not found'}), 404
    
    return jsonify({
        'success': True,
        'data': observation.to_dict()
    })


@vehicle_api.route('/observations/<int:id>', methods=['DELETE'])
def delete_observation(id):
    """Delete single observation"""
    
    observation = VehicleObservation.query.get(id)
    if not observation:
        return jsonify({'success': False, 'error': 'Observation not found'}), 404
    
    identity_id = observation.vehicle_id
    
    # Update identity count
    identity = observation.identity
    if identity:
        identity.detection_count = max(0, identity.detection_count - 1)
    
    # Log action
    audit = AuditLog(
        action='delete',
        entity_type='observation',
        entity_id=id,
        performed_by='api'
    )
    db.session.add(audit)
    
    db.session.delete(observation)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'deleted_observation_id': id,
        'identity_id': identity_id
    })


# ========================================
# STATISTICS
# ========================================

@vehicle_api.route('/stats', methods=['GET'])
def get_stats():
    """Get overall statistics"""
    
    from sqlalchemy import func
    
    total_identities = VehicleIdentity.query.count()
    verified_count = VehicleIdentity.query.filter_by(verified=True).count()
    total_observations = VehicleObservation.query.count()
    
    # By identity method
    method_stats = db.session.query(
        VehicleIdentity.identity_method,
        func.count(VehicleIdentity.id)
    ).group_by(VehicleIdentity.identity_method).all()
    
    # By vehicle type
    type_stats = db.session.query(
        VehicleIdentity.vehicle_type,
        func.count(VehicleIdentity.id)
    ).group_by(VehicleIdentity.vehicle_type).all()
    
    # OCR success rate
    ocr_attempted = VehicleObservation.query.filter_by(ocr_attempted=True).count()
    ocr_success = VehicleObservation.query.filter_by(ocr_success=True).count()
    
    return jsonify({
        'success': True,
        'data': {
            'total_identities': total_identities,
            'verified_identities': verified_count,
            'unverified_identities': total_identities - verified_count,
            'total_observations': total_observations,
            'identity_methods': {m: c for m, c in method_stats},
            'vehicle_types': {t or 'unknown': c for t, c in type_stats},
            'ocr_stats': {
                'attempted': ocr_attempted,
                'success': ocr_success,
                'success_rate': round(ocr_success / ocr_attempted * 100, 2) if ocr_attempted > 0 else 0
            }
        }
    })


# ========================================
# AUDIT LOG
# ========================================

@vehicle_api.route('/audit', methods=['GET'])
def get_audit_log():
    """Get audit log entries"""
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    action = request.args.get('action', '')
    
    query = AuditLog.query
    
    if action:
        query = query.filter_by(action=action)
    
    query = query.order_by(AuditLog.timestamp.desc())
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'success': True,
        'data': [a.to_dict() for a in pagination.items],
        'pagination': {
            'page': pagination.page,
            'pages': pagination.pages,
            'total': pagination.total
        }
    })
