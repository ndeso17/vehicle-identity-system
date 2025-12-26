from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class VehicleIdentity(db.Model):
    """
    Represents a unique vehicle identity (cluster).
    
    Identity is determined by priority:
    1. plate_text (OCR confidence >= threshold)
    2. face_embedding (driver face cosine similarity)
    3. visual_hash (vehicle type + color + size)
    """
    __tablename__ = 'vehicle_identity'

    id = db.Column(db.Integer, primary_key=True)
    
    # Core Identity Feature (Priority 1: Plate)
    plate_text = db.Column(db.String(20), nullable=True, index=True)
    plate_confidence = db.Column(db.Float, default=0.0)
    
    # Secondary Identity (Priority 2: Face Embedding)
    # Stored as JSON string of list of floats (128D or 1024D)
    face_embedding = db.Column(db.Text, nullable=True)
    
    # Tertiary Identity (Priority 3: Visual Features)
    visual_hash = db.Column(db.String(64), nullable=True)  # Hash of type+color+size
    vehicle_type = db.Column(db.String(50), nullable=True)  # car, motorcycle, bus, truck
    vehicle_color = db.Column(db.String(50), nullable=True)  # Dominant color
    
    # Identity Method Tracking
    identity_method = db.Column(db.String(20), default='visual')  # 'plate', 'face', 'visual'
    
    # Timestamps
    first_seen = db.Column(db.DateTime, default=datetime.utcnow)
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)
    detection_count = db.Column(db.Integer, default=1)
    
    # Verification Status
    verified = db.Column(db.Boolean, default=False)
    verified_at = db.Column(db.DateTime, nullable=True)
    verified_by = db.Column(db.String(100), nullable=True)  # Future: user tracking
    
    # Presentation
    representative_image = db.Column(db.String(255), nullable=True)  # Path relative to static/
    
    # Merge/Split History (JSON array of actions)
    # Format: [{"action": "merged_from", "identity_id": 5, "at": "ISO timestamp"}, ...]
    merge_history = db.Column(db.Text, nullable=True)
    
    # Relationships
    observations = db.relationship('VehicleObservation', backref='identity', lazy='dynamic', cascade="all, delete-orphan")

    def to_dict(self, include_observations=False):
        result = {
            'id': self.id,
            'plate_text': self.plate_text,
            'plate_confidence': self.plate_confidence,
            'vehicle_type': self.vehicle_type,
            'vehicle_color': self.vehicle_color,
            'identity_method': self.identity_method,
            'first_seen': self.first_seen.isoformat() if self.first_seen else None,
            'last_seen': self.last_seen.isoformat() if self.last_seen else None,
            'verified': self.verified,
            'verified_at': self.verified_at.isoformat() if self.verified_at else None,
            'representative_image': self.representative_image,
            'detection_count': self.detection_count,
            'observation_count': self.observations.count(),
            'has_face_data': self.face_embedding is not None,
            'merge_history': json.loads(self.merge_history) if self.merge_history else []
        }
        
        if include_observations:
            result['observations'] = [obs.to_dict() for obs in self.observations.order_by(VehicleObservation.timestamp.desc()).all()]
        
        return result
    
    def add_merge_history(self, action, related_id):
        """Add entry to merge history"""
        history = json.loads(self.merge_history) if self.merge_history else []
        history.append({
            'action': action,
            'identity_id': related_id,
            'at': datetime.utcnow().isoformat()
        })
        self.merge_history = json.dumps(history)

    def get_face_embedding_array(self):
        """Return face embedding as numpy-compatible list"""
        if self.face_embedding:
            try:
                return json.loads(self.face_embedding)
            except:
                return None
        return None


class VehicleObservation(db.Model):
    """
    Represents a single detection/observation of a vehicle.
    Multiple observations can belong to one VehicleIdentity.
    """
    __tablename__ = 'vehicle_observation'

    id = db.Column(db.Integer, primary_key=True)
    vehicle_id = db.Column(db.Integer, db.ForeignKey('vehicle_identity.id'), nullable=False)
    
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    source_type = db.Column(db.String(20))  # 'video', 'image', 'webcam', 'ipcam'
    source_path = db.Column(db.String(500), nullable=True)  # Original source file/URL
    
    # Snapshot of features at this moment
    plate_text = db.Column(db.String(20), nullable=True)
    plate_confidence = db.Column(db.Float, default=0.0)
    
    # OCR Status (separate from result)
    ocr_attempted = db.Column(db.Boolean, default=True)
    ocr_success = db.Column(db.Boolean, default=False)
    
    vehicle_type = db.Column(db.String(50), nullable=True)
    vehicle_color = db.Column(db.String(50), nullable=True)
    
    # Driver Detection
    driver_detected = db.Column(db.Boolean, default=False)
    face_detected = db.Column(db.Boolean, default=False)
    
    # Images (paths relative to static/)
    frame_image_path = db.Column(db.String(255), nullable=True)  # Full original frame
    image_path = db.Column(db.String(255), nullable=True)  # Vehicle crop
    plate_image_path = db.Column(db.String(255), nullable=True)  # Plate crop
    driver_image_path = db.Column(db.String(255), nullable=True)  # Driver crop
    annotated_image_path = db.Column(db.String(255), nullable=True)  # Frame with bboxes
    
    # Bounding Boxes (JSON stored as text)
    # Format: {"vehicle": [x,y,w,h], "plate": [x,y,w,h], "driver": [x,y,w,h]}
    bbox_data = db.Column(db.Text, nullable=True) 
    
    def to_dict(self):
        return {
            'id': self.id,
            'vehicle_id': self.vehicle_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'source_type': self.source_type,
            'source_path': self.source_path,
            'plate_text': self.plate_text,
            'plate_confidence': self.plate_confidence,
            'ocr_attempted': self.ocr_attempted,
            'ocr_success': self.ocr_success,
            'vehicle_type': self.vehicle_type,
            'vehicle_color': self.vehicle_color,
            'driver_detected': self.driver_detected,
            'face_detected': self.face_detected,
            'frame_image_path': self.frame_image_path,
            'image_path': self.image_path,
            'plate_image_path': self.plate_image_path,
            'driver_image_path': self.driver_image_path,
            'annotated_image_path': self.annotated_image_path,
            'bbox_data': json.loads(self.bbox_data) if self.bbox_data else {}
        }
    
    def get_bboxes(self):
        """Return bounding boxes as dict"""
        if self.bbox_data:
            try:
                return json.loads(self.bbox_data)
            except:
                return {}
        return {}


class AuditLog(db.Model):
    """
    Tracks all manual corrections and system actions.
    """
    __tablename__ = 'audit_log'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    action = db.Column(db.String(50))  # 'verify', 'merge', 'split', 'edit', 'delete'
    entity_type = db.Column(db.String(50))  # 'identity', 'observation'
    entity_id = db.Column(db.Integer)
    
    # Action details (JSON)
    details = db.Column(db.Text, nullable=True)
    
    # Who performed the action
    performed_by = db.Column(db.String(100), default='system')
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'action': self.action,
            'entity_type': self.entity_type,
            'entity_id': self.entity_id,
            'details': json.loads(self.details) if self.details else {},
            'performed_by': self.performed_by
        }
