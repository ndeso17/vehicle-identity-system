from datetime import datetime
from . import db
import json

class VehicleGroup(db.Model):
    __tablename__ = 'vehicle_groups'

    id = db.Column(db.Integer, primary_key=True)
    primary_plate = db.Column(db.String(20), nullable=True)  # Primary OCR result
    confidence = db.Column(db.Float, default=0.0)            # Aggregate confidence
    vehicle_type = db.Column(db.String(50), nullable=True)   # car, motorcycle, etc.
    color = db.Column(db.String(50), nullable=True)          # Dominant color
    is_verified = db.Column(db.Boolean, default=False)       # Manually verified by admin
    needs_review = db.Column(db.Boolean, default=False)      # Flagged for manual review
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    observations = db.relationship('Observation', backref='group', lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            'id': self.id,
            'primary_plate': self.primary_plate or "Unknown",
            'confidence': self.confidence,
            'vehicle_type': self.vehicle_type,
            'color': self.color,
            'is_verified': self.is_verified,
            'needs_review': self.needs_review,
            'observation_count': len(self.observations),
            'last_seen': self.updated_at.isoformat() if self.updated_at else None
        }

class Observation(db.Model):
    __tablename__ = 'observations'

    id = db.Column(db.Integer, primary_key=True)
    group_id = db.Column(db.Integer, db.ForeignKey('vehicle_groups.id'), nullable=True)
    
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    image_path = db.Column(db.String(255), nullable=False)   # Path to snapshot
    
    plate_text = db.Column(db.String(20), nullable=True)
    plate_confidence = db.Column(db.Float, default=0.0)
    
    # Store face encoding as JSON list of floats
    face_encoding = db.Column(db.Text, nullable=True) 
    
    # Store bounding boxes as JSON: {vehicle: [], plate: [], driver: []}
    bboxes = db.Column(db.Text, nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'group_id': self.group_id,
            'timestamp': self.timestamp.isoformat(),
            'image_path': self.image_path,
            'plate_text': self.plate_text,
            'plate_confidence': self.plate_confidence,
            'has_face': self.face_encoding is not None,
            'bboxes': json.loads(self.bboxes) if self.bboxes else {}
        }
