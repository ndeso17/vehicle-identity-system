"""
Identity Manager - Core logic for vehicle identity matching and clustering.

This module handles:
1. Priority-based identity matching (Plate → Face → Visual)
2. Creating new identities
3. Creating observations
4. Merge/Split operations
5. Verification status management
"""

import os
import json
import uuid
import re
import datetime
import numpy as np
import cv2
from flask import current_app

from Models import db, VehicleIdentity, VehicleObservation, AuditLog
from .config import (
    PLATE_PRIMARY_CONF, PLATE_FALLBACK_CONF,
    FACE_SIM_THRESHOLD, VISUAL_MATCH_THRESHOLD,
    CLUSTER_MATCH_THRESHOLD,
    WEIGHT_PLATE, WEIGHT_FACE, WEIGHT_TYPE, WEIGHT_COLOR, WEIGHT_TIME,
    TIME_WINDOW_STRONG, TIME_WINDOW_WEAK,
    CROPS_FOLDER, FRAMES_FOLDER, ANNOTATED_FOLDER
)


class IdentityManager:
    """
    Manages vehicle identities and observations.
    
    Identity matching priority:
    1. Plate text (OCR confidence >= PLATE_PRIMARY_CONF)
    2. Face embedding (cosine similarity >= FACE_SIM_THRESHOLD)
    3. Visual features (type + color combination)
    """
    
    def __init__(self):
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create storage directories"""
        for folder in [CROPS_FOLDER, FRAMES_FOLDER, ANNOTATED_FOLDER]:
            os.makedirs(folder, exist_ok=True)
    
    # ========================================
    # IMAGE SAVING
    # ========================================
    
    def _save_image(self, image, folder, prefix="img"):
        """
        Save image to specified folder.
        
        Args:
            image: numpy array (BGR)
            folder: destination folder path
            prefix: filename prefix
            
        Returns:
            Relative path from static/ or None
        """
        if image is None or not hasattr(image, 'size') or image.size == 0:
            return None
        
        filename = f"{prefix}_{uuid.uuid4().hex[:12]}.jpg"
        full_path = os.path.join(folder, filename)
        
        try:
            cv2.imwrite(full_path, image)
            # Return path relative to static folder for web serving
            rel_path = os.path.relpath(full_path, os.path.dirname(CROPS_FOLDER))
            return rel_path
        except Exception as e:
            print(f"[IdentityManager] Failed to save image: {e}")
            return None
    
    def save_vehicle_crop(self, image):
        return self._save_image(image, CROPS_FOLDER, "veh")
    
    def save_plate_crop(self, image):
        return self._save_image(image, CROPS_FOLDER, "plate")
    
    def save_driver_crop(self, image):
        return self._save_image(image, CROPS_FOLDER, "driver")
    
    def save_frame(self, image):
        return self._save_image(image, FRAMES_FOLDER, "frame")
    
    def save_annotated(self, image):
        return self._save_image(image, ANNOTATED_FOLDER, "annotated")
    
    # ========================================
    # EMBEDDING & SIMILARITY
    # ========================================
    
    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors"""
        if not a or not b:
            return 0.0
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def extract_face_embedding(self, face_crop):
        """
        Extract face embedding from driver crop.
        
        Tries face_recognition library first, falls back to appearance hash.
        
        Args:
            face_crop: numpy array of face/driver region
            
        Returns:
            List of floats (embedding) or None
        """
        if face_crop is None or not hasattr(face_crop, 'size') or face_crop.size == 0:
            return None
        
        # Try face_recognition library
        try:
            import face_recognition
            rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb)
            if encodings:
                return encodings[0].tolist()
        except ImportError:
            pass
        except Exception:
            pass
        
        # Fallback: appearance hash (32x32 grayscale flattened, normalized)
        try:
            small = cv2.resize(face_crop, (32, 32), interpolation=cv2.INTER_AREA)
            if len(small.shape) == 3:
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            else:
                gray = small
            vec = gray.flatten().astype(float)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec.tolist()
        except Exception:
            return None
    
    def _compute_visual_hash(self, vehicle_type, vehicle_color):
        """Create a simple hash from vehicle type and color"""
        key = f"{vehicle_type or 'unknown'}_{vehicle_color or 'unknown'}"
        return key.lower()
    
    # ========================================
    # TIME SCORING
    # ========================================
    
    def _compute_time_score(self, last_seen):
        """
        Compute temporal proximity score.
        
        Args:
            last_seen: datetime object or ISO string
            
        Returns:
            Score 0.0 to 1.0
        """
        if not last_seen:
            return 0.0
        
        try:
            if isinstance(last_seen, str):
                last = datetime.datetime.fromisoformat(last_seen)
            else:
                last = last_seen
            
            now = datetime.datetime.utcnow()
            diff_seconds = abs((now - last).total_seconds())
            
            if diff_seconds <= TIME_WINDOW_STRONG:
                return 1.0
            elif diff_seconds <= TIME_WINDOW_WEAK:
                return 0.5
            return 0.0
        except Exception:
            return 0.0
    
    # ========================================
    # MAIN ENTRY POINT
    # ========================================
    
    def process_detection(self, 
                          plate_text,
                          plate_conf,
                          vehicle_crop,
                          plate_crop,
                          driver_crop,
                          frame_image,
                          annotated_image,
                          vehicle_type,
                          vehicle_color,
                          bbox_data,
                          source_type="image",
                          source_path=None):
        """
        Process a single vehicle detection and assign to identity.
        
        Args:
            plate_text: OCR result (may be None or empty)
            plate_conf: OCR confidence (0.0-1.0)
            vehicle_crop: numpy array of vehicle region
            plate_crop: numpy array of plate region
            driver_crop: numpy array of driver region
            frame_image: numpy array of full frame
            annotated_image: numpy array of annotated frame
            vehicle_type: detected vehicle type
            vehicle_color: detected vehicle color
            bbox_data: dict with vehicle/plate/driver bounding boxes
            source_type: 'image', 'video', 'webcam', 'ipcam'
            source_path: original source file/URL
            
        Returns:
            tuple: (identity_id, observation_id, is_new_identity)
        """
        
        # 1. Extract features
        face_embedding = self.extract_face_embedding(driver_crop)
        normalized_plate = self._normalize_plate(plate_text)
        visual_hash = self._compute_visual_hash(vehicle_type, vehicle_color)
        
        # 2. Determine OCR status
        ocr_attempted = True
        ocr_success = bool(normalized_plate and plate_conf >= PLATE_FALLBACK_CONF)
        
        # 3. Find matching identity
        identity_id, identity_method = self._find_matching_identity(
            normalized_plate, plate_conf, face_embedding, vehicle_type, vehicle_color
        )
        
        is_new_identity = False
        
        # 4. Create new identity if no match
        if identity_id is None:
            identity_id, identity_method = self._create_identity(
                normalized_plate, plate_conf, face_embedding, 
                vehicle_type, vehicle_color, visual_hash, vehicle_crop
            )
            is_new_identity = True
        
        # 5. Save images
        vehicle_image_path = self.save_vehicle_crop(vehicle_crop)
        plate_image_path = self.save_plate_crop(plate_crop)
        driver_image_path = self.save_driver_crop(driver_crop)
        frame_image_path = self.save_frame(frame_image)
        annotated_image_path = self.save_annotated(annotated_image)
        
        # 6. Create observation
        observation_id = self._create_observation(
            identity_id=identity_id,
            plate_text=normalized_plate,
            plate_conf=plate_conf,
            ocr_attempted=ocr_attempted,
            ocr_success=ocr_success,
            vehicle_type=vehicle_type,
            vehicle_color=vehicle_color,
            driver_detected=driver_crop is not None,
            face_detected=face_embedding is not None,
            frame_image_path=frame_image_path,
            image_path=vehicle_image_path,
            plate_image_path=plate_image_path,
            driver_image_path=driver_image_path,
            annotated_image_path=annotated_image_path,
            bbox_data=bbox_data,
            source_type=source_type,
            source_path=source_path
        )
        
        # 7. Update identity aggregates
        self._update_identity(identity_id, normalized_plate, plate_conf, 
                             face_embedding, vehicle_type, vehicle_color, vehicle_crop)
        
        return identity_id, observation_id, is_new_identity
    
    def _normalize_plate(self, plate_text):
        """Normalize plate text: uppercase, remove special chars"""
        if not plate_text:
            return None
        return re.sub(r"[^A-Z0-9]", "", plate_text.upper())
    
    # ========================================
    # IDENTITY MATCHING
    # ========================================
    
    def _find_matching_identity(self, plate_text, plate_conf, face_embedding, vehicle_type, vehicle_color):
        """
        Find matching identity using priority-based matching.
        
        Priority:
        1. Exact plate match (if confidence >= threshold)
        2. Face embedding similarity (if plate not available)
        3. Visual feature match (fallback)
        
        Returns:
            tuple: (identity_id or None, identity_method or None)
        """
        
        # Priority 1: High confidence plate match
        if plate_text and plate_conf >= PLATE_PRIMARY_CONF:
            identity = VehicleIdentity.query.filter_by(plate_text=plate_text).first()
            if identity:
                return identity.id, 'plate'
        
        # Priority 2 & 3: Score-based matching for all candidates
        candidates = VehicleIdentity.query.all()
        
        best_score = 0.0
        best_id = None
        best_method = None
        
        for candidate in candidates:
            score, method = self._calculate_match_score(
                candidate, plate_text, plate_conf, face_embedding, vehicle_type, vehicle_color
            )
            if score > best_score:
                best_score = score
                best_id = candidate.id
                best_method = method
        
        if best_score >= CLUSTER_MATCH_THRESHOLD:
            return best_id, best_method
        
        return None, None
    
    def _calculate_match_score(self, identity, plate_text, plate_conf, face_embedding, vehicle_type, vehicle_color):
        """
        Calculate weighted match score between detection and identity.
        
        Returns:
            tuple: (score 0.0-1.0, primary matching method)
        """
        scores = {
            'plate': 0.0,
            'face': 0.0,
            'type': 0.0,
            'color': 0.0,
            'time': 0.0
        }
        
        # Plate score
        if plate_text and identity.plate_text:
            if plate_text == identity.plate_text:
                scores['plate'] = plate_conf
        
        # Face score
        if face_embedding and identity.face_embedding:
            try:
                db_embedding = json.loads(identity.face_embedding)
                scores['face'] = self._cosine_similarity(face_embedding, db_embedding)
            except Exception:
                pass
        
        # Type score
        if vehicle_type and identity.vehicle_type:
            if vehicle_type.lower() == identity.vehicle_type.lower():
                scores['type'] = 1.0
        
        # Color score
        if vehicle_color and identity.vehicle_color:
            if vehicle_color.lower() == identity.vehicle_color.lower():
                scores['color'] = 1.0
        
        # Time score
        scores['time'] = self._compute_time_score(identity.last_seen)
        
        # Weighted sum
        total_score = (
            WEIGHT_PLATE * scores['plate'] +
            WEIGHT_FACE * scores['face'] +
            WEIGHT_TYPE * scores['type'] +
            WEIGHT_COLOR * scores['color'] +
            WEIGHT_TIME * scores['time']
        )
        
        max_possible = WEIGHT_PLATE + WEIGHT_FACE + WEIGHT_TYPE + WEIGHT_COLOR + WEIGHT_TIME
        normalized_score = total_score / max_possible if max_possible > 0 else 0.0
        
        # Determine primary method
        if scores['plate'] >= PLATE_PRIMARY_CONF:
            method = 'plate'
        elif scores['face'] >= FACE_SIM_THRESHOLD:
            method = 'face'
        else:
            method = 'visual'
        
        return normalized_score, method
    
    # ========================================
    # IDENTITY CRUD
    # ========================================
    
    def _create_identity(self, plate_text, plate_conf, face_embedding, 
                        vehicle_type, vehicle_color, visual_hash, vehicle_crop):
        """Create a new vehicle identity"""
        
        # Determine identity method
        if plate_text and plate_conf >= PLATE_PRIMARY_CONF:
            identity_method = 'plate'
        elif face_embedding:
            identity_method = 'face'
        else:
            identity_method = 'visual'
        
        identity = VehicleIdentity(
            plate_text=plate_text if plate_conf >= PLATE_FALLBACK_CONF else None,
            plate_confidence=plate_conf if plate_text else 0.0,
            face_embedding=json.dumps(face_embedding) if face_embedding else None,
            visual_hash=visual_hash,
            vehicle_type=vehicle_type,
            vehicle_color=vehicle_color,
            identity_method=identity_method,
            first_seen=datetime.datetime.utcnow(),
            last_seen=datetime.datetime.utcnow(),
            detection_count=0,
            verified=False
        )
        
        # Save representative image
        if vehicle_crop is not None:
            path = self.save_vehicle_crop(vehicle_crop)
            identity.representative_image = path
        
        db.session.add(identity)
        db.session.commit()
        
        return identity.id, identity_method
    
    def _create_observation(self, identity_id, plate_text, plate_conf,
                           ocr_attempted, ocr_success, vehicle_type, vehicle_color,
                           driver_detected, face_detected,
                           frame_image_path, image_path, plate_image_path, 
                           driver_image_path, annotated_image_path,
                           bbox_data, source_type, source_path):
        """Create a new observation for an identity"""
        
        observation = VehicleObservation(
            vehicle_id=identity_id,
            timestamp=datetime.datetime.utcnow(),
            source_type=source_type,
            source_path=source_path,
            plate_text=plate_text,
            plate_confidence=plate_conf,
            ocr_attempted=ocr_attempted,
            ocr_success=ocr_success,
            vehicle_type=vehicle_type,
            vehicle_color=vehicle_color,
            driver_detected=driver_detected,
            face_detected=face_detected,
            frame_image_path=frame_image_path,
            image_path=image_path,
            plate_image_path=plate_image_path,
            driver_image_path=driver_image_path,
            annotated_image_path=annotated_image_path,
            bbox_data=json.dumps(bbox_data) if bbox_data else None
        )
        
        db.session.add(observation)
        db.session.commit()
        
        return observation.id
    
    def _update_identity(self, identity_id, plate_text, plate_conf, 
                        face_embedding, vehicle_type, vehicle_color, vehicle_crop):
        """Update identity aggregates after new observation"""
        
        identity = VehicleIdentity.query.get(identity_id)
        if not identity:
            return
        
        identity.last_seen = datetime.datetime.utcnow()
        identity.detection_count += 1
        
        # Update plate if better confidence
        if plate_text and plate_conf > identity.plate_confidence:
            identity.plate_text = plate_text
            identity.plate_confidence = plate_conf
            if plate_conf >= PLATE_PRIMARY_CONF:
                identity.identity_method = 'plate'
        
        # Update face embedding if not present
        if face_embedding and not identity.face_embedding:
            identity.face_embedding = json.dumps(face_embedding)
            if identity.identity_method == 'visual':
                identity.identity_method = 'face'
        
        # Update vehicle info if not present
        if vehicle_type and not identity.vehicle_type:
            identity.vehicle_type = vehicle_type
        if vehicle_color and not identity.vehicle_color:
            identity.vehicle_color = vehicle_color
        
        # Update representative image if not present
        if not identity.representative_image and vehicle_crop is not None:
            path = self.save_vehicle_crop(vehicle_crop)
            identity.representative_image = path
        
        db.session.commit()
    
    # ========================================
    # MERGE & SPLIT OPERATIONS
    # ========================================
    
    def merge_identities(self, primary_id, secondary_ids, performed_by='system'):
        """
        Merge multiple identities into one.
        
        All observations from secondary identities are moved to primary.
        Secondary identities are deleted.
        
        Args:
            primary_id: ID of identity to keep
            secondary_ids: List of IDs to merge into primary
            performed_by: Who performed the action
            
        Returns:
            dict with result
        """
        primary = VehicleIdentity.query.get(primary_id)
        if not primary:
            return {'success': False, 'error': 'Primary identity not found'}
        
        merged_count = 0
        
        for sec_id in secondary_ids:
            if sec_id == primary_id:
                continue
            
            secondary = VehicleIdentity.query.get(sec_id)
            if not secondary:
                continue
            
            # Move all observations
            VehicleObservation.query.filter_by(vehicle_id=sec_id).update(
                {'vehicle_id': primary_id}
            )
            
            # Update primary counts
            primary.detection_count += secondary.detection_count
            
            # Take better plate if available
            if secondary.plate_confidence > primary.plate_confidence:
                primary.plate_text = secondary.plate_text
                primary.plate_confidence = secondary.plate_confidence
            
            # Take face embedding if primary doesn't have one
            if not primary.face_embedding and secondary.face_embedding:
                primary.face_embedding = secondary.face_embedding
            
            # Record merge history
            primary.add_merge_history('merged_from', sec_id)
            
            # Log the action
            audit = AuditLog(
                action='merge',
                entity_type='identity',
                entity_id=primary_id,
                details=json.dumps({'merged_from': sec_id}),
                performed_by=performed_by
            )
            db.session.add(audit)
            
            # Delete secondary
            db.session.delete(secondary)
            merged_count += 1
        
        # Require re-verification after merge
        primary.verified = False
        primary.verified_at = None
        
        db.session.commit()
        
        return {
            'success': True, 
            'merged_count': merged_count,
            'primary_id': primary_id,
            'new_observation_count': primary.observations.count()
        }
    
    def split_identity(self, identity_id, observation_ids, performed_by='system'):
        """
        Split observations from an identity to a new identity.
        
        Args:
            identity_id: ID of identity to split from
            observation_ids: List of observation IDs to move
            performed_by: Who performed the action
            
        Returns:
            dict with result including new identity ID
        """
        original = VehicleIdentity.query.get(identity_id)
        if not original:
            return {'success': False, 'error': 'Identity not found'}
        
        if not observation_ids:
            return {'success': False, 'error': 'No observations specified'}
        
        # Create new identity (copy basic info)
        new_identity = VehicleIdentity(
            plate_text=None,  # Will be determined from observations
            vehicle_type=original.vehicle_type,
            vehicle_color=original.vehicle_color,
            identity_method='visual',  # Start as visual, may upgrade
            first_seen=datetime.datetime.utcnow(),
            last_seen=datetime.datetime.utcnow(),
            detection_count=0,
            verified=False
        )
        db.session.add(new_identity)
        db.session.flush()  # Get the new ID
        
        # Move specified observations
        moved_count = 0
        for obs_id in observation_ids:
            obs = VehicleObservation.query.get(obs_id)
            if obs and obs.vehicle_id == identity_id:
                obs.vehicle_id = new_identity.id
                new_identity.detection_count += 1
                original.detection_count -= 1
                
                # Update plate if observation has better one
                if obs.plate_text and obs.plate_confidence > (new_identity.plate_confidence or 0):
                    new_identity.plate_text = obs.plate_text
                    new_identity.plate_confidence = obs.plate_confidence
                
                # Set representative image from first observation
                if not new_identity.representative_image and obs.image_path:
                    new_identity.representative_image = obs.image_path
                
                moved_count += 1
        
        # Record split history
        original.add_merge_history('split_to', new_identity.id)
        new_identity.add_merge_history('split_from', identity_id)
        
        # Require re-verification
        original.verified = False
        original.verified_at = None
        
        # Log the action
        audit = AuditLog(
            action='split',
            entity_type='identity',
            entity_id=identity_id,
            details=json.dumps({
                'new_identity_id': new_identity.id,
                'observation_ids': observation_ids
            }),
            performed_by=performed_by
        )
        db.session.add(audit)
        
        db.session.commit()
        
        return {
            'success': True,
            'new_identity_id': new_identity.id,
            'moved_observation_count': moved_count,
            'original_remaining_count': original.observations.count()
        }
    
    # ========================================
    # VERIFICATION
    # ========================================
    
    def verify_identity(self, identity_id, verified_by='system'):
        """Mark an identity as verified"""
        identity = VehicleIdentity.query.get(identity_id)
        if not identity:
            return {'success': False, 'error': 'Identity not found'}
        
        identity.verified = True
        identity.verified_at = datetime.datetime.utcnow()
        identity.verified_by = verified_by
        
        # Log the action
        audit = AuditLog(
            action='verify',
            entity_type='identity',
            entity_id=identity_id,
            performed_by=verified_by
        )
        db.session.add(audit)
        db.session.commit()
        
        return {'success': True, 'identity_id': identity_id}
    
    def unverify_identity(self, identity_id, performed_by='system'):
        """Mark an identity as unverified"""
        identity = VehicleIdentity.query.get(identity_id)
        if not identity:
            return {'success': False, 'error': 'Identity not found'}
        
        identity.verified = False
        identity.verified_at = None
        
        audit = AuditLog(
            action='unverify',
            entity_type='identity',
            entity_id=identity_id,
            performed_by=performed_by
        )
        db.session.add(audit)
        db.session.commit()
        
        return {'success': True, 'identity_id': identity_id}
    
    def update_plate_text(self, identity_id, new_plate_text, performed_by='system'):
        """Manually update plate text for an identity"""
        identity = VehicleIdentity.query.get(identity_id)
        if not identity:
            return {'success': False, 'error': 'Identity not found'}
        
        old_plate = identity.plate_text
        identity.plate_text = self._normalize_plate(new_plate_text)
        identity.plate_confidence = 1.0  # Manual = confident
        identity.identity_method = 'plate'
        identity.verified = False  # Require re-verification
        
        audit = AuditLog(
            action='edit',
            entity_type='identity',
            entity_id=identity_id,
            details=json.dumps({'old_plate': old_plate, 'new_plate': identity.plate_text}),
            performed_by=performed_by
        )
        db.session.add(audit)
        db.session.commit()
        
        return {'success': True, 'identity_id': identity_id}
    
    def delete_identity(self, identity_id, performed_by='system'):
        """Delete an identity and all its observations"""
        identity = VehicleIdentity.query.get(identity_id)
        if not identity:
            return {'success': False, 'error': 'Identity not found'}
        
        observation_count = identity.observations.count()
        
        audit = AuditLog(
            action='delete',
            entity_type='identity',
            entity_id=identity_id,
            details=json.dumps({'observation_count': observation_count}),
            performed_by=performed_by
        )
        db.session.add(audit)
        
        db.session.delete(identity)
        db.session.commit()
        
        return {'success': True, 'deleted_observations': observation_count}


# Singleton instance
_identity_manager = None

def get_identity_manager():
    """Get singleton IdentityManager instance"""
    global _identity_manager
    if _identity_manager is None:
        _identity_manager = IdentityManager()
    return _identity_manager
