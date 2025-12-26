import os
import json
import uuid
import numpy as np
from collections import defaultdict, Counter
from .config import REGISTRY_PATH, PLATE_PRIMARY_CONF, FACE_SIM_THRESHOLD, FACE_EMBEDDING_METHOD, WEIGHT_PLATE, WEIGHT_FACE, WEIGHT_TYPE, WEIGHT_COLOR, WEIGHT_TIME


def normalize_plate_text(text: str) -> str:
    if not text:
        return ""
    # Uppercase, remove spaces, non-alphanumeric
    import re
    t = text.upper()
    t = re.sub(r"\s+", "", t)
    t = re.sub(r"[^A-Z0-9]", "", t)
    return t


def _cosine_sim(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _compute_appearance_embedding(image):
    """
    Fallback lightweight embedding: resize grayscale crop to small size and flatten normalized.
    Returns 128-D vector (or smaller) depending on crop.
    """
    import cv2
    if image is None or image.size == 0:
        return []
    try:
        small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
        if len(small.shape) == 3:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        else:
            gray = small
        vec = gray.flatten().astype(float)
        # normalize
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec.tolist()
        return (vec / norm).tolist()
    except Exception:
        return []


def _extract_face_embedding(face_crop):
    """
    Try to extract a face embedding. If `face_recognition` is available use it,
    otherwise fall back to appearance embedding.
    """
    try:
        import face_recognition
        # face_recognition expects RGB
        import cv2
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb)
        if encs:
            return encs[0].tolist()
    except Exception:
        pass
    return _compute_appearance_embedding(face_crop)


class VehicleRegistry:
    def __init__(self, path=REGISTRY_PATH):
        self.path = path
        self._data = {}  # vehicle_id -> record (cluster)
        self._load()

    def _load(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            if os.path.exists(self.path):
                with open(self.path, 'r') as fh:
                    self._data = json.load(fh)
            else:
                self._data = {}
        except Exception:
            self._data = {}

    def _save(self):
        try:
            with open(self.path, 'w') as fh:
                json.dump(self._data, fh)
        except Exception as e:
            print(f"[REGISTRY] Failed to save registry: {e}")

    def _new_vehicle_id(self):
        return f"veh_{str(uuid.uuid4())[:8]}"

    def _ensure_cluster(self, vid):
        rec = self._data.setdefault(vid, {})
        rec.setdefault('cluster_id', vid)
        rec.setdefault('plate_candidates', [])
        rec.setdefault('final_plate', None)
        rec.setdefault('plate_confidence', 0.0)
        rec.setdefault('face_embeddings', [])
        rec.setdefault('vehicle_type', None)
        rec.setdefault('dominant_color', None)
        rec.setdefault('representative_bbox', None)
        rec.setdefault('first_seen', None)
        rec.setdefault('last_seen', None)
        rec.setdefault('detection_history', [])
        rec.setdefault('manual_corrections', [])
        
        # NEW FIELDS for enhanced clustering
        rec.setdefault('representative_image', None)  # base64 thumbnail
        rec.setdefault('appearance_features', {})     # color histogram, etc.
        rec.setdefault('ocr_failures', [])            # failed OCR attempts
        rec.setdefault('ocr_successes', [])           # successful OCR attempts
        rec.setdefault('merge_history', [])           # track merges
        rec.setdefault('split_history', [])           # track splits
        rec.setdefault('detection_count', 0)          # total appearances
        rec.setdefault('blacklist_pairs', [])         # clusters marked as different
        
        return rec

    def _record_raw_detection(self, vid, detection):
        rec = self._ensure_cluster(vid)
        rec['detection_history'].append(detection)
        # update first/last seen
        ts = detection.get('timestamp')
        if ts:
            if rec['first_seen'] is None:
                rec['first_seen'] = ts
            rec['last_seen'] = ts

    def _compute_time_score(self, rec, timestamp):
        # simple recency score: 1.0 if last_seen within 10 minutes, else decays
        try:
            import datetime as _dt
            if rec.get('last_seen') is None:
                return 0.0
            last = _dt.datetime.fromisoformat(rec['last_seen'])
            cur = _dt.datetime.fromisoformat(timestamp)
            diff = abs((cur - last).total_seconds())
            if diff <= 600:
                return 1.0
            if diff <= 3600:
                return 0.5
            return 0.0
        except Exception:
            return 0.0

    def _score_cluster(self, rec, plate_text, plate_conf, face_emb, vehicle_type, color, timestamp):
        # plate score
        plate_score = 0.0
        if plate_text:
            if rec.get('final_plate') == plate_text:
                plate_score = plate_conf
            else:
                # check candidates
                for c in rec.get('plate_candidates', []):
                    if normalize_plate_text(c.get('text', '')) == plate_text:
                        plate_score = max(plate_score, float(c.get('confidence', 0.0)))

        # face score: best similarity among embeddings
        face_score = 0.0
        if face_emb and rec.get('face_embeddings'):
            for e in rec['face_embeddings']:
                s = _cosine_sim(e, face_emb)
                if s > face_score:
                    face_score = s

        # type score
        type_score = 1.0 if (rec.get('vehicle_type') and vehicle_type and rec.get('vehicle_type') == vehicle_type) else 0.0

        # color score
        color_score = 1.0 if (rec.get('dominant_color') and color and rec.get('dominant_color') == color) else 0.0

        # time score
        time_score = self._compute_time_score(rec, timestamp) if timestamp else 0.0

        # weighted sum
        total = WEIGHT_PLATE * plate_score + WEIGHT_FACE * face_score + WEIGHT_TYPE * type_score + WEIGHT_COLOR * color_score + WEIGHT_TIME * time_score
        max_possible = WEIGHT_PLATE + WEIGHT_FACE + WEIGHT_TYPE + WEIGHT_COLOR + WEIGHT_TIME
        normalized = total / max_possible if max_possible > 0 else 0.0
        return normalized

    def _append_embedding(self, vid, emb):
        if not emb:
            return
        rec = self._data.setdefault(vid, {})
        rec.setdefault('face_embeddings', [])
        rec['face_embeddings'].append(emb)

    def _append_plate_candidate(self, vid, plate_text, conf):
        rec = self._data.setdefault(vid, {})
        rec.setdefault('plate_candidates', [])
        rec['plate_candidates'].append({'text': plate_text, 'confidence': float(conf)})

    def _merge_plate_candidates(self, vid):
        rec = self._data.get(vid, {})
        cands = rec.get('plate_candidates', [])
        if not cands:
            return None, 0.0
        # Weighted vote by confidence
        scores = {}
        for c in cands:
            t = normalize_plate_text(c['text'])
            scores.setdefault(t, 0.0)
            scores[t] += float(c.get('confidence', 0.0))
        best = max(scores.items(), key=lambda x: x[1])
        # estimate average confidence
        texts = [c for c in cands if normalize_plate_text(c['text']) == best[0]]
        avg_conf = sum([c['confidence'] for c in texts]) / len(texts) if texts else 0.0
        rec['final_plate'] = best[0]
        rec['plate_confidence'] = float(avg_conf)
        return best[0], float(avg_conf)

    def match_by_plate(self, plate_text):
        t = normalize_plate_text(plate_text)
        for vid, rec in self._data.items():
            if rec.get('final_plate') == t:
                return vid
            for c in rec.get('plate_candidates', []):
                if normalize_plate_text(c.get('text', '')) == t:
                    return vid
        return None

    def match_by_face(self, emb):
        best_vid = None
        best_sim = 0.0
        for vid, rec in self._data.items():
            for e in rec.get('face_embeddings', []):
                sim = _cosine_sim(e, emb)
                if sim > best_sim:
                    best_sim = sim
                    best_vid = vid
        if best_sim >= FACE_SIM_THRESHOLD:
            return best_vid, best_sim
        return None, best_sim

    def match_best_cluster(self, plate_text, plate_conf, face_emb, vehicle_type, color, timestamp):
        """
        Score all existing clusters and return best match id and score.
        """
        best_vid = None
        best_score = 0.0
        for vid, rec in self._data.items():
            # ensure rec keys
            rec = self._ensure_cluster(vid)
            score = self._score_cluster(rec, plate_text, plate_conf, face_emb, vehicle_type, color, timestamp)
            if score > best_score:
                best_score = score
                best_vid = vid
        return best_vid, best_score

    def _compute_color_similarity(self, color1, color2):
        """
        Simple color name matching. Returns 1.0 for exact match, 0.0 otherwise.
        TODO: Implement HSV-based similarity for better matching.
        """
        if not color1 or not color2:
            return 0.0
        return 1.0 if color1.lower() == color2.lower() else 0.0

    def _update_representative_image(self, vid, vehicle_crop_b64, bbox, detection_id):
        """
        Update the representative image for a cluster if this detection is better quality.
        Currently uses simple heuristic: larger bbox = better quality.
        """
        rec = self._ensure_cluster(vid)
        if vehicle_crop_b64 and bbox:
            # Calculate bbox area
            area = bbox.get('w', 0) * bbox.get('h', 0)
            
            # Update if no representative or this one is larger
            current_bbox = rec.get('representative_bbox')
            if current_bbox is None:
                rec['representative_image'] = vehicle_crop_b64
                rec['representative_bbox'] = bbox
            else:
                current_area = current_bbox.get('w', 0) * current_bbox.get('h', 0)
                if area > current_area:
                    rec['representative_image'] = vehicle_crop_b64
                    rec['representative_bbox'] = bbox

    def merge_clusters(self, source_cluster_ids, target_cluster_id=None, reason="manual_merge"):
        """
        Merge multiple clusters into one.
        If target_cluster_id is None, use the first cluster as target.
        """
        if not source_cluster_ids or len(source_cluster_ids) < 2:
            return None
        
        if target_cluster_id is None:
            target_cluster_id = source_cluster_ids[0]
            source_cluster_ids = source_cluster_ids[1:]
        
        target = self._ensure_cluster(target_cluster_id)
        
        # Merge all data from source clusters
        for src_id in source_cluster_ids:
            if src_id == target_cluster_id:
                continue
            
            src = self._data.get(src_id)
            if not src:
                continue
            
            # Merge detection history
            target['detection_history'].extend(src.get('detection_history', []))
            
            # Merge plate candidates
            target['plate_candidates'].extend(src.get('plate_candidates', []))
            
            # Merge face embeddings
            target['face_embeddings'].extend(src.get('face_embeddings', []))
            
            # Merge OCR attempts
            target['ocr_failures'].extend(src.get('ocr_failures', []))
            target['ocr_successes'].extend(src.get('ocr_successes', []))
            
            # Update counts
            target['detection_count'] = target.get('detection_count', 0) + src.get('detection_count', 0)
            
            # Update timestamps
            if src.get('first_seen'):
                if not target.get('first_seen') or src['first_seen'] < target['first_seen']:
                    target['first_seen'] = src['first_seen']
            
            if src.get('last_seen'):
                if not target.get('last_seen') or src['last_seen'] > target['last_seen']:
                    target['last_seen'] = src['last_seen']
            
            # Log merge
            target['merge_history'].append({
                'merged_from': src_id,
                'timestamp': __import__('datetime').datetime.now().isoformat(),
                'reason': reason
            })
            
            # Remove source cluster
            del self._data[src_id]
        
        # Re-consolidate plate candidates
        self._merge_plate_candidates(target_cluster_id)
        
        # Save
        self._save()
        
        return target_cluster_id

    def split_cluster(self, cluster_id, detection_ids, reason="manual_split"):
        """
        Split a cluster by moving specified detections to a new cluster.
        """
        cluster = self._data.get(cluster_id)
        if not cluster:
            return None
        
        # Create new cluster
        new_vid = self._new_vehicle_id()
        new_cluster = self._ensure_cluster(new_vid)
        
        # Move specified detections
        remaining_detections = []
        moved_detections = []
        
        for det in cluster.get('detection_history', []):
            det_id = det.get('detection_id') or det.get('vehicle_id')
            if det_id in detection_ids:
                moved_detections.append(det)
            else:
                remaining_detections.append(det)
        
        cluster['detection_history'] = remaining_detections
        new_cluster['detection_history'] = moved_detections
        
        # Update counts
        cluster['detection_count'] = len(remaining_detections)
        new_cluster['detection_count'] = len(moved_detections)
        
        # Log split
        cluster['split_history'].append({
            'split_to': new_vid,
            'timestamp': __import__('datetime').datetime.now().isoformat(),
            'reason': reason,
            'detection_count': len(moved_detections)
        })
        
        new_cluster['split_history'].append({
            'split_from': cluster_id,
            'timestamp': __import__('datetime').datetime.now().isoformat(),
            'reason': reason
        })
        
        # Save
        self._save()
        
        return new_vid

    def mark_same_vehicle(self, cluster_id_a, cluster_id_b):
        """
        Mark two clusters as same vehicle (will merge them).
        """
        return self.merge_clusters([cluster_id_a, cluster_id_b], reason="marked_same")

    def mark_different_vehicle(self, cluster_id_a, cluster_id_b):
        """
        Mark two clusters as different vehicles (blacklist pairing).
        This prevents future automatic merging.
        """
        cluster_a = self._ensure_cluster(cluster_id_a)
        cluster_b = self._ensure_cluster(cluster_id_b)
        
        # Add to blacklist (stored as sorted tuple to avoid duplicates)
        pair = tuple(sorted([cluster_id_a, cluster_id_b]))
        
        if pair not in cluster_a.get('blacklist_pairs', []):
            cluster_a['blacklist_pairs'].append(pair)
        
        if pair not in cluster_b.get('blacklist_pairs', []):
            cluster_b['blacklist_pairs'].append(pair)
        
        self._save()
        return True


    def assign_groups(self, vehicles, frame=None, source_type='image'):
        """
        Given a list of processed vehicle dicts (as pipeline produced), assign
        `vehicle_group_id` and `grouping_method` for each vehicle and update registry.
        Modifies vehicles in-place and saves registry.
        
        ENHANCED: Now handles OCR failures gracefully and uses multi-feature matching.
        """
        from .config import CLUSTER_MATCH_THRESHOLD, PLATE_FALLBACK_CONF
        
        for v in vehicles:
            plate = v.get('plate', {})
            plate_text = normalize_plate_text(plate.get('text', ''))
            plate_conf = float(plate.get('confidence', 0.0)) if plate.get('confidence') is not None else 0.0
            plate_readable = plate.get('readable', False)

            assigned_vid = None
            method = 'new'
            face_id = None
            face_sim = 0.0

            # prepare features for scoring
            face_emb = None
            if v.get('driver') and v['driver'].get('present') and frame is not None:
                try:
                    db = v['driver']['bbox']
                    dx, dy, dw, dh = db['x'], db['y'], db['w'], db['h']
                    x1, y1, x2, y2 = int(dx), int(dy), int(dx+dw), int(dy+dh)
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        face_emb = _extract_face_embedding(face_crop)
                except Exception as e:
                    print(f"[REGISTRY] Failed to extract face embedding: {e}")
                    face_emb = None

            # 1. Plate primary match (only if high confidence)
            if plate_text and plate_conf >= PLATE_PRIMARY_CONF:
                vid = self.match_by_plate(plate_text)
                if vid:
                    assigned_vid = vid
                    method = 'plate_match'

            # 2. Probabilistic multi-feature matching (OCR-failure tolerant)
            if assigned_vid is None:
                best_vid, best_score = self.match_best_cluster(
                    plate_text, 
                    plate_conf, 
                    face_emb, 
                    v.get('vehicle_type'), 
                    v.get('vehicle_color'), 
                    v.get('timestamp')
                )
                
                # Use configurable threshold
                if best_vid and best_score >= CLUSTER_MATCH_THRESHOLD:
                    assigned_vid = best_vid
                    method = 'cluster_match'
                    
                    # Log why it matched
                    if not plate_text or plate_conf < PLATE_FALLBACK_CONF:
                        method = 'cluster_match_no_ocr'  # OCR failed, matched by appearance/face
                else:
                    # Create new cluster
                    assigned_vid = self._new_vehicle_id()
                    method = 'new'

            # Ensure cluster exists
            rec = self._ensure_cluster(assigned_vid)

            # 3. Track OCR attempts (success and failure)
            ocr_attempt = {
                'timestamp': v.get('timestamp'),
                'text': plate.get('text', ''),
                'confidence': plate_conf,
                'readable': plate_readable
            }
            
            if plate_readable and plate_text:
                rec['ocr_successes'].append(ocr_attempt)
                # Also add plate candidate
                self._append_plate_candidate(assigned_vid, plate_text, plate_conf)
            else:
                rec['ocr_failures'].append(ocr_attempt)

            # 4. Update face embeddings if available
            if face_emb:
                self._append_embedding(assigned_vid, face_emb)

            # 5. Update representative image
            vehicle_crop_b64 = v.get('crops', {}).get('vehicle')
            if vehicle_crop_b64:
                self._update_representative_image(
                    assigned_vid, 
                    vehicle_crop_b64, 
                    v.get('bbox'),
                    v.get('vehicle_id')
                )

            # 6. Update cluster metadata
            rec['vehicle_type'] = v.get('vehicle_type') or rec.get('vehicle_type')
            rec['dominant_color'] = v.get('vehicle_color') or rec.get('dominant_color')
            
            # Update timestamps
            ts = v.get('timestamp')
            if ts:
                if rec['first_seen'] is None:
                    rec['first_seen'] = ts
                rec['last_seen'] = ts

            # Increment detection count
            rec['detection_count'] = rec.get('detection_count', 0) + 1

            # Finalize merge of plate candidates
            self._merge_plate_candidates(assigned_vid)

            # Append full detection record to history
            rec['detection_history'].append(v)

            # Update appearances/timestamps (legacy support)
            rec.setdefault('appearances', []).append({
                'timestamp': v.get('timestamp'), 
                'source': source_type
            })
            rec.setdefault('source_types', [])
            if source_type not in rec['source_types']:
                rec['source_types'].append(source_type)

            # Attach grouping info to vehicle
            v['vehicle_group_id'] = assigned_vid
            v['grouping_method'] = method
            v['grouping_face_id'] = face_id
            v['grouping_face_similarity'] = float(face_sim)

        # save registry
        self._save()
        return vehicles

    def merge_clusters(self, source_ids, target_id=None, reason='manual_merge'):
        """
        Merge clusters in `source_ids` into `target_id` (if provided) or a new cluster.
        Records manual correction and returns target_id.
        """
        if not source_ids:
            return None
        if target_id is None:
            target_id = self._new_vehicle_id()
        self._ensure_cluster(target_id)
        for sid in source_ids:
            if sid == target_id:
                continue
            srec = self._data.get(sid)
            if not srec:
                continue
            # move plate candidates
            for c in srec.get('plate_candidates', []):
                self._append_plate_candidate(target_id, c.get('text', ''), c.get('confidence', 0.0))
            # move face embeddings
            for e in srec.get('face_embeddings', []):
                self._append_embedding(target_id, e)
            # move detection history
            for d in srec.get('detection_history', []):
                self._record_raw_detection(target_id, d)
            # note manual correction
            self._data[target_id].setdefault('manual_corrections', []).append({'action': 'merged_from', 'source': sid, 'reason': reason})
            # remove source cluster
            try:
                del self._data[sid]
            except Exception:
                pass
        # recompute final plate
        self._merge_plate_candidates(target_id)
        self._save()
        return target_id

    def split_cluster(self, cluster_id, keep_ids=None, reason='manual_split'):
        """
        Split cluster into multiple clusters. `keep_ids` is list of indices from detection_history to keep in original cluster;
        others will be moved into a new cluster. Simplified split: create new cluster with half of detections if keep_ids not provided.
        """
        rec = self._data.get(cluster_id)
        if not rec:
            return None
        history = rec.get('detection_history', [])
        if not history:
            return None
        if keep_ids is None:
            # split half
            k = len(history) // 2
            keep = history[:k]
            move = history[k:]
        else:
            keep = [history[i] for i in keep_ids if i < len(history)]
            move = [h for i, h in enumerate(history) if i not in (keep_ids or [])]

        # update original
        rec['detection_history'] = keep
        # create new cluster
        new_id = self._new_vehicle_id()
        self._ensure_cluster(new_id)
        for d in move:
            self._record_raw_detection(new_id, d)
        rec.setdefault('manual_corrections', []).append({'action': 'split_to', 'target': new_id, 'reason': reason})
        self._save()
        return new_id


_REGISTRY = None

def get_registry():
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = VehicleRegistry()
    return _REGISTRY
