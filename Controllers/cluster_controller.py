from flask import jsonify, request
from Libs.identity_registry import get_registry
import datetime


class ClusterController:
    """
    Controller for vehicle cluster management operations.
    Handles merge, split, and manual verification actions.
    """

    @staticmethod
    def get_all_clusters():
        """
        GET /api/clusters
        Returns paginated list of all vehicle clusters with metadata.
        
        Query params:
        - page: page number (default: 1)
        - per_page: items per page (default: 48)
        - search: filter by plate text
        - vehicle_type: filter by type (car, motorcycle, etc.)
        - min_appearances: minimum detection count
        """
        try:
            registry = get_registry()
            
            # Get query parameters
            page = int(request.args.get('page', 1))
            per_page = int(request.args.get('per_page', 48))
            search = request.args.get('search', '').strip()
            vehicle_type = request.args.get('vehicle_type', '').strip()
            min_appearances = int(request.args.get('min_appearances', 0))
            
            # Get all clusters
            all_clusters = list(registry._data.values())
            
            # Apply filters
            filtered = []
            for cluster in all_clusters:
                # Search filter (plate text)
                if search:
                    final_plate = cluster.get('final_plate', '') or ''
                    if search.upper() not in final_plate.upper():
                        continue
                
                # Vehicle type filter
                if vehicle_type:
                    if cluster.get('vehicle_type', '').lower() != vehicle_type.lower():
                        continue
                
                # Min appearances filter
                if min_appearances > 0:
                    if cluster.get('detection_count', 0) < min_appearances:
                        continue
                
                filtered.append(cluster)
            
            # Sort by last_seen (most recent first)
            filtered.sort(key=lambda x: x.get('last_seen', ''), reverse=True)
            
            # Pagination
            total = len(filtered)
            start = (page - 1) * per_page
            end = start + per_page
            page_clusters = filtered[start:end]
            
            # Prepare response
            summaries = []
            for c in page_clusters:
                summaries.append({
                    'cluster_id': c.get('cluster_id'),
                    'vehicle_type': c.get('vehicle_type'),
                    'dominant_color': c.get('dominant_color'),
                    'final_plate': c.get('final_plate'),
                    'plate_confidence': c.get('plate_confidence'),
                    'detection_count': c.get('detection_count', 0),
                    'first_seen': c.get('first_seen'),
                    'last_seen': c.get('last_seen'),
                    'representative_image': c.get('representative_image'),
                    'ocr_success_count': len(c.get('ocr_successes', [])),
                    'ocr_failure_count': len(c.get('ocr_failures', []))
                })
            
            return jsonify({
                'success': True,
                'clusters': summaries,
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': total,
                    'pages': (total + per_page - 1) // per_page
                }
            })
        
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @staticmethod
    def get_cluster_detail(cluster_id):
        """
        GET /api/clusters/<cluster_id>
        Returns detailed information about a specific cluster.
        """
        try:
            registry = get_registry()
            cluster = registry._data.get(cluster_id)
            
            if not cluster:
                return jsonify({
                    'success': False,
                    'error': 'Cluster not found'
                }), 404
            
            # Return full cluster data
            return jsonify({
                'success': True,
                'cluster': cluster
            })
        
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @staticmethod
    def merge_clusters():
        """
        POST /api/clusters/merge
        Merge multiple clusters into one.
        
        Request body:
        {
            "source_cluster_ids": ["veh_abc123", "veh_xyz789"],
            "target_cluster_id": "veh_abc123" (optional),
            "reason": "manual_merge" (optional)
        }
        """
        try:
            data = request.get_json()
            source_cluster_ids = data.get('source_cluster_ids', [])
            target_cluster_id = data.get('target_cluster_id')
            reason = data.get('reason', 'manual_merge')
            
            if not source_cluster_ids or len(source_cluster_ids) < 2:
                return jsonify({
                    'success': False,
                    'error': 'At least 2 clusters required for merge'
                }), 400
            
            registry = get_registry()
            result_cluster_id = registry.merge_clusters(
                source_cluster_ids, 
                target_cluster_id, 
                reason
            )
            
            if result_cluster_id:
                # Log manual correction
                _log_manual_correction('merge', {
                    'source_clusters': source_cluster_ids,
                    'target_cluster': result_cluster_id,
                    'reason': reason
                })
                
                return jsonify({
                    'success': True,
                    'result_cluster_id': result_cluster_id,
                    'message': f'Successfully merged {len(source_cluster_ids)} clusters'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Merge operation failed'
                }), 500
        
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @staticmethod
    def split_cluster():
        """
        POST /api/clusters/split
        Split a cluster by moving selected detections to a new cluster.
        
        Request body:
        {
            "cluster_id": "veh_abc123",
            "detection_ids": ["det_001", "det_002"],
            "reason": "manual_split" (optional)
        }
        """
        try:
            data = request.get_json()
            cluster_id = data.get('cluster_id')
            detection_ids = data.get('detection_ids', [])
            reason = data.get('reason', 'manual_split')
            
            if not cluster_id or not detection_ids:
                return jsonify({
                    'success': False,
                    'error': 'cluster_id and detection_ids required'
                }), 400
            
            registry = get_registry()
            new_cluster_id = registry.split_cluster(cluster_id, detection_ids, reason)
            
            if new_cluster_id:
                # Log manual correction
                _log_manual_correction('split', {
                    'original_cluster': cluster_id,
                    'new_cluster': new_cluster_id,
                    'detection_count': len(detection_ids),
                    'reason': reason
                })
                
                return jsonify({
                    'success': True,
                    'original_cluster_id': cluster_id,
                    'new_cluster_id': new_cluster_id,
                    'message': f'Successfully split cluster into 2 clusters'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Split operation failed'
                }), 500
        
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @staticmethod
    def mark_same():
        """
        POST /api/clusters/mark_same
        Mark two clusters as the same vehicle (will merge them).
        
        Request body:
        {
            "cluster_id_a": "veh_abc123",
            "cluster_id_b": "veh_xyz789"
        }
        """
        try:
            data = request.get_json()
            cluster_id_a = data.get('cluster_id_a')
            cluster_id_b = data.get('cluster_id_b')
            
            if not cluster_id_a or not cluster_id_b:
                return jsonify({
                    'success': False,
                    'error': 'Both cluster_id_a and cluster_id_b required'
                }), 400
            
            registry = get_registry()
            result_cluster_id = registry.mark_same_vehicle(cluster_id_a, cluster_id_b)
            
            if result_cluster_id:
                # Log manual correction
                _log_manual_correction('mark_same', {
                    'cluster_a': cluster_id_a,
                    'cluster_b': cluster_id_b,
                    'result_cluster': result_cluster_id
                })
                
                return jsonify({
                    'success': True,
                    'result_cluster_id': result_cluster_id,
                    'message': 'Successfully marked as same vehicle and merged'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Operation failed'
                }), 500
        
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @staticmethod
    def mark_different():
        """
        POST /api/clusters/mark_different
        Mark two clusters as different vehicles (prevents future automatic merging).
        
        Request body:
        {
            "cluster_id_a": "veh_abc123",
            "cluster_id_b": "veh_xyz789"
        }
        """
        try:
            data = request.get_json()
            cluster_id_a = data.get('cluster_id_a')
            cluster_id_b = data.get('cluster_id_b')
            
            if not cluster_id_a or not cluster_id_b:
                return jsonify({
                    'success': False,
                    'error': 'Both cluster_id_a and cluster_id_b required'
                }), 400
            
            registry = get_registry()
            success = registry.mark_different_vehicle(cluster_id_a, cluster_id_b)
            
            if success:
                # Log manual correction
                _log_manual_correction('mark_different', {
                    'cluster_a': cluster_id_a,
                    'cluster_b': cluster_id_b
                })
                
                return jsonify({
                    'success': True,
                    'message': 'Successfully marked as different vehicles'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Operation failed'
                }), 500
        
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500


def _log_manual_correction(action_type, data):
    """
    Helper function to log manual corrections to JSONL file.
    """
    try:
        from Libs.config import MANUAL_CORRECTIONS_PATH
        import os
        import json
        
        os.makedirs(os.path.dirname(MANUAL_CORRECTIONS_PATH), exist_ok=True)
        
        log_entry = {
            'action_id': f"action_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            'action_type': action_type,
            'timestamp': datetime.datetime.now().isoformat(),
            'user': 'manual',  # Could be extended to track actual user
            'data': data
        }
        
        with open(MANUAL_CORRECTIONS_PATH, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    except Exception as e:
        print(f"[CLUSTER_CONTROLLER] Failed to log manual correction: {e}")
