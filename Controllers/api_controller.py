"""
API Controller - Handles image/video processing with identity management integration.
"""

import cv2
import numpy as np
import json
import base64
import os
import time
from flask import Response, jsonify, render_template, current_app
from Libs.pipeline import Pipeline
from Libs.identity_manager import get_identity_manager

# Initialize Pipeline (lazy loading)
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        try:
            pipeline = Pipeline()
        except Exception as e:
            print(f"[ApiController] Error initializing pipeline: {e}")
    return pipeline


class ApiController:
    @staticmethod
    def upload_image(request):
        """
        Process uploaded image through pipeline and persist to database.
        
        Flow:
        1. Decode image
        2. Run pipeline (detection + OCR)
        3. For each vehicle: call identity_manager.process_detection()
        4. Return rendered result page
        """
        pl = get_pipeline()
        if not pl:
            return jsonify({"error": "Pipeline failed to initialize"}), 500

        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Read and decode image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image file"}), 400

        # Process through pipeline
        try:
            result, annotated_image = pl.process_frame_with_visualization(
                image, source_type="image"
            )
        except Exception as e:
            print(f"[ApiController] Pipeline error: {e}")
            # Fallback to basic processing
            result = pl.process_frame(image, source_type="image")
            annotated_image = image.copy()

        # Persist to database
        identity_manager = get_identity_manager()
        identity_ids = []
        
        for vehicle in result.get('vehicles', []):
            try:
                # Extract crops from base64 if available
                vehicle_crop = None
                plate_crop = None
                driver_crop = None
                
                crops = vehicle.get('crops', {})
                if crops.get('vehicle'):
                    vehicle_crop = ApiController._decode_base64_image(crops['vehicle'])
                if crops.get('plate'):
                    plate_crop = ApiController._decode_base64_image(crops['plate'])
                if crops.get('driver'):
                    driver_crop = ApiController._decode_base64_image(crops['driver'])
                
                # Build bbox data
                bbox_data = {
                    'vehicle': vehicle.get('bbox'),
                    'plate': vehicle.get('plate', {}).get('bbox'),
                    'driver': vehicle.get('driver', {}).get('bbox') if vehicle.get('driver', {}).get('present') else None
                }
                
                # Process detection
                identity_id, observation_id, is_new = identity_manager.process_detection(
                    plate_text=vehicle.get('plate', {}).get('text'),
                    plate_conf=vehicle.get('plate', {}).get('confidence', 0.0),
                    vehicle_crop=vehicle_crop,
                    plate_crop=plate_crop,
                    driver_crop=driver_crop,
                    frame_image=image,
                    annotated_image=annotated_image,
                    vehicle_type=vehicle.get('vehicle_type'),
                    vehicle_color=vehicle.get('vehicle_color'),
                    bbox_data=bbox_data,
                    source_type="image",
                    source_path=file.filename
                )
                
                identity_ids.append({
                    'identity_id': identity_id,
                    'observation_id': observation_id,
                    'is_new': is_new
                })
                
            except Exception as e:
                print(f"[ApiController] Failed to process vehicle: {e}")
        
        # Encode annotated image to base64 for display
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        # Save annotated image
        try:
            os.makedirs('results/inference_img', exist_ok=True)
            fname = os.path.join('results', 'inference_img', f"inf_{int(time.time())}.jpg")
            cv2.imwrite(fname, annotated_image)
        except Exception as e:
            print(f"[ApiController] Failed to save inference image: {e}")
        
        # Add identity info to result
        result['identities'] = identity_ids
        json_output = json.dumps(result, indent=2)

        return render_template('result.html', 
                               original_image=img_str, 
                               result_json=json_output,
                               result_data=result,
                               identity_ids=identity_ids)

    @staticmethod
    def _decode_base64_image(b64_string):
        """Decode base64 string to numpy array"""
        if not b64_string:
            return None
        try:
            img_data = base64.b64decode(b64_string)
            nparr = np.frombuffer(img_data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    @staticmethod
    def stream_feed(source):
        """
        Generator for MJPEG stream with identity persistence.
        Source can be 'webcam' (int 0) or RTSP url (str).
        """
        pl = get_pipeline()
        identity_manager = get_identity_manager()
        cap = cv2.VideoCapture(source)
        
        frame_count = 0
        process_every_n = 3  # Process every N frames for performance
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            
            # Process every N frames
            if pl and frame_count % process_every_n == 0:
                try:
                    result, annotated = pl.process_frame_with_visualization(
                        frame, source_type="webcam" if isinstance(source, int) else "ipcam"
                    )
                    
                    # Persist to database (async would be better for performance)
                    for vehicle in result.get('vehicles', []):
                        try:
                            crops = vehicle.get('crops', {})
                            vehicle_crop = ApiController._decode_base64_image(crops.get('vehicle'))
                            plate_crop = ApiController._decode_base64_image(crops.get('plate'))
                            driver_crop = ApiController._decode_base64_image(crops.get('driver'))
                            
                            bbox_data = {
                                'vehicle': vehicle.get('bbox'),
                                'plate': vehicle.get('plate', {}).get('bbox'),
                                'driver': vehicle.get('driver', {}).get('bbox') if vehicle.get('driver', {}).get('present') else None
                            }
                            
                            identity_manager.process_detection(
                                plate_text=vehicle.get('plate', {}).get('text'),
                                plate_conf=vehicle.get('plate', {}).get('confidence', 0.0),
                                vehicle_crop=vehicle_crop,
                                plate_crop=plate_crop,
                                driver_crop=driver_crop,
                                frame_image=frame,
                                annotated_image=annotated,
                                vehicle_type=vehicle.get('vehicle_type'),
                                vehicle_color=vehicle.get('vehicle_color'),
                                bbox_data=bbox_data,
                                source_type="webcam" if isinstance(source, int) else "ipcam"
                            )
                        except Exception as e:
                            print(f"[Stream] Identity persist error: {e}")
                    
                    frame = annotated
                    
                except Exception as e:
                    print(f"[Stream] Pipeline Error: {e}")

            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        cap.release()

    @staticmethod
    def video_feed_webcam():
        """MJPEG stream from webcam"""
        return Response(
            ApiController.stream_feed(0),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    
    @staticmethod
    def video_feed_ipcam(url):
        """MJPEG stream from IP camera"""
        return Response(
            ApiController.stream_feed(url),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
