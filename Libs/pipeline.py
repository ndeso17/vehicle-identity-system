import cv2
import datetime
import numpy as np
import os
import base64

# Import libraries - assuming they are in the same package or available in path
from .plat_nomor import LicensePlateDetector
from .jenis_kendaraan import VehicleDetector
from .warna_kendaraan import ColorDetector
from .pengemudi_kendaraan import DriverAttribution
from .utils_association import associate_vehicle_to_plate

from .config import (
    COLOR_PLATE_BOX, COLOR_PLATE_BOX_ORIGINAL, COLOR_VEHICLE_BOX, COLOR_DRIVER_BOX, COLOR_CABIN_DEBUG,
    BOX_THICKNESS, FONT_SCALE, FONT_THICKNESS
)



class Pipeline:
    def __init__(self):
        """
        Initialize and cache all models.
        """
        print("[Pipeline] Initializing models...")
        self.plate_detector = LicensePlateDetector()
        self.vehicle_detector = VehicleDetector()
        self.color_detector = ColorDetector()
        self.driver_attribution = DriverAttribution()
        # identity registry will be used for grouping
        print("[Pipeline] All models loaded.")

    def _to_base64(self, crop):
        """Helper: Encodes numpy crop to base64 string"""
        if crop is None or crop.size == 0:
            return None
        try:
            success, buffer = cv2.imencode('.jpg', crop)
            if not success: return None
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"[PIPELINE] Base64 encoding failed: {e}")
            return None

    def process_frame(self, frame, source_type="image"):
        """
        Main orchestration method with TOLERANT detection logic.
        
        Flow:
        1. Detect Plate (retain ALL detections, even if OCR fails)
        2. Detect Vehicle (associate plates to vehicles)
        3. Detect Color
        4. Detect Driver
        5. Output JSON with plate.detected and plate.readable flags
        
        Key Change: NO LONGER STOPS if OCR fails - separates detection from reading.
        """
        timestamp = datetime.datetime.now().isoformat()
        
        # 1. Detect Plates (TNKB) - Get ALL detections
        plates_raw = self.detect_plate_raw(frame)

        print(f"[PIPELINE] Detected {len(plates_raw)} plate(s) in frame")

        # If no plates detected at all, return empty
        if not plates_raw:
            print("[PIPELINE] No plates detected - returning empty vehicles list")
            result = {
                "timestamp": timestamp,
                "source": source_type,
                "vehicles": []
            }
            # Persist mining data as JSONL (including grouping info will be appended later)
            try:
                os.makedirs('results', exist_ok=True)
                with open(os.path.join('results', 'mining_data.jsonl'), 'a') as fh:
                    import json
                    fh.write(json.dumps(result) + "\n")
            except Exception as e:
                print(f"[PIPELINE] Failed to write mining data: {e}")

            return result

        # Basic object detections (vehicles/persons)
        try:
            objs = self.vehicle_detector.detect_objects(frame)
            if isinstance(objs, tuple) or isinstance(objs, list) and len(objs) == 2:
                try:
                    vehicles_raw, persons_raw = objs
                except Exception:
                    vehicles_raw = objs
                    persons_raw = []
            else:
                vehicles_raw = objs
                persons_raw = []
        except Exception:
            vehicles_raw = []
            persons_raw = []

        processed_vehicles = []
        plates_readable_count = 0

        for plate_idx, p in enumerate(plates_raw):
            plate_bbox_original = p.get('bbox') or [0, 0, 0, 0]
            final_crop = p.get('crop')

            # OCR read
            try:
                ocr_text, ocr_conf = self.read_plate_text(final_crop)
            except Exception:
                ocr_text, ocr_conf = "", 0.0

            is_readable = bool(ocr_text and ocr_text.strip()) and (ocr_conf > 0.2)
            if is_readable:
                plates_readable_count += 1

            # Use plate bbox as final bbox (xyxy)
            plate_bbox_final = plate_bbox_original

            # 2a. Detect Vehicle (Find best matching vehicle for this plate)
            vehicle_info = self.detect_vehicle(frame, plate_bbox_final, precomputed_vehicles=vehicles_raw)
            if not vehicle_info:
                vehicle_info = {
                    "vehicle_type": "Unknown",
                    "bbox": {"x": 0, "y": 0, "w": 0, "h": 0},
                    "raw_bbox": [0, 0, 0, 0],
                    "crop": None
                }

            # 2b. Detect Color
            vehicle_color = "Unknown"
            try:
                if vehicle_info.get('crop') is not None and getattr(vehicle_info.get('crop'), 'size', 0) > 0:
                    vehicle_color = self.detect_color(vehicle_info['crop'])
            except Exception:
                vehicle_color = "Unknown"

            # 2c. Detect Driver
            driver_info = {"present": False, "bbox": {"x": 0, "y": 0, "w": 0, "h": 0}}
            try:
                if vehicle_info.get('raw_bbox') and vehicle_info.get('raw_bbox') != [0, 0, 0, 0]:
                    driver_res = None
                    try:
                        person_boxes = [pr.get('bbox') for pr in persons_raw] if persons_raw else []
                        driver_res = self.driver_attribution.get_driver(vehicle_info['raw_bbox'], vehicle_info['vehicle_type'], persons_raw, person_boxes, plate_bbox=plate_bbox_final)
                    except Exception:
                        try:
                            driver_res = self.driver_attribution.get_driver(vehicle_info['raw_bbox'], vehicle_info['vehicle_type'], persons_raw, person_boxes)
                        except Exception:
                            driver_res = None

                    if driver_res:
                        driver_info = {"present": True, "bbox": driver_res.get('bbox', {"x":0,"y":0,"w":0,"h":0}), "confidence_reason": driver_res.get('confidence_reason', '')}
            except Exception:
                pass

            # Convert plate bbox to xywh for JSON output
            try:
                rx1, ry1, rx2, ry2 = plate_bbox_final
            except Exception:
                rx1 = ry1 = rx2 = ry2 = 0
            plate_bbox_dict = {"x": int(rx1), "y": int(ry1), "w": int(max(0, rx2 - rx1)), "h": int(max(0, ry2 - ry1))}

            # Store original bbox for debugging if needed
            try:
                ox1, oy1, ox2, oy2 = plate_bbox_original
            except Exception:
                ox1 = oy1 = ox2 = oy2 = 0
            original_bbox_dict = {"x": int(ox1), "y": int(oy1), "w": int(max(0, ox2 - ox1)), "h": int(max(0, oy2 - oy1))}

            # Encode Crops to Base64
            b64_vehicle = self._to_base64(vehicle_info.get('crop'))
            b64_plate = self._to_base64(final_crop)
            b64_driver = None
            if driver_info.get('present') and driver_info.get('bbox'):
                try:
                    dx, dy, dw, dh = driver_info['bbox']['x'], driver_info['bbox']['y'], driver_info['bbox']['w'], driver_info['bbox']['h']
                    h_img, w_img = frame.shape[:2]
                    dxc = max(0, int(dx)); dyc = max(0, int(dy))
                    driver_crop = frame[dyc:min(h_img, dyc+int(dh)), dxc:min(w_img, dxc+int(dw))]
                    b64_driver = self._to_base64(driver_crop)
                except Exception:
                    b64_driver = None

            vehicle_obj = {
                "vehicle_id": f"veh_{datetime.datetime.now().strftime('%f')}_{np.random.randint(100, 999)}",
                "vehicle_type": vehicle_info.get('vehicle_type', 'Unknown'),
                "vehicle_color": vehicle_color,
                "bbox": vehicle_info.get('bbox', {"x":0,"y":0,"w":0,"h":0}),
                "plate": {
                    "detected": True,
                    "bbox": plate_bbox_dict,
                    "text": ocr_text or "",
                    "confidence": round(float(ocr_conf or 0.0), 3),
                    "readable": bool(is_readable),
                    "refined": False,
                    "original_bbox": original_bbox_dict
                },
                "driver": driver_info,
                "crops": {"vehicle": b64_vehicle, "plate": b64_plate, "driver": b64_driver}
            }

            processed_vehicles.append(vehicle_obj)
        result = {
            "timestamp": timestamp,
            "source": source_type,
            "vehicles": processed_vehicles
        }

        # Grouping done per-vehicle above.
        
        # Persist mining data as JSONL (including grouping info)
        try:
            os.makedirs('results', exist_ok=True)
            with open(os.path.join('results', 'mining_data.jsonl'), 'a') as fh:
                import json
                fh.write(json.dumps(result) + "\n")
        except Exception as e:
            print(f"[PIPELINE] Failed to write mining data: {e}")

        return result

    def detect_plate_raw(self, frame):
        """
        Detects plates WITHOUT filtering by OCR.
        Returns ALL plate detections from YOLO model.
        
        Returns list of {'bbox': [x1,y1,x2,y2], 'confidence': float, 'crop': np.array}
        """
        raw_plates = self.plate_detector.detect_plate(frame)
        return raw_plates

    def read_plate_text(self, plate_crop, debug=False):
        """
        Attempts to read OCR text from plate crop.
        Returns (text, confidence) tuple.
        Returns ("", 0.0) if OCR fails - does NOT raise exception.
        """
        # OCR preprocessing & pytesseract pipeline per spec
        def _preprocess_and_ocr(img, debug=False):
            try:
                import pytesseract
            except Exception:
                print("[PIPELINE] pytesseract not available")
                return "", 0.0

            if img is None or img.size == 0:
                return "", 0.0

            try:
                # Grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Resize 2x
                gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                # CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0)
                gray = clahe.apply(gray)
                # Adaptive threshold
                th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
                # Morphology open (3x3)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                proc = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

                if debug:
                    os.makedirs('results/debug_ocr', exist_ok=True)
                    fname = os.path.join('results', 'debug_ocr', f"ocr_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png")
                    cv2.imwrite(fname, proc)

                # Tesseract config: PSM 7 and whitelist A-Z0-9
                tconf = "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 7"
                data = pytesseract.image_to_data(proc, output_type=pytesseract.Output.DICT, config=tconf)

                texts = []
                confs = []
                for i, txt in enumerate(data.get('text', [])):
                    txts = txt.strip()
                    if txts:
                        try:
                            c = int(data['conf'][i])
                        except Exception:
                            c = -1
                        if c > 0:
                            texts.append(txts)
                            confs.append(c)

                if not texts:
                    return "", 0.0

                final_text = "".join(texts)
                # Average confidence
                avg_conf = float(sum(confs)) / len(confs) if confs else 0.0
                # Normalize to 0..1
                norm_conf = avg_conf / 100.0

                # Apply confidence filter > 30%
                if avg_conf < 30:
                    return final_text, norm_conf

                return final_text, norm_conf
            except Exception as e:
                print(f"[PIPELINE] OCR Exception: {e}")
                return "", 0.0

        # Call preprocessing OCR
        return _preprocess_and_ocr(plate_crop, debug=debug)

    def detect_vehicle(self, frame, plate_bbox, precomputed_vehicles=None):
        """
        Identifies the vehicle associated with the plate.
        Returns dict with type, bbox (formatted), and crop.
        """
        # If precomputed not provided, run detection
        if precomputed_vehicles is None:
            precomputed_vehicles, _ = self.vehicle_detector.detect_objects(frame)
        
        # Convert list of objects to list of bboxes for association util
        vehicle_bboxes = [v['bbox'] for v in precomputed_vehicles]
        
        idx = associate_vehicle_to_plate(vehicle_bboxes, plate_bbox)
        
        if idx != -1:
            matched = precomputed_vehicles[idx]
            x1, y1, x2, y2 = matched['bbox']
            
            # Ensure coords within frame
            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            crop = frame[y1:y2, x1:x2]
            
            return {
                "vehicle_type": matched['type'],
                "bbox": {"x": int(x1), "y": int(y1), "w": int(x2-x1), "h": int(y2-y1)},
                "raw_bbox": [x1, y1, x2, y2],
                "crop": crop
            }
        
        return None

    def detect_color(self, vehicle_crop):
        """
        Detects color from vehicle crop.
        """
        return self.color_detector.detect_color(vehicle_crop)

    def detect_driver(self, vehicle_bbox, vehicle_type, persons_raw):
        """
        Detects driver within the vehicle.
        """
        person_bboxes = [p['bbox'] for p in persons_raw]
        driver_entry = self.driver_attribution.get_driver(vehicle_bbox, vehicle_type, persons_raw, person_bboxes)
        return driver_entry

    # =================== VISUAL DEBUGGING FUNCTIONS ===================
    # Colors (BGR format) - prefer config values
    VIS_COLOR_VEHICLE = COLOR_VEHICLE_BOX
    VIS_COLOR_DRIVER = COLOR_DRIVER_BOX
    VIS_COLOR_PLATE_REFINED_BGR = COLOR_PLATE_BOX
    VIS_COLOR_PLATE_FALLBACK_BGR = COLOR_PLATE_BOX_ORIGINAL

    def _draw_label(self, image, text, x, y, bg_color, text_color=(0, 0, 0), font_scale=0.5):
        """Helper to draw text with background box"""
        thickness = 1
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Clamp x,y to be inside image
        img_h, img_w = image.shape[:2]
        x = max(0, min(x, img_w - w))
        y = max(h + 4, min(y, img_h)) # ensure label doesn't go off top
        
        # Draw Background
        cv2.rectangle(image, (x, y - h - 4), (x + w, y + 2), bg_color, -1)
        # Draw Text
        cv2.putText(image, text, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
        return h + 6 # return height consumed including padding

    def draw_plate_bbox(self, image, bbox_dict, text=None, conf=None, readable=False, refined=False):
        x, y, w, h = bbox_dict['x'], bbox_dict['y'], bbox_dict['w'], bbox_dict['h']
        
        # Color logic: ORANGE (Config)
        color = self.VIS_COLOR_PLATE_REFINED_BGR 
        
        # Draw Box
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        
        # Base Label: PLATE (conf)
        conf_val = conf if conf else 0.0
        base_label = f"PLATE ({conf_val:.2f})"
        
        # Draw base label below
        self._draw_label(image, base_label, x, y + h + 15, color, text_color=(0,0,0), font_scale=0.4)
        
        # OCR Overlay (Only if readable)
        if readable and text:
            # Text format: DK 1301 KV \n (conf: 0.45)
            # Draw directly above bbox
            ocr_lines = [text, f"(conf: {conf_val:.2f})"]
            
            text_y = y - 5 # Start slightly above
            for line in reversed(ocr_lines):
                h_used = self._draw_label(image, line, x, text_y, (255,255,255), text_color=(0,0,0), font_scale=0.6)
                text_y -= h_used

    def draw_vehicle_bbox(self, image, bbox_dict, vehicle_type, vehicle_id):
        x, y, w, h = bbox_dict['x'], bbox_dict['y'], bbox_dict['w'], bbox_dict['h']
        color = self.VIS_COLOR_VEHICLE
        
        # Draw Box
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        
        # Label: ID | Type
        short_id = vehicle_id.split('_')[-1] if '_' in vehicle_id else vehicle_id
        label = f"{short_id} | {vehicle_type.upper()}"
        self._draw_label(image, label, x, y, color, text_color=(0,0,0))

    def draw_driver_bbox(self, image, bbox_dict):
        x, y, w, h = bbox_dict['x'], bbox_dict['y'], bbox_dict['w'], bbox_dict['h']
        color = self.VIS_COLOR_DRIVER
        
        # Draw Box
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        
        # Label
        self._draw_label(image, "DRIVER", x, y, color, text_color=(255,255,255))

    def annotate_image(self, image, vehicles_data):
        """
        Draw all annotations (vehicles, plates, drivers) on image.
        Order: Vehicle (Bottom) -> Driver (Middle) -> Plate (Top)
        """
        annotated = image.copy()
        
        # Layer 1: Vehicles (Bottom)
        for vehicle in vehicles_data:
            if vehicle['bbox']['w'] > 0:
                self.draw_vehicle_bbox(annotated, vehicle['bbox'], 
                                     vehicle['vehicle_type'], vehicle['vehicle_id'])
        
        # Layer 2: Drivers (Middle)
        for vehicle in vehicles_data:
             if vehicle['driver']['present']:
                self.draw_driver_bbox(annotated, vehicle['driver']['bbox'])
                # Draw cabin debug area if possible
                try:
                    vx = vehicle['bbox']['x']; vy = vehicle['bbox']['y']; vw = vehicle['bbox']['w']; vh = vehicle['bbox']['h']
                    # Derive cabin area similarly to DriverAttribution for visualization
                    if vehicle['vehicle_type'] == 'car':
                        cx1 = int(vx)
                        cx2 = int(vx + 0.6 * vw)
                        cy1 = int(vy)
                        cy2 = int(vy + 0.6 * vh)
                    elif vehicle['vehicle_type'] == 'motorcycle':
                        cx1 = int(vx)
                        cx2 = int(vx + vw)
                        cy1 = int(vy)
                        cy2 = int(vy + 0.8 * vh)
                    else:
                        cx1 = int(vx)
                        cx2 = int(vx + vw)
                        cy1 = int(vy)
                        cy2 = int(vy + 0.6 * vh)
                    cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), COLOR_CABIN_DEBUG, 1)
                except Exception:
                    pass
                
        # Layer 3: Plates (Top)
        for vehicle in vehicles_data:
             if vehicle['plate']['detected']:
                # Pass refined flag
                self.draw_plate_bbox(
                    annotated,
                    vehicle['plate']['bbox'],
                    text=vehicle['plate']['text'],
                    conf=vehicle['plate']['confidence'],
                    readable=vehicle['plate']['readable'],
                    refined=False
                )
        
        return annotated

    def process_frame_with_visualization(self, frame, source_type="image", save_path=None):
        """
        Process frame and return both JSON data AND annotated image.
        
        Args:
            frame: Input image
            source_type: "image" or "video"
            save_path: Optional path to save annotated image
        
        Returns:
            Tuple of (json_data, annotated_image)
        """
        # Get JSON data
        json_data = self.process_frame(frame, source_type)
        
        # Create annotated image
        annotated_image = self.annotate_image(frame, json_data['vehicles'])
        
        # Optionally save
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, annotated_image)
            print(f"[PIPELINE] Annotated image saved to: {save_path}")
        
        return json_data, annotated_image
