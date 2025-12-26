import cv2
import torch
from ultralytics import YOLO
import numpy as np
from .config import MODEL_PLAT_NOMOR_PRIMARY_PATH, MODEL_PLAT_NOMOR_FALLBACK_PATH, MODEL_OCR_CHAR_PATH, CONF_THRESHOLD_PLAT, DEVICE
from . import ocr_character

class LicensePlateDetector:
    def __init__(self):
        # Try primary then fallback model paths
        self.model_plat = None
        tried = []
        for path in (MODEL_PLAT_NOMOR_PRIMARY_PATH, MODEL_PLAT_NOMOR_FALLBACK_PATH):
            tried.append(path)
            try:
                print(f"Loading License Plate Model: {path}")
                self.model_plat = YOLO(path)
                print(f"Loaded plate model from: {path}")
                break
            except Exception as e:
                print(f"Failed to load plate model from {path}: {e}")

        if self.model_plat is None:
            print(f"CRITICAL ERROR: Failed to load any License Plate model. Tried: {tried}")
        
        print(f"Loading OCR Model: {MODEL_OCR_CHAR_PATH}")
        try:
            # Load CNN-based OCR model from state_dict
            self.model_ocr = ocr_character.load_ocr_model()
            if self.model_ocr is None:
                raise Exception("OCR model loading returned None")
        except Exception as e:
            print(f"WARNING: Failed to load OCR model: {e}")
            print("OCR functionality will be disabled.")
            self.model_ocr = None

    def detect_plate(self, image):
        """
        Detect license plates in the image.
        Returns a list of dicts: {'bbox': [x1, y1, x2, y2], 'confidence': float, 'crop': np.array}
        """
        if self.model_plat is None:
            print("Error: License Plate model not loaded.")
            return []
            
        results = self.model_plat.predict(image, conf=CONF_THRESHOLD_PLAT, device=DEVICE, verbose=False)
        plates = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                
                # Expand crop slightly (optional)
                h, w, _ = image.shape
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(w, int(x2))
                y2 = min(h, int(y2))
                
                plate_crop = image[y1:y2, x1:x2]
                
                plates.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'crop': plate_crop
                })
        
        print(f"[PLATE_DETECTOR] Found {len(plates)} plate candidate(s)")
        for i, p in enumerate(plates):
            bbox_str = f"[{p['bbox'][0]},{p['bbox'][1]},{p['bbox'][2]},{p['bbox'][3]}]"
            print(f"[PLATE_DETECTOR]   Plate {i+1}: bbox={bbox_str}, conf={p['confidence']:.3f}")
        
        return plates

    def read_text(self, plate_crop):
        """
        Perform OCR on the license plate crop.
        
        NOTE: This is a SINGLE CHARACTER recognition model.
        For full plate text, this should ideally segment the plate into individual
        characters first, or use a sequence-based OCR approach.
        
        Current implementation: Returns single character prediction.
        
        Returns: text (str), confidence (float)
        """
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0

        if self.model_ocr is None:
            return "", 0.0

        try:
            # Use the CNN-based OCR model
            result = ocr_character.recognize_characters(plate_crop, model=self.model_ocr)
            
            # Extract text and confidence
            text = result.get('text', '')
            confidence = result.get('confidence', 0.0)
            
            return text, confidence
            
        except Exception as e:
            print(f"[OCR] Error during text recognition: {e}")
            return "", 0.0
