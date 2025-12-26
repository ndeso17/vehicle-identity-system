from ultralytics import YOLO
from .config import MODEL_YOLO_PATH, CONF_THRESHOLD_VEHICLE, TARGET_VEHICLE_CLASSES, DEVICE, CONF_THRESHOLD_PERSON, TARGET_PERSON_CLASS

class VehicleDetector:
    def __init__(self):
        print(f"Loading Vehicle Detection Model: {MODEL_YOLO_PATH}")
        self.model = YOLO(MODEL_YOLO_PATH)

    def detect_objects(self, image):
        """
        Detect vehicles and persons.
        Returns: 
            vehicles: list of {'bbox': ..., 'class_id': ..., 'conf': ...}
            persons: list of {'bbox': ..., 'conf': ...}
        """
        results = self.model.predict(image, conf=min(CONF_THRESHOLD_VEHICLE, CONF_THRESHOLD_PERSON), device=DEVICE, verbose=False)
        
        vehicles = []
        persons = []
        
        for result in results:
            names = result.names
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0].item())
                conf = box.conf[0].item()
                
                bbox = [int(x1), int(y1), int(x2), int(y2)]
                
                if cls in TARGET_VEHICLE_CLASSES and conf >= CONF_THRESHOLD_VEHICLE:
                    vehicle_type = names[cls] # car, motorcycle, etc.
                    vehicles.append({
                        'bbox': bbox,
                        'class_id': cls,
                        'type': vehicle_type,
                        'conf': conf
                    })
                elif cls == TARGET_PERSON_CLASS and conf >= CONF_THRESHOLD_PERSON:
                    persons.append({
                        'bbox': bbox,
                        'conf': conf
                    })
                    
        return vehicles, persons
