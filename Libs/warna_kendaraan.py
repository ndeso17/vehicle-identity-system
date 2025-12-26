from .utils_color import get_dominant_color

class ColorDetector:
    def __init__(self):
        pass # No model to load

    def detect_color(self, vehicle_crop):
        """
        Detect color of the vehicle crop.
        """
        return get_dominant_color(vehicle_crop)
