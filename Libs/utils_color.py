import cv2
import numpy as np
from collections import Counter

def get_dominant_color(image_bgr):
    """
    Detect dominant color in an image crop (BGR) using HSV ranges.
    Returns a string representing the color name.
    """
    if image_bgr is None or image_bgr.size == 0:
        return "Unknown"

    hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # Define color ranges in HSV
    # OpenCV HSV ranges: H: 0-179, S: 0-255, V: 0-255
    color_ranges = {
        "Merah": [
            ([0, 50, 50], [10, 255, 255]),
            ([170, 50, 50], [180, 255, 255])
        ],
        "Hitam": [([0, 0, 0], [180, 255, 50])],  # Low Value
        "Putih": [([0, 0, 200], [180, 30, 255])], # High Value, Low Saturation
        "Abu-abu": [([0, 0, 50], [180, 50, 200])], # Medium Value, Low Saturation
        "Biru": [([100, 50, 50], [130, 255, 255])],
        "Hijau": [([35, 50, 50], [85, 255, 255])],
        "Kuning": [([20, 50, 50], [35, 255, 255])],
        "Oranye": [([10, 50, 50], [20, 255, 255])],
        "Ungu": [([130, 50, 50], [170, 255, 255])]
    }

    # Focus on center 25% region per spec (central quarter)
    h, w, _ = hsv_image.shape
    center_h, center_w = h // 2, w // 2
    crop_h, crop_w = max(1, h // 4), max(1, w // 4) # 25% region
    start_y = max(0, center_h - crop_h // 2)
    start_x = max(0, center_w - crop_w // 2)
    center_crop = hsv_image[start_y:start_y+crop_h, start_x:start_x+crop_w]

    max_pixels = 0
    dominant_color = "Lainnya"

    for color, ranges in color_ranges.items():
        mask_combined = None
        for (lower, upper) in ranges:
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            mask = cv2.inRange(center_crop, lower, upper)
            
            if mask_combined is None:
                mask_combined = mask
            else:
                mask_combined = cv2.bitwise_or(mask_combined, mask)
        
        count = cv2.countNonZero(mask_combined)
        if count > max_pixels:
            max_pixels = count
            dominant_color = color

    return dominant_color
