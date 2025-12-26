"""
OCR Character Recognition Module
Handles loading and inference of the CNN-based OCR model for license plate character recognition.
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
from .config import MODEL_OCR_CHAR_PATH, DEVICE

# Character mapping for 36 classes (0-9, A-Z)
CHAR_MAP = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
    19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
    28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'
}


class OCRModel(nn.Module):
    """
    CNN architecture for OCR character recognition.
    Matches the state_dict structure from character_ocr.pt
    
    Architecture:
    - Input: (1, 1, 64, 64) - grayscale 64x64 images
    - Features: 3x Conv+BatchNorm blocks with MaxPooling
    - Classifier: 2 FC layers with dropout
    - Output: 36 classes (0-9, A-Z)
    """
    
    def __init__(self, num_classes=36):
        super(OCRModel, self).__init__()
        
        # Feature extraction layers
        # Input: 1x64x64 -> 32x64x64 -> 32x32x32 (after pool)
        # -> 64x32x32 -> 64x16x16 (after pool)
        # -> 128x16x16 -> 128x8x8 (after pool)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),     # features.0
            nn.BatchNorm2d(32),                              # features.1
            nn.ReLU(inplace=True),                           # features.2
            nn.MaxPool2d(kernel_size=2, stride=2),           # features.3
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),    # features.4
            nn.BatchNorm2d(64),                              # features.5
            nn.ReLU(inplace=True),                           # features.6
            nn.MaxPool2d(kernel_size=2, stride=2),           # features.7
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),   # features.8
            nn.BatchNorm2d(128),                             # features.9
            nn.ReLU(inplace=True),                           # features.10
            nn.MaxPool2d(kernel_size=2, stride=2),           # features.11
        )
        
        # Classifier layers
        # Flattened: 128 * 8 * 8 = 8192
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),                                 # classifier.0
            nn.Linear(8192, 256),                            # classifier.1
            nn.ReLU(inplace=True),                           # classifier.2
            nn.Dropout(0.5),                                 # classifier.3
            nn.Linear(256, num_classes),                     # classifier.4
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 1, 64, 64)
        
        Returns:
            Output logits of shape (batch, num_classes)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten: (batch, 128*8*8) = (batch, 8192)
        x = self.classifier(x)
        return x


def load_ocr_model(model_path=MODEL_OCR_CHAR_PATH, device=DEVICE):
    """
    Load the OCR model from state_dict file.
    
    Args:
        model_path: Path to the .pt file containing state_dict
        device: Device to load the model on ('cpu' or 'cuda')
    
    Returns:
        Loaded OCRModel in eval mode, or None if loading fails
    """
    try:
        # Create model instance
        model = OCRModel(num_classes=36)
        
        # Load state_dict
        state_dict = torch.load(model_path, map_location=device)
        
        # Verify it's a state_dict (OrderedDict)
        if not isinstance(state_dict, dict):
            print(f"ERROR: Expected state_dict (dict), got {type(state_dict)}")
            return None
        
        # Load weights into model
        model.load_state_dict(state_dict)
        
        # Set to evaluation mode
        model.eval()
        
        print(f"[OCR] Model loaded successfully from {model_path}")
        return model
        
    except Exception as e:
        print(f"[OCR] Failed to load model: {e}")
        return None


def preprocess_plate_image(image):
    """
    Preprocess plate image for OCR inference.
    
    Args:
        image: Input image (numpy array), can be BGR or grayscale
    
    Returns:
        Preprocessed tensor of shape (1, 1, 64, 64)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Resize to 64x64 (model input size)
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Convert to tensor: (1, 1, 64, 64)
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    
    return tensor


def recognize_characters(plate_image, model=None):
    """
    Recognize characters from a license plate image using OCR.
    
    This is the main public API function for OCR inference.
    
    Args:
        plate_image: Plate crop image (numpy array)
        model: Pre-loaded OCRModel (optional, will load if None)
    
    Returns:
        dict with keys:
            - 'text': Recognized text string
            - 'confidence': Confidence score (0.0 to 1.0)
    """
    # Handle invalid input
    if plate_image is None or plate_image.size == 0:
        return {"text": "", "confidence": 0.0}
    
    # Load model if not provided
    if model is None:
        model = load_ocr_model()
        if model is None:
            return {"text": "", "confidence": 0.0}
    
    try:
        # Preprocess image
        input_tensor = preprocess_plate_image(plate_image)
        
        # Run inference (no gradient computation needed)
        with torch.no_grad():
            output = model(input_tensor)
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(output, dim=1)
            
            # Get predicted class and confidence
            confidence, predicted_class = torch.max(probabilities, 1)
            
            # Convert to Python values
            predicted_idx = predicted_class.item()
            conf_value = confidence.item()
            
            # Map to character
            predicted_char = CHAR_MAP.get(predicted_idx, '?')
            
            return {
                "text": predicted_char,
                "confidence": round(conf_value, 3)
            }
    
    except Exception as e:
        print(f"[OCR] Recognition error: {e}")
        return {"text": "", "confidence": 0.0}


# For backward compatibility and convenience
_global_model = None

def get_ocr_model():
    """
    Get or initialize the global OCR model instance.
    This allows the model to be loaded once and reused.
    
    Returns:
        OCRModel instance or None
    """
    global _global_model
    if _global_model is None:
        _global_model = load_ocr_model()
    return _global_model
