"""DNN-based face detector."""

import cv2
import numpy as np
from typing import List, Tuple
from .base_detector import BaseDetector

class DNNDetector(BaseDetector):
    """DNN face detector implementation."""
    
    def __init__(self, config):
        """Initialize DNN detector."""
        super().__init__(config)
        self.net = None
        
    def initialize(self) -> bool:
        """Initialize DNN face detector."""
        try:
            # Try to load DNN model
            model_file = "opencv_face_detector_uint8.pb"
            config_file = "opencv_face_detector.pbtxt"
            
            self.net = cv2.dnn.readNetFromTensorflow(
                cv2.samples.findFile("face_detector/opencv_face_detector_uint8.pb"),
                cv2.samples.findFile("face_detector/opencv_face_detector.pbtxt")
            )
            
            self.is_initialized = True
            print("✓ DNN Face Detector loaded successfully")
            return True
            
        except Exception as e:
            print("⚠ DNN model not found, using Haar cascades only")
            return False
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple]:
        """Detect faces using DNN."""
        if not self.is_initialized:
            return []
        
        h, w = frame.shape[:2]
        
        # Create blob from frame
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), [104, 117, 123], False, False
        )
        
        # Set input and run forward pass
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.config.CONFIDENCE_THRESHOLD:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                faces.append((x1, y1, x2 - x1, y2 - y1, confidence))
        
        return faces