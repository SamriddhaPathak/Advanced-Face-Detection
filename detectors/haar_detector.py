"""Haar cascade-based face detector."""

import cv2
import numpy as np
from typing import List, Tuple
from .base_detector import BaseDetector

class HaarDetector(BaseDetector):
    """Haar cascade face detector implementation."""
    
    def __init__(self, config):
        """Initialize Haar detector."""
        super().__init__(config)
        self.face_cascade = None
        self.profile_cascade = None
        self.eye_cascade = None
        
    def initialize(self) -> bool:
        """Initialize Haar cascade classifiers."""
        try:
            # Load face cascade
            self.face_cascade = cv2.CascadeClassifier(self.config.HAAR_FACE_CASCADE)
            if self.face_cascade.empty():
                print("⚠ Warning: Could not load frontal face cascade")
                return False
            
            # Load profile cascade
            self.profile_cascade = cv2.CascadeClassifier(self.config.HAAR_PROFILE_CASCADE)
            if self.profile_cascade.empty():
                print("⚠ Warning: Could not load profile face cascade")
            
            # Load eye cascade
            self.eye_cascade = cv2.CascadeClassifier(self.config.HAAR_EYE_CASCADE)
            if self.eye_cascade.empty():
                print("⚠ Warning: Could not load eye cascade")
            
            self.is_initialized = True
            print("✓ Haar Cascades loaded successfully")
            return True
            
        except Exception as e:
            print(f"⚠ Error loading Haar cascades: {e}")
            return False
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple]:
        """Detect faces using Haar cascades."""
        if not self.is_initialized:
            return []
        
        # Convert to grayscale and enhance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = []
        
        # Detect frontal faces
        if self.face_cascade is not None:
            frontal_faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.config.HAAR_SCALE_FACTOR,
                minNeighbors=self.config.HAAR_MIN_NEIGHBORS,
                minSize=self.config.MIN_FACE_SIZE,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Add confidence score (default 1.0 for Haar)
            for face in frontal_faces:
                faces.append((*face, 1.0))
        
        # Detect profile faces
        if self.profile_cascade is not None:
            profile_faces = self.profile_cascade.detectMultiScale(
                gray,
                scaleFactor=self.config.HAAR_SCALE_FACTOR,
                minNeighbors=self.config.HAAR_MIN_NEIGHBORS,
                minSize=self.config.MIN_FACE_SIZE
            )
            
            for face in profile_faces:
                faces.append((*face, 0.9))  # Slightly lower confidence for profile
        
        return faces
    
    def validate_face_with_eyes(self, frame: np.ndarray, face_rect: Tuple) -> bool:
        """Validate face detection by checking for eyes."""
        if self.eye_cascade is None:
            return True
        
        x, y, w, h = face_rect[:4]
        roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        
        eyes = self.eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(10, 10)
        )
        
        return len(eyes) >= 1