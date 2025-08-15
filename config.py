"""Configuration settings for the face detection system."""

import cv2

class Config:
    """Configuration class for face detection system."""
    
    # Detection parameters
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    MIN_FACE_SIZE = (30, 30)
    
    # Camera settings
    DEFAULT_CAMERA_INDEX = 0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    
    # Performance settings
    FPS_BUFFER_SIZE = 30
    DETECTION_HISTORY_SIZE = 10
    STABILITY_THRESHOLD = 3
    OVERLAP_THRESHOLD = 0.3
    
    # Haar cascade parameters
    HAAR_SCALE_FACTOR = 1.1
    HAAR_MIN_NEIGHBORS = 5
    
    # Visual settings
    HIGH_CONFIDENCE_COLOR = (0, 255, 0)    # Green
    MEDIUM_CONFIDENCE_COLOR = (0, 255, 255) # Yellow
    LOW_CONFIDENCE_COLOR = (0, 165, 255)    # Orange
    INFO_PANEL_COLOR = (0, 0, 0)
    TEXT_COLOR = (0, 255, 0)
    
    # File paths
    HAAR_FACE_CASCADE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    HAAR_PROFILE_CASCADE = cv2.data.haarcascades + 'haarcascade_profileface.xml'
    HAAR_EYE_CASCADE = cv2.data.haarcascades + 'haarcascade_eye.xml'
