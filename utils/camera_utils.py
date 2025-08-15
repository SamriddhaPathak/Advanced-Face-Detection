"""Camera handling utilities."""

import cv2
import time
from collections import deque

class CameraUtils:
    """Utility functions for camera operations."""
    
    def __init__(self, config):
        """Initialize camera utilities."""
        self.config = config
        self.fps_counter = deque(maxlen=config.FPS_BUFFER_SIZE)
    
    def initialize_camera(self, camera_index: int = None) -> cv2.VideoCapture:
        """Initialize camera with optimal settings."""
        if camera_index is None:
            camera_index = self.config.DEFAULT_CAMERA_INDEX
        
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, self.config.CAMERA_FPS)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Camera initialized. Resolution: {actual_width}x{actual_height}")
        
        return cap
    
    def calculate_fps(self) -> float:
        """Calculate current FPS."""
        current_time = time.time()
        self.fps_counter.append(current_time)
        
        if len(self.fps_counter) > 1:
            fps = len(self.fps_counter) / (current_time - self.fps_counter[0])
            return fps
        return 0.0
    
    @staticmethod
    def save_frame(frame, prefix: str = "face_detection") -> str:
        """Save current frame to file."""
        filename = f"{prefix}_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        return filename
