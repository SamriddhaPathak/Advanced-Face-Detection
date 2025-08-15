"""Drawing and visualization utilities."""

import cv2
import numpy as np
from typing import List, Tuple

class DrawingUtils:
    """Utility functions for drawing and visualization."""
    
    @staticmethod
    def draw_face_detections(frame: np.ndarray, faces: List[Tuple], config, 
                           show_confidence: bool = True, show_id: bool = False) -> np.ndarray:
        """Draw face detection results on frame."""
        from .face_utils import FaceUtils
        
        for i, face in enumerate(faces):
            x, y, w, h = face[:4]
            confidence = face[4] if len(face) > 4 else 1.0
            
            # Get color based on confidence
            color = FaceUtils.get_confidence_color(confidence, config)
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw labels
            label_y = y - 10 if y > 30 else y + h + 20
            
            if show_confidence:
                conf_text = f"Face {confidence:.2f}"
                cv2.putText(frame, conf_text, (x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if show_id:
                id_text = f"ID: {i}"
                cv2.putText(frame, id_text, (x, label_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    @staticmethod
    def draw_info_panel(frame: np.ndarray, faces: List[Tuple], fps: float, 
                       detection_method: str, config) -> np.ndarray:
        """Draw information panel on frame."""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), config.INFO_PANEL_COLOR, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw information
        info_lines = [
            f"FPS: {fps:.1f}",
            f"Faces Detected: {len(faces)}",
            f"Resolution: {w}x{h}",
            f"Method: {detection_method}"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (20, 35 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.TEXT_COLOR, 1)
        
        return frame
    
    @staticmethod
    def draw_detection_mode(frame: np.ndarray, mode: str, config) -> np.ndarray:
        """Draw current detection mode on frame."""
        cv2.putText(frame, f"Mode: {mode.upper()}", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame