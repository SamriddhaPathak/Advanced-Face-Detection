"""Face processing utilities."""

import numpy as np
from typing import List, Tuple

class FaceUtils:
    """Utility functions for face processing."""
    
    @staticmethod
    def remove_overlapping_faces(faces: List[Tuple], overlap_threshold: float = 0.3) -> List[Tuple]:
        """Remove overlapping face detections."""
        if len(faces) == 0:
            return []
        
        # Ensure all faces have confidence scores
        processed_faces = []
        for face in faces:
            if len(face) == 4:
                processed_faces.append((*face, 1.0))
            else:
                processed_faces.append(face)
        
        # Sort by confidence (highest first)
        processed_faces = sorted(processed_faces, key=lambda x: x[4], reverse=True)
        
        keep = []
        while processed_faces:
            current = processed_faces.pop(0)
            keep.append(current)
            
            # Remove overlapping faces
            processed_faces = [
                face for face in processed_faces 
                if FaceUtils.calculate_overlap(current, face) < overlap_threshold
            ]
        
        return keep
    
    @staticmethod
    def calculate_overlap(face1: Tuple, face2: Tuple) -> float:
        """Calculate overlap ratio between two faces."""
        x1, y1, w1, h1 = face1[:4]
        x2, y2, w2, h2 = face2[:4]
        
        # Calculate intersection
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    @staticmethod
    def get_confidence_color(confidence: float, config) -> Tuple[int, int, int]:
        """Get color based on confidence level."""
        if confidence > 0.8:
            return config.HIGH_CONFIDENCE_COLOR
        elif confidence > 0.6:
            return config.MEDIUM_CONFIDENCE_COLOR
        else:
            return config.LOW_CONFIDENCE_COLOR
