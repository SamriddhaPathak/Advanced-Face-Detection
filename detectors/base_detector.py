"""Base class for face detectors."""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple

class BaseDetector(ABC):
    """Abstract base class for face detectors."""
    
    def __init__(self, config):
        """Initialize the detector with configuration."""
        self.config = config
        self.is_initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the detector. Returns True if successful."""
        pass
    
    @abstractmethod
    def detect_faces(self, frame: np.ndarray) -> List[Tuple]:
        """
        Detect faces in the given frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of face detections as (x, y, w, h, confidence) tuples
        """
        pass
    
    def is_available(self) -> bool:
        """Check if the detector is available and initialized."""
        return self.is_initialized
    
    def get_detector_name(self) -> str:
        """Get the name of the detector."""
        return self.__class__.__name__
