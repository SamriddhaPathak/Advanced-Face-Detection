"""Main face detection system implementation."""

import cv2
import numpy as np
import time
from typing import List, Tuple
from detectors import HaarDetector, DNNDetector
from utils import FaceUtils, DrawingUtils, CameraUtils

class FaceDetectionSystem:
    """Main face detection system class."""
    
    def __init__(self, config):
        """Initialize the face detection system."""
        self.config = config
        
        # Initialize detectors
        self.haar_detector = HaarDetector(config)
        self.dnn_detector = DNNDetector(config)
        
        # Initialize utilities
        self.camera_utils = CameraUtils(config)
        
        # Initialize detectors
        self._initialize_detectors()
        
        # Detection modes
        self.detection_modes = ['robust', 'haar', 'dnn']
        self.current_mode = 'robust'
        
        print("Face Detection System Initialized")
        self._print_available_methods()
    
    def _initialize_detectors(self):
        """Initialize all available detectors."""
        self.haar_available = self.haar_detector.initialize()
        self.dnn_available = self.dnn_detector.initialize()
        
        if not self.haar_available and not self.dnn_available:
            raise RuntimeError("No detection methods available!")
    
    def _print_available_methods(self):
        """Print available detection methods."""
        methods = []
        if self.haar_available:
            methods.append("Haar Cascades")
        if self.dnn_available:
            methods.append("DNN")
        
        print(f"Available detection methods: {', '.join(methods)}")
    
    def detect_faces(self, frame: np.ndarray, mode: str = None) -> List[Tuple]:
        """Detect faces using specified mode."""
        if mode is None:
            mode = self.current_mode
        
        all_faces = []
        
        if mode == 'robust':
            # Use all available methods
            if self.haar_available:
                haar_faces = self.haar_detector.detect_faces(frame)
                all_faces.extend(haar_faces)
            
            if self.dnn_available:
                dnn_faces = self.dnn_detector.detect_faces(frame)
                all_faces.extend(dnn_faces)
            
            # Remove overlaps and validate
            unique_faces = FaceUtils.remove_overlapping_faces(
                all_faces, self.config.OVERLAP_THRESHOLD
            )
            
            # Validate with eye detection if Haar is available
            if self.haar_available:
                validated_faces = []
                for face in unique_faces:
                    if self.haar_detector.validate_face_with_eyes(frame, face):
                        validated_faces.append(face)
                return validated_faces
            
            return unique_faces
            
        elif mode == 'haar' and self.haar_available:
            return self.haar_detector.detect_faces(frame)
            
        elif mode == 'dnn' and self.dnn_available:
            return self.dnn_detector.detect_faces(frame)
        
        # Fallback
        if self.haar_available:
            return self.haar_detector.detect_faces(frame)
        elif self.dnn_available:
            return self.dnn_detector.detect_faces(frame)
        
        return []
    
    def get_detection_method_name(self, mode: str = None) -> str:
        """Get human-readable detection method name."""
        if mode is None:
            mode = self.current_mode
        
        if mode == 'robust':
            methods = []
            if self.haar_available:
                methods.append("Haar")
            if self.dnn_available:
                methods.append("DNN")
            return " + ".join(methods)
        elif mode == 'haar':
            return "Haar Only"
        elif mode == 'dnn':
            return "DNN Only"
        
        return "Unknown"
    
    def set_detection_mode(self, mode: str) -> bool:
        """Set detection mode."""
        if mode not in self.detection_modes:
            return False
        
        if mode == 'haar' and not self.haar_available:
            return False
        
        if mode == 'dnn' and not self.dnn_available:
            return False
        
        self.current_mode = mode
        return True
    
    def run(self, camera_index: int = None, window_name: str = "Advanced Face Detection"):
        """Run the main detection loop."""
        try:
            # Initialize camera
            cap = self.camera_utils.initialize_camera(camera_index)
            
            self._print_controls()
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                frame_count += 1
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect faces
                faces = self.detect_faces(frame, self.current_mode)
                
                # Draw detections
                frame = DrawingUtils.draw_face_detections(
                    frame, faces, self.config, show_confidence=True
                )
                
                # Calculate FPS
                fps = self.camera_utils.calculate_fps()
                
                # Draw info panel
                method_name = self.get_detection_method_name(self.current_mode)
                frame = DrawingUtils.draw_info_panel(
                    frame, faces, fps, method_name, self.config
                )
                
                # Draw detection mode
                frame = DrawingUtils.draw_detection_mode(
                    frame, self.current_mode, self.config
                )
                
                # Show frame
                cv2.imshow(window_name, frame)
                
                # Handle key presses
                if not self._handle_key_press(frame):
                    break
            
            print(f"\nSession completed. Processed {frame_count} frames.")
            
        except Exception as e:
            print(f"Error during detection: {e}")
        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()
    
    def _handle_key_press(self, frame) -> bool:
        """Handle keyboard input. Returns False to quit."""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            return False
        elif key == ord('s'):
            filename = CameraUtils.save_frame(frame)
            print(f"Frame saved as {filename}")
        elif key == ord('h') and self.haar_available:
            self.set_detection_mode('haar')
            print("Switched to Haar cascade mode")
        elif key == ord('d') and self.dnn_available:
            self.set_detection_mode('dnn')
            print("Switched to DNN mode")
        elif key == ord('r'):
            self.set_detection_mode('robust')
            print("Switched to robust mode")
        
        return True
    
    def _print_controls(self):
        """Print available controls."""
        print("\nControls:")
        print("- Press 'q' to quit")
        print("- Press 's' to save current frame")
        if self.haar_available:
            print("- Press 'h' to toggle Haar cascade only")
        if self.dnn_available:
            print("- Press 'd' to toggle DNN only")
        print("- Press 'r' to toggle robust mode")
