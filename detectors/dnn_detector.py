"""Enhanced Deep Learning-based face detector with multiple model support."""

import cv2
import numpy as np
import os
import urllib.request
from typing import List, Tuple, Optional
from .base_detector import BaseDetector

class DNNDetector(BaseDetector):
    """Enhanced DNN face detector with multiple model support."""
    
    def __init__(self, config):
        """Initialize DNN detector."""
        super().__init__(config)
        self.net = None
        self.model_type = None
        self.input_size = (300, 300)
        self.mean_values = [104, 117, 123]
        
        # Model configurations
        self.models = {
            'opencv_dnn': {
                'model_file': 'opencv_face_detector_uint8.pb',
                'config_file': 'opencv_face_detector.pbtxt',
                'input_size': (300, 300),
                'mean_values': [104, 117, 123],
                'scale_factor': 1.0,
                'swap_rb': False
            },
            'caffe_dnn': {
                'model_file': 'deploy.prototxt',
                'weights_file': 'res10_300x300_ssd_iter_140000.caffemodel',
                'input_size': (300, 300),
                'mean_values': [104, 117, 123],
                'scale_factor': 1.0,
                'swap_rb': False
            },
            'onnx_ultraface': {
                'model_file': 'version-RFB-320.onnx',
                'input_size': (320, 240),
                'mean_values': [127, 127, 127],
                'scale_factor': 1.0/128.0,
                'swap_rb': True
            }
        }
        
        # Download URLs for models
        self.model_urls = {
            'opencv_face_detector_uint8.pb': 
                'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb',
            'opencv_face_detector.pbtxt':
                'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/opencv_face_detector.pbtxt',
            'deploy.prototxt':
                'https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt',
            'res10_300x300_ssd_iter_140000.caffemodel':
                'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel',
            'version-RFB-320.onnx':
                'https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/raw/master/models/onnx/version-RFB-320.onnx'
        }
        
        self.models_dir = 'models'
        
    def initialize(self) -> bool:
        """Initialize DNN face detector with the best available model."""
        # Create models directory
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Try different models in order of preference
        model_priority = ['caffe_dnn', 'opencv_dnn', 'onnx_ultraface']
        
        for model_name in model_priority:
            print(f"Attempting to initialize {model_name}...")
            if self._initialize_model(model_name):
                self.model_type = model_name
                self.is_initialized = True
                print(f"✓ {model_name} initialized successfully")
                return True
        
        print("⚠ No DNN models could be initialized")
        return False
    
    def _initialize_model(self, model_name: str) -> bool:
        """Initialize a specific model."""
        try:
            model_config = self.models[model_name]
            
            if model_name == 'opencv_dnn':
                return self._init_opencv_dnn(model_config)
            elif model_name == 'caffe_dnn':
                return self._init_caffe_dnn(model_config)
            elif model_name == 'onnx_ultraface':
                return self._init_onnx_ultraface(model_config)
            
        except Exception as e:
            print(f"Failed to initialize {model_name}: {e}")
            return False
        
        return False
    
    def _init_opencv_dnn(self, config: dict) -> bool:
        """Initialize OpenCV's built-in DNN model."""
        try:
            # Try to use OpenCV's built-in samples first
            model_path = cv2.samples.findFile("face_detector/opencv_face_detector_uint8.pb")
            config_path = cv2.samples.findFile("face_detector/opencv_face_detector.pbtxt")
            
            if os.path.exists(model_path) and os.path.exists(config_path):
                self.net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
            else:
                # Download models if not found
                model_file = self._download_model('opencv_face_detector_uint8.pb')
                config_file = self._download_model('opencv_face_detector.pbtxt')
                
                if model_file and config_file:
                    self.net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
                else:
                    return False
            
            self._update_model_config(config)
            return True
            
        except Exception as e:
            print(f"OpenCV DNN initialization failed: {e}")
            return False
    
    def _init_caffe_dnn(self, config: dict) -> bool:
        """Initialize Caffe DNN model."""
        try:
            # Download required files
            prototxt_file = self._download_model('deploy.prototxt')
            model_file = self._download_model('res10_300x300_ssd_iter_140000.caffemodel')
            
            if not (prototxt_file and model_file):
                return False
            
            self.net = cv2.dnn.readNetFromCaffe(prototxt_file, model_file)
            self._update_model_config(config)
            return True
            
        except Exception as e:
            print(f"Caffe DNN initialization failed: {e}")
            return False
    
    def _init_onnx_ultraface(self, config: dict) -> bool:
        """Initialize ONNX UltraFace model."""
        try:
            model_file = self._download_model('version-RFB-320.onnx')
            
            if not model_file:
                return False
            
            self.net = cv2.dnn.readNetFromONNX(model_file)
            self._update_model_config(config)
            return True
            
        except Exception as e:
            print(f"ONNX UltraFace initialization failed: {e}")
            return False
    
    def _update_model_config(self, config: dict):
        """Update detector configuration based on model."""
        self.input_size = config['input_size']
        self.mean_values = config['mean_values']
        self.scale_factor = config['scale_factor']
        self.swap_rb = config['swap_rb']
    
    def _download_model(self, filename: str) -> Optional[str]:
        """Download model file if it doesn't exist."""
        file_path = os.path.join(self.models_dir, filename)
        
        # Check if file already exists
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            return file_path
        
        if filename not in self.model_urls:
            print(f"No download URL available for {filename}")
            return None
        
        try:
            print(f"Downloading {filename}...")
            url = self.model_urls[filename]
            
            # Download with progress
            def download_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, (downloaded * 100) // total_size)
                    print(f"\rDownloading {filename}: {percent}%", end='', flush=True)
            
            urllib.request.urlretrieve(url, file_path, download_progress)
            print(f"\n✓ {filename} downloaded successfully")
            
            # Verify download
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                return file_path
            else:
                print(f"✗ Download verification failed for {filename}")
                return None
                
        except Exception as e:
            print(f"\n✗ Failed to download {filename}: {e}")
            return None
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple]:
        """Detect faces using the initialized DNN model."""
        if not self.is_initialized or self.net is None:
            return []
        
        try:
            if self.model_type == 'onnx_ultraface':
                return self._detect_ultraface(frame)
            else:
                return self._detect_standard_dnn(frame)
                
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def _detect_standard_dnn(self, frame: np.ndarray) -> List[Tuple]:
        """Standard DNN detection for OpenCV and Caffe models."""
        h, w = frame.shape[:2]
        
        # Create blob
        blob = cv2.dnn.blobFromImage(
            frame, 
            self.scale_factor, 
            self.input_size, 
            self.mean_values, 
            self.swap_rb, 
            False
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
                
                # Ensure coordinates are within frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:  # Valid detection
                    faces.append((x1, y1, x2 - x1, y2 - y1, confidence))
        
        return faces
    
    def _detect_ultraface(self, frame: np.ndarray) -> List[Tuple]:
        """UltraFace specific detection logic."""
        h, w = frame.shape[:2]
        
        # Resize frame to model input size
        resized_frame = cv2.resize(frame, self.input_size)
        
        # Create blob
        blob = cv2.dnn.blobFromImage(
            resized_frame,
            self.scale_factor,
            self.input_size,
            self.mean_values,
            self.swap_rb,
            False
        )
        
        # Set input and run forward pass
        self.net.setInput(blob)
        outputs = self.net.forward()
        
        # UltraFace has different output format
        # outputs[0] contains bounding boxes, outputs[1] contains scores
        if len(outputs) >= 2:
            boxes = outputs[0][0]
            scores = outputs[1][0]
        else:
            # Fallback for different output formats
            return self._detect_standard_dnn(frame)
        
        faces = []
        
        # Scale factors for converting back to original image size
        scale_x = w / self.input_size[0]
        scale_y = h / self.input_size[1]
        
        for i in range(boxes.shape[0]):
            if len(scores.shape) > 1:
                confidence = scores[i][1]  # Background vs Face confidence
            else:
                confidence = scores[i]
            
            if confidence > self.config.CONFIDENCE_THRESHOLD:
                box = boxes[i]
                
                x1 = int(box[0] * scale_x)
                y1 = int(box[1] * scale_y)
                x2 = int(box[2] * scale_x)
                y2 = int(box[3] * scale_y)
                
                # Ensure coordinates are within frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:  # Valid detection
                    faces.append((x1, y1, x2 - x1, y2 - y1, confidence))
        
        return faces
    
    def get_model_info(self) -> dict:
        """Get information about the current model."""
        if not self.is_initialized:
            return {"status": "Not initialized"}
        
        return {
            "model_type": self.model_type,
            "input_size": self.input_size,
            "mean_values": self.mean_values,
            "scale_factor": self.scale_factor,
            "swap_rb": self.swap_rb,
            "status": "Initialized"
        }
    
    def set_backend_and_target(self, backend=cv2.dnn.DNN_BACKEND_DEFAULT, 
                              target=cv2.dnn.DNN_TARGET_CPU):
        """Set computational backend and target for the network."""
        if self.net is not None:
            try:
                self.net.setPreferableBackend(backend)
                self.net.setPreferableTarget(target)
                
                backend_names = {
                    cv2.dnn.DNN_BACKEND_DEFAULT: "Default",
                    cv2.dnn.DNN_BACKEND_HALIDE: "Halide",
                    cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE: "Intel IE",
                    cv2.dnn.DNN_BACKEND_OPENCV: "OpenCV",
                    cv2.dnn.DNN_BACKEND_VKCOM: "Vulkan",
                    cv2.dnn.DNN_BACKEND_CUDA: "CUDA"
                }
                
                target_names = {
                    cv2.dnn.DNN_TARGET_CPU: "CPU",
                    cv2.dnn.DNN_TARGET_OPENCL: "OpenCL",
                    cv2.dnn.DNN_TARGET_OPENCL_FP16: "OpenCL FP16",
                    cv2.dnn.DNN_TARGET_MYRIAD: "Myriad",
                    cv2.dnn.DNN_TARGET_VULKAN: "Vulkan",
                    cv2.dnn.DNN_TARGET_CUDA: "CUDA",
                    cv2.dnn.DNN_TARGET_CUDA_FP16: "CUDA FP16"
                }
                
                backend_name = backend_names.get(backend, f"Unknown({backend})")
                target_name = target_names.get(target, f"Unknown({target})")
                
                print(f"✓ DNN Backend: {backend_name}, Target: {target_name}")
                
            except Exception as e:
                print(f"⚠ Failed to set backend/target: {e}")
    
    def optimize_for_hardware(self):
        """Automatically optimize for available hardware."""
        if self.net is None:
            return
        
        # Try different backends in order of preference
        optimization_options = [
            (cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA, "CUDA GPU"),
            (cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_OPENCL, "OpenCL GPU"),
            (cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU, "CPU")
        ]
        
        for backend, target, name in optimization_options:
            try:
                self.set_backend_and_target(backend, target)
                
                # Test with a small dummy image
                dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
                test_result = self.detect_faces(dummy_frame)
                
                print(f"✓ Optimized for {name}")
                break
                
            except Exception:
                continue
        
        print("✓ Hardware optimization completed")


# Enhanced usage example
def example_usage():
    """Example of how to use the enhanced DNN detector."""
    from config import Config
    
    config = Config()
    detector = DNNDetector(config)
    
    if detector.initialize():
        # Get model information
        info = detector.get_model_info()
        print(f"Model Info: {info}")
        
        # Optimize for hardware
        detector.optimize_for_hardware()
        
        # Use the detector
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            faces = detector.detect_faces(frame)
            
            # Draw results
            for face in faces:
                x, y, w, h, conf = face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{conf:.2f}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow("Enhanced DNN Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    else:
        print("Failed to initialize DNN detector")

if __name__ == "__main__":
    example_usage()