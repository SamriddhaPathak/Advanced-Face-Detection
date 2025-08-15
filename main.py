"""Main entry point for the Advanced Face Detection System."""

import argparse
import sys
from config import Config
from core import FaceDetectionSystem

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Advanced Face Detection System')
    parser.add_argument('--camera', type=int, default=Config.DEFAULT_CAMERA_INDEX, 
                       help='Camera index (default: 0)')
    parser.add_argument('--confidence', type=float, default=Config.CONFIDENCE_THRESHOLD,
                       help='Confidence threshold for DNN (default: 0.5)')
    parser.add_argument('--nms', type=float, default=Config.NMS_THRESHOLD,
                       help='NMS threshold (default: 0.4)')
    parser.add_argument('--width', type=int, default=Config.CAMERA_WIDTH,
                       help='Camera width (default: 640)')
    parser.add_argument('--height', type=int, default=Config.CAMERA_HEIGHT,
                       help='Camera height (default: 480)')
    
    args = parser.parse_args()
    
    # Update configuration with command line arguments
    config = Config()
    config.CONFIDENCE_THRESHOLD = args.confidence
    config.NMS_THRESHOLD = args.nms
    config.CAMERA_WIDTH = args.width
    config.CAMERA_HEIGHT = args.height
    
    try:
        # Create and run face detection system
        system = FaceDetectionSystem(config)
        system.run(camera_index=args.camera)
        
    except KeyboardInterrupt:
        print("\nDetection stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()