
    parser.add_argument('--nms', type=float, default=Config.NMS_THRESHOLD,
                       help='NMS threshold (default: 0.4)')
    parser.add_argument('--width', type=int, default=Config.CAMERA_WIDTH,
                       help='Camera width (default: 640)')