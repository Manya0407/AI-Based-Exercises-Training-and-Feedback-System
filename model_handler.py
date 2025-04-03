import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

# Global model instance
model = None
movenet = None

# Keypoint indices for all exercises
KEYPOINTS = {
    'LEFT_SHOULDER': 5,
    'RIGHT_SHOULDER': 6,
    'LEFT_ELBOW': 7,
    'RIGHT_ELBOW': 8,
    'LEFT_WRIST': 9,
    'RIGHT_WRIST': 10,
    'LEFT_HIP': 11,
    'RIGHT_HIP': 12,
    'LEFT_KNEE': 13,
    'RIGHT_KNEE': 14,
    'LEFT_ANKLE': 15,
    'RIGHT_ANKLE': 16,
    'LEFT_EAR': 3,
    'RIGHT_EAR': 4
}

def load_model():
    """Load the MoveNet model with fallback mechanism"""
    global model, movenet
    
    if model is not None:
        return True
        
    try:
        # Try loading lightning model first
        model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        print("Lightning model loaded successfully")
    except Exception as e:
        print(f"Error loading lightning model: {e}")
        try:
            # Fallback to thunder model
            print("Attempting to load fallback model...")
            model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
            print("Thunder model loaded successfully")
        except Exception as e:
            print(f"Error loading fallback model: {e}")
            return False
    
    movenet = model.signatures['serving_default']
    return True

def detect_pose(frame):
    """Detect pose keypoints in a frame"""
    if model is None or movenet is None:
        if not load_model():
            return None
            
    try:
        # Ensure frame is valid
        if frame is None or frame.size == 0:
            print("Invalid frame received")
            return None
            
        # Resize image to prevent memory issues
        if frame.shape[0] > 720 or frame.shape[1] > 1280:
            frame = cv2.resize(frame, (min(1280, frame.shape[1]), min(720, frame.shape[0])))
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize image to expected format
        input_img = tf.image.resize_with_pad(tf.expand_dims(frame_rgb, axis=0), 192, 192)
        input_img = tf.cast(input_img, dtype=tf.int32)
        
        # Run inference
        outputs = movenet(input_img)
        keypoints = outputs['output_0'].numpy().reshape((17, 3))
        return keypoints
    except Exception as e:
        print(f"Error in pose detection: {e}")
        return None

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    try:
        if a is None or b is None or c is None:
            return None
            
        a, b, c = np.array(a), np.array(b), np.array(c)
        
        # Check for zero vectors that would cause division by zero
        ba = a - b
        bc = c - b
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return None
            
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle
    except Exception as e:
        print(f"Error in angle calculation: {e}")
        return None

def create_fullscreen_window(title):
    """Create a fullscreen window with proper error handling"""
    try:
        cv2.namedWindow(title, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        return True
    except Exception as e:
        print(f"Error creating fullscreen window: {e}")
        return False

def initialize_webcam():
    """Initialize webcam with proper error handling"""
    try:
        # Try to open the default camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            # Try alternative indices if default fails
            for i in range(1, 5):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"Opened camera at index {i}")
                    break
        
        if not cap.isOpened():
            print("Could not open any webcam.")
            return None
            
        # Set lower resolution to improve performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap
    except Exception as e:
        print(f"Error initializing webcam: {e}")
        return None 