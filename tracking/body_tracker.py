import mediapipe as mp
import numpy as np
import math
from typing import Dict, List, Tuple, Optional

class BodyTracker:
    def __init__(self, 
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5, 
                min_tracking_confidence=0.5):
        """Initialize the body pose tracker with MediaPipe"""
        self.pose_tracker = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Keep historical data for velocity calculations
        self.prev_landmarks = None
        self.prev_time = 0
    
    def process(self, image: np.ndarray):
        """Process an image and return pose landmarks"""
        return self.pose_tracker.process(image)
    
    @staticmethod
    def extract_points(pose_landmarks) -> np.ndarray:
        """Extract 3D points from pose landmarks"""
        points_3d = []
        for lm in pose_landmarks.landmark:
            x_3d = (lm.x - 0.5) * 3.0
            y_3d = -(lm.y - 0.5) * 2.0
            z_3d = -lm.z * 2.0
            points_3d.append([x_3d, y_3d, z_3d])
        return np.array(points_3d, dtype=np.float32)
    
    @staticmethod
    def calculate_body_orientation(pose_landmarks) -> Dict[str, float]:
        """Calculate rotation angles of the body in 3D space"""
        if not pose_landmarks:
            return {"x": 0, "y": 0, "z": 0}
            
        # Get key landmarks for orientation calculation
        left_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
        
        # Create vectors to define body orientation
        # Shoulder vector (right to left shoulder)
        shoulder_vec = np.array([
            left_shoulder.x - right_shoulder.x,
            left_shoulder.y - right_shoulder.y,
            left_shoulder.z - right_shoulder.z
        ])
        
        # Hip vector (right to left hip)
        hip_vec = np.array([
            left_hip.x - right_hip.x,
            left_hip.y - right_hip.y,
            left_hip.z - right_hip.z
        ])
        
        # Spine vector (from center of hips to center of shoulders)
        hip_center = np.array([
            (left_hip.x + right_hip.x) / 2,
            (left_hip.y + right_hip.y) / 2,
            (left_hip.z + right_hip.z) / 2
        ])
        shoulder_center = np.array([
            (left_shoulder.x + right_shoulder.x) / 2,
            (left_shoulder.y + right_shoulder.y) / 2,
            (left_shoulder.z + right_shoulder.z) / 2
        ])
        spine_vec = shoulder_center - hip_center
        
        # Normalize vectors
        shoulder_vec = shoulder_vec / np.linalg.norm(shoulder_vec)
        hip_vec = hip_vec / np.linalg.norm(hip_vec)
        spine_vec = spine_vec / np.linalg.norm(spine_vec)
        
        # Forward vector - perpendicular to shoulder vector and up
        forward_vec = np.array([shoulder_vec[2], 0, -shoulder_vec[0]])
        forward_vec = forward_vec / np.linalg.norm(forward_vec)
        
        # Calculate rotations
        # Y rotation (yaw) - left/right rotation using shoulder orientation
        yaw = math.atan2(shoulder_vec[2], shoulder_vec[0])
        
        # X rotation (pitch) - up/down tilt using spine vector
        pitch = math.atan2(spine_vec[1], math.sqrt(spine_vec[0]**2 + spine_vec[2]**2))
        
        # Z rotation (roll) - lean left/right using difference between shoulder and hip angles
        roll = math.atan2(shoulder_vec[1], hip_vec[1])
        
        return {
            "x": math.degrees(pitch),
            "y": math.degrees(yaw),
            "z": math.degrees(roll)
        }