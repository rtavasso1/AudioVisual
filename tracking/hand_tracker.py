import mediapipe as mp
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from enum import IntEnum

class HandLandmark(IntEnum):
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

class HandTracker:
    def __init__(self, 
                static_image_mode=False, 
                max_num_hands=2, 
                model_complexity=1,
                min_detection_confidence=0.5, 
                min_tracking_confidence=0.5):
        """Initialize the hand tracker with MediaPipe"""
        self.hand_tracker = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Keep historical data for velocity calculations
        self.prev_landmarks = {
            "left": None,
            "right": None
        }
        self.prev_time = {
            "left": 0,
            "right": 0
        }
    
    def process(self, image: np.ndarray):
        """Process an image and return hand landmarks"""
        return self.hand_tracker.process(image)
    
    @staticmethod
    def extract_points(hand_landmarks) -> np.ndarray:
        """Extract 3D points from hand landmarks"""
        points_3d = []
        for lm in hand_landmarks.landmark:
            x_3d = (lm.x - 0.5) * 3.0
            y_3d = -(lm.y - 0.5) * 2.0
            z_3d = -lm.z * 2.0
            points_3d.append([x_3d, y_3d, z_3d])
        return np.array(points_3d, dtype=np.float32)
    
    @staticmethod
    def calculate_hand_orientation(hand_landmarks) -> Dict[str, float]:
        """Calculate rotation angles of the hand in 3D space"""
        if not hand_landmarks:
            return {"x": 0, "y": 0, "z": 0}
        
        # Get key landmarks to determine orientation
        wrist = hand_landmarks.landmark[HandLandmark.WRIST]
        middle_mcp = hand_landmarks.landmark[HandLandmark.MIDDLE_FINGER_MCP]
        index_mcp = hand_landmarks.landmark[HandLandmark.INDEX_FINGER_MCP]
        pinky_mcp = hand_landmarks.landmark[HandLandmark.PINKY_MCP]
        
        # Create vectors to define hand plane and direction
        # Forward vector (wrist to middle finger MCP)
        fwd_vec = np.array([
            middle_mcp.x - wrist.x,
            middle_mcp.y - wrist.y,
            middle_mcp.z - wrist.z
        ])
        fwd_vec = fwd_vec / np.linalg.norm(fwd_vec)
        
        # Right vector (pinky to index across hand)
        right_vec = np.array([
            index_mcp.x - pinky_mcp.x,
            index_mcp.y - pinky_mcp.y,
            index_mcp.z - pinky_mcp.z
        ])
        right_vec = right_vec / np.linalg.norm(right_vec)
        
        # Up vector (cross product of forward and right)
        up_vec = np.cross(right_vec, fwd_vec)
        up_vec = up_vec / np.linalg.norm(up_vec)
        
        # Calculate rotations around each axis
        # X rotation (pitch) - using forward vector's y component
        pitch = math.atan2(fwd_vec[1], math.sqrt(fwd_vec[0]**2 + fwd_vec[2]**2))
        
        # Y rotation (yaw) - using forward vector's x and z
        yaw = math.atan2(fwd_vec[0], fwd_vec[2])
        
        # Z rotation (roll) - using up vector's x and y
        roll = math.atan2(up_vec[0], up_vec[1])
        
        return {
            "x": math.degrees(pitch),
            "y": math.degrees(yaw),
            "z": math.degrees(roll)
        }
    
    @staticmethod
    def calculate_depth_velocity(current_landmarks, prev_landmarks, time_delta) -> float:
        """Calculate the speed of hand movement towards or away from the camera with smoothing"""
        if not current_landmarks or not prev_landmarks or time_delta <= 0:
            return 0.0
        
        # Use multiple landmarks to get a more stable depth value
        # Average depth from wrist and all finger MCP joints
        current_depth_points = [
            current_landmarks.landmark[HandLandmark.WRIST],
            current_landmarks.landmark[HandLandmark.INDEX_FINGER_MCP],
            current_landmarks.landmark[HandLandmark.MIDDLE_FINGER_MCP],
            current_landmarks.landmark[HandLandmark.RING_FINGER_MCP],
            current_landmarks.landmark[HandLandmark.PINKY_MCP]
        ]
        
        prev_depth_points = [
            prev_landmarks.landmark[HandLandmark.WRIST],
            prev_landmarks.landmark[HandLandmark.INDEX_FINGER_MCP],
            prev_landmarks.landmark[HandLandmark.MIDDLE_FINGER_MCP],
            prev_landmarks.landmark[HandLandmark.RING_FINGER_MCP],
            prev_landmarks.landmark[HandLandmark.PINKY_MCP]
        ]
        
        # Calculate average depth
        current_z = sum(point.z for point in current_depth_points) / len(current_depth_points)
        prev_z = sum(point.z for point in prev_depth_points) / len(prev_depth_points)
        
        # Calculate velocity (negative = towards camera, positive = away from camera)
        velocity = (current_z - prev_z) / time_delta
        
        # Apply additional threshold to filter out tiny movements (jitter)
        if abs(velocity) < 0.01:
            velocity = 0.0
            
        return velocity