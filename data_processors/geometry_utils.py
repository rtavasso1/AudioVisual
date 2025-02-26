# data_processors/geometry_utils.py

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any

class GeometryUtils:
    """Utility class for geometric calculations on landmarks."""
    
    @staticmethod
    def calculate_distance_3d(point1, point2) -> float:
        """Calculate the 3D Euclidean distance between two landmarks.
        
        Args:
            point1: First landmark with x, y, z attributes
            point2: Second landmark with x, y, z attributes
            
        Returns:
            Euclidean distance between the points
        """
        return math.sqrt(
            (point1.x - point2.x) ** 2 +
            (point1.y - point2.y) ** 2 +
            (point1.z - point2.z) ** 2
        )
    
    @staticmethod
    def calculate_palm_center(hand_landmarks) -> Any:
        """Calculate the center of the palm using multiple landmarks.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            An object with x, y, z attributes representing the palm center
        """
        from tracking.hand_tracker import HandLandmark
        
        # Use the average position of wrist and base of all fingers
        palm_landmarks = [
            hand_landmarks.landmark[HandLandmark.WRIST],
            hand_landmarks.landmark[HandLandmark.THUMB_CMC],
            hand_landmarks.landmark[HandLandmark.INDEX_FINGER_MCP],
            hand_landmarks.landmark[HandLandmark.MIDDLE_FINGER_MCP],
            hand_landmarks.landmark[HandLandmark.RING_FINGER_MCP],
            hand_landmarks.landmark[HandLandmark.PINKY_MCP]
        ]
        
        # Calculate average position
        x_avg = sum(lm.x for lm in palm_landmarks) / len(palm_landmarks)
        y_avg = sum(lm.y for lm in palm_landmarks) / len(palm_landmarks)
        z_avg = sum(lm.z for lm in palm_landmarks) / len(palm_landmarks)
        
        # Create a simple object with x, y, z attributes
        class Point3D:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z
                
        return Point3D(x_avg, y_avg, z_avg)
    
    @staticmethod
    def calculate_angle(point1, point2, point3) -> float:
        """Calculate the angle between three points in 3D space.
        
        Args:
            point1, point2, point3: Three landmarks with x, y, z attributes,
                                    where point2 is the vertex of the angle
            
        Returns:
            Angle in degrees
        """
        # Create vectors
        vector1 = np.array([point1.x - point2.x, point1.y - point2.y, point1.z - point2.z])
        vector2 = np.array([point3.x - point2.x, point3.y - point2.y, point3.z - point2.z])
        
        # Normalize vectors
        vector1_norm = np.linalg.norm(vector1)
        vector2_norm = np.linalg.norm(vector2)
        
        if vector1_norm == 0 or vector2_norm == 0:
            return 0.0
            
        vector1 = vector1 / vector1_norm
        vector2 = vector2 / vector2_norm
        
        # Calculate dot product and clamp to prevent numerical errors
        dot_product = np.clip(np.dot(vector1, vector2), -1.0, 1.0)
        
        # Convert to degrees
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    @staticmethod
    def calculate_velocity(curr_point, prev_point, time_delta: float) -> Tuple[float, float, float]:
        """Calculate the velocity vector between two points over time.
        
        Args:
            curr_point: Current landmark with x, y, z attributes
            prev_point: Previous landmark with x, y, z attributes
            time_delta: Time elapsed between the two points (in seconds)
            
        Returns:
            Tuple of (vx, vy, vz) velocities
        """
        if time_delta <= 0:
            return (0.0, 0.0, 0.0)
            
        vx = (curr_point.x - prev_point.x) / time_delta
        vy = (curr_point.y - prev_point.y) / time_delta
        vz = (curr_point.z - prev_point.z) / time_delta
        
        return (vx, vy, vz)
    
    @staticmethod
    def calculate_smoothed_landmarks(curr_landmarks, prev_landmarks, smoothing_factor: float = 0.5):
        """Apply exponential smoothing to landmarks to reduce jitter.
        
        Args:
            curr_landmarks: Current MediaPipe landmarks
            prev_landmarks: Previous MediaPipe landmarks
            smoothing_factor: Weight of current landmarks (0-1), lower = more smoothing
            
        Returns:
            Smoothed landmarks object
        """
        if not prev_landmarks:
            return curr_landmarks
            
        # Create a copy of current landmarks to modify
        smoothed = type(curr_landmarks)()
        
        # Apply smoothing to each landmark
        for i in range(len(curr_landmarks.landmark)):
            curr_lm = curr_landmarks.landmark[i]
            prev_lm = prev_landmarks.landmark[i]
            
            # Create a new landmark with smoothed values
            smooth_lm = type(curr_lm)()
            smooth_lm.x = curr_lm.x * smoothing_factor + prev_lm.x * (1 - smoothing_factor)
            smooth_lm.y = curr_lm.y * smoothing_factor + prev_lm.y * (1 - smoothing_factor)
            smooth_lm.z = curr_lm.z * smoothing_factor + prev_lm.z * (1 - smoothing_factor)
            
            # Add the smoothed landmark
            smoothed.landmark.append(smooth_lm)
            
        return smoothed