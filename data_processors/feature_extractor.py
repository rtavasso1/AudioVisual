# data_processors/feature_extractor.py

import numpy as np
import time
from typing import Dict, Any, Optional
from data_processors.geometry_utils import GeometryUtils

class FeatureExtractor:
    """Extracts high-level features from raw tracking data."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.prev_time = time.time()
        self.prev_face_data = None
        self.prev_left_hand = None
        self.prev_right_hand = None
        self.prev_body_data = None
        
        # For velocity/acceleration calculation
        self.prev_features = {}
        
    def process(self, face_data=None, left_hand=None, right_hand=None, body_data=None) -> Dict[str, Any]:
        """Process tracking data and extract meaningful features.
        
        Args:
            face_data: MediaPipe face mesh landmarks
            left_hand: MediaPipe left hand landmarks
            right_hand: MediaPipe right hand landmarks
            body_data: MediaPipe body pose landmarks
            
        Returns:
            Dictionary of extracted features
        """
        current_time = time.time()
        time_delta = current_time - self.prev_time
        
        features = {}
        
        # Extract hand features
        if left_hand:
            features.update(self.extract_hand_features("left", left_hand, self.prev_left_hand, time_delta))
        
        if right_hand:
            features.update(self.extract_hand_features("right", right_hand, self.prev_right_hand, time_delta))
            
        # Extract face features if available
        if face_data:
            features.update(self.extract_face_features(face_data, self.prev_face_data, time_delta))
            
        # Extract body features if available
        if body_data:
            features.update(self.extract_body_features(body_data, self.prev_body_data, time_delta))
        
        # Calculate inter-object relationships
        if left_hand and right_hand:
            features.update(self.extract_hand_relationship_features(left_hand, right_hand))
            
        if face_data and (left_hand or right_hand):
            features.update(self.extract_face_hand_relationship_features(
                face_data, left_hand, right_hand))
            
        # Update previous state for next calculation
        self.prev_time = current_time
        self.prev_face_data = face_data
        self.prev_left_hand = left_hand
        self.prev_right_hand = right_hand
        self.prev_body_data = body_data
        
        # Calculate derivatives (velocity, acceleration) 
        features.update(self.calculate_derivatives(features))
        self.prev_features = features.copy()
        
        return features
    
    def extract_hand_features(self, hand_type: str, hand_landmarks, prev_landmarks, time_delta: float) -> Dict[str, Any]:
        """Extract features from hand landmarks.
        
        Args:
            hand_type: "left" or "right"
            hand_landmarks: MediaPipe hand landmarks
            prev_landmarks: Previous frame hand landmarks
            time_delta: Time elapsed since last frame
            
        Returns:
            Dictionary of hand features
        """
        from tracking.hand_tracker import HandLandmark, HandTracker
        
        features = {}
        
        # Basic positional features
        wrist = hand_landmarks.landmark[HandLandmark.WRIST]
        thumb_tip = hand_landmarks.landmark[HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[HandLandmark.PINKY_TIP]
        
        # Hand position (normalized 0-1)
        features[f"hand_pos_x_{hand_type}"] = wrist.x
        features[f"hand_pos_y_{hand_type}"] = wrist.y
        features[f"hand_pos_z_{hand_type}"] = wrist.z
        
        # Hand height (simple mapping to 0-1 range for controlling parameters)
        hand_height = 1.0 - wrist.y  # Invert so higher hand = higher value
        features[f"hand_height_{hand_type}"] = max(0.0, min(1.0, hand_height))
        
        # Calculate pinch value (distance between thumb and index finger)
        pinch_dist = GeometryUtils.calculate_distance_3d(thumb_tip, index_tip)
        # Normalize to 0-1 range (based on typical hand proportions)
        pinch_value = 1.0 - min(pinch_dist / 0.1, 1.0)
        features[f"hand_pinch_{hand_type}"] = pinch_value
        
        # Calculate other finger pinches
        middle_pinch = 1.0 - min(GeometryUtils.calculate_distance_3d(thumb_tip, middle_tip) / 0.1, 1.0)
        ring_pinch = 1.0 - min(GeometryUtils.calculate_distance_3d(thumb_tip, ring_tip) / 0.1, 1.0)
        pinky_pinch = 1.0 - min(GeometryUtils.calculate_distance_3d(thumb_tip, pinky_tip) / 0.1, 1.0)
        
        features[f"hand_middle_pinch_{hand_type}"] = middle_pinch
        features[f"hand_ring_pinch_{hand_type}"] = ring_pinch
        features[f"hand_pinky_pinch_{hand_type}"] = pinky_pinch
        
        # Calculate hand openness (average distance from palm center to fingertips)
        palm_center = GeometryUtils.calculate_palm_center(hand_landmarks)
        finger_tips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
        openness = np.mean([GeometryUtils.calculate_distance_3d(palm_center, tip) for tip in finger_tips])
        # Normalize to 0-1 range
        openness_norm = min(openness / 0.2, 1.0)
        features[f"hand_openness_{hand_type}"] = openness_norm
        
        # Calculate hand orientation
        orientation = HandTracker.calculate_hand_orientation(hand_landmarks)
        features[f"hand_rot_x_{hand_type}"] = orientation["x"] / 180.0  # Normalize to -1 to 1
        features[f"hand_rot_y_{hand_type}"] = orientation["y"] / 180.0
        features[f"hand_rot_z_{hand_type}"] = orientation["z"] / 180.0
        
        # Calculate depth velocity (movement towards/away from camera)
        if prev_landmarks and time_delta > 0:
            depth_velocity = HandTracker.calculate_depth_velocity(
                hand_landmarks, prev_landmarks, time_delta)
            # Normalize to a reasonable range (-1 to 1)
            norm_velocity = np.clip(depth_velocity / 0.5, -1.0, 1.0)
            features[f"hand_depth_velocity_{hand_type}"] = norm_velocity
        
        return features
    
    def extract_face_features(self, face_landmarks, prev_landmarks, time_delta: float) -> Dict[str, Any]:
        """Extract features from face landmarks.
        
        Args:
            face_landmarks: MediaPipe face mesh landmarks
            prev_landmarks: Previous frame face landmarks
            time_delta: Time elapsed since last frame
            
        Returns:
            Dictionary of face features
        """
        import mediapipe as mp
        
        features = {}
        
        # Basic face position (using nose tip landmark)
        nose_tip_idx = 4  # Nose tip index in MediaPipe face mesh
        nose_tip = face_landmarks.landmark[nose_tip_idx]
        
        features["face_pos_x"] = nose_tip.x
        features["face_pos_y"] = nose_tip.y
        features["face_pos_z"] = nose_tip.z
        
        # Mouth openness
        # Indices for upper and lower lip
        upper_lip_idx = 13
        lower_lip_idx = 14
        upper_lip = face_landmarks.landmark[upper_lip_idx]
        lower_lip = face_landmarks.landmark[lower_lip_idx]
        
        mouth_distance = GeometryUtils.calculate_distance_3d(upper_lip, lower_lip)
        # Normalize based on typical face proportions
        mouth_openness = min(mouth_distance / 0.05, 1.0)
        features["face_mouth_openness"] = mouth_openness
        
        # Eye openness (average of both eyes)
        # Simplified - using distance between eyelids
        left_eye_top_idx = 159
        left_eye_bottom_idx = 145
        right_eye_top_idx = 386
        right_eye_bottom_idx = 374
        
        left_eye_top = face_landmarks.landmark[left_eye_top_idx]
        left_eye_bottom = face_landmarks.landmark[left_eye_bottom_idx]
        right_eye_top = face_landmarks.landmark[right_eye_top_idx]
        right_eye_bottom = face_landmarks.landmark[right_eye_bottom_idx]
        
        left_eye_openness = GeometryUtils.calculate_distance_3d(left_eye_top, left_eye_bottom)
        right_eye_openness = GeometryUtils.calculate_distance_3d(right_eye_top, right_eye_bottom)
        
        # Normalize based on typical eye proportions
        avg_eye_openness = (left_eye_openness + right_eye_openness) / 2.0
        eye_openness_norm = min(avg_eye_openness / 0.025, 1.0)
        features["face_eye_openness"] = eye_openness_norm
        
        # Face orientation (using specific face landmarks)
        # More robust face orientation calculations
        left_temple_idx = 234
        right_temple_idx = 454
        chin_idx = 152
        forehead_idx = 10
        left_cheek_idx = 50    # Left cheek landmark
        right_cheek_idx = 280  # Right cheek landmark
        
        left_temple = face_landmarks.landmark[left_temple_idx]
        right_temple = face_landmarks.landmark[right_temple_idx] 
        chin = face_landmarks.landmark[chin_idx]
        forehead = face_landmarks.landmark[forehead_idx]
        left_cheek = face_landmarks.landmark[left_cheek_idx]
        right_cheek = face_landmarks.landmark[right_cheek_idx]
        
        # Calculate face rotation around y-axis (yaw - left/right)
        head_width = GeometryUtils.calculate_distance_3d(left_temple, right_temple)
        temple_diff_x = left_temple.x - right_temple.x
        
        # Normalized to approximately -1 to 1 range
        yaw = temple_diff_x / head_width * 2.0
        features["face_rot_y"] = yaw
        
        # Calculate face rotation around x-axis (pitch - up/down)
        face_height = GeometryUtils.calculate_distance_3d(forehead, chin)
        height_diff_y = forehead.y - chin.y
        
        # Normalized to approximately -1 to 1 range
        pitch = height_diff_y / face_height * 2.0
        features["face_rot_x"] = pitch
        
        # Calculate face rotation around z-axis (roll - tilting left/right)
        # Compare the y-positions of the temples to determine roll
        temple_diff_y = left_temple.y - right_temple.y
        
        # Alternative calculation using cheeks for more stability
        cheek_diff_y = left_cheek.y - right_cheek.y
        
        # Combine both measures for better stability
        roll = (temple_diff_y + cheek_diff_y) / 2.0 / head_width * 4.0  # Scale appropriately
        
        # Normalize to approximately -1 to 1 range
        roll = np.clip(roll, -1.0, 1.0)
        features["face_rot_z"] = roll
        
        return features
    
    def extract_body_features(self, body_landmarks, prev_landmarks, time_delta: float) -> Dict[str, Any]:
        """Extract features from body landmarks.
        
        Args:
            body_landmarks: MediaPipe pose landmarks
            prev_landmarks: Previous frame body landmarks
            time_delta: Time elapsed since last frame
            
        Returns:
            Dictionary of body features
        """
        import mediapipe as mp
        from tracking.body_tracker import BodyTracker
        
        features = {}
        
        # Basic body position (using mid hip point)
        mid_hip_idx = mp.solutions.pose.PoseLandmark.LEFT_HIP.value
        hip = body_landmarks.landmark[mid_hip_idx]
        
        features["body_pos_x"] = hip.x
        features["body_pos_y"] = hip.y
        features["body_pos_z"] = hip.z
        
        # Body orientation
        orientation = BodyTracker.calculate_body_orientation(body_landmarks)
        features["body_rotation_x"] = orientation["x"] / 90.0  # Normalize to approximately -1 to 1
        features["body_rotation_y"] = orientation["y"] / 90.0
        features["body_rotation_z"] = orientation["z"] / 90.0
        
        # Calculate depth movement (towards/away from camera)
        if prev_landmarks and time_delta > 0:
            # Use multiple points for more stable depth calculation
            landmarks_to_use = [
                mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value,
                mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value,
                mp.solutions.pose.PoseLandmark.LEFT_HIP.value,
                mp.solutions.pose.PoseLandmark.RIGHT_HIP.value
            ]
            
            curr_z_values = [body_landmarks.landmark[idx].z for idx in landmarks_to_use]
            prev_z_values = [prev_landmarks.landmark[idx].z for idx in landmarks_to_use]
            
            curr_avg_z = sum(curr_z_values) / len(curr_z_values)
            prev_avg_z = sum(prev_z_values) / len(prev_z_values)
            
            depth_velocity = (curr_avg_z - prev_avg_z) / time_delta
            # Filter tiny movements and normalize to a reasonable range
            if abs(depth_velocity) < 0.01:
                depth_velocity = 0.0
            norm_velocity = np.clip(depth_velocity / 0.5, -1.0, 1.0)
            features["body_depth_velocity"] = norm_velocity
        
        # Extract arm positions relative to body
        left_wrist_idx = mp.solutions.pose.PoseLandmark.LEFT_WRIST.value
        right_wrist_idx = mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value
        left_shoulder_idx = mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value
        right_shoulder_idx = mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value
        
        left_wrist = body_landmarks.landmark[left_wrist_idx]
        right_wrist = body_landmarks.landmark[right_wrist_idx]
        left_shoulder = body_landmarks.landmark[left_shoulder_idx]
        right_shoulder = body_landmarks.landmark[right_shoulder_idx]
        
        # Calculate arm extension (0 = arms down, 1 = arms fully extended)
        left_arm_ext_y = (left_shoulder.y - left_wrist.y)
        right_arm_ext_y = (right_shoulder.y - right_wrist.y)
        
        # Normalize to approximately 0-1 range
        left_arm_extension = min(max(left_arm_ext_y * 2.0, 0.0), 1.0)
        right_arm_extension = min(max(right_arm_ext_y * 2.0, 0.0), 1.0)
        
        features["left_arm_extension"] = left_arm_extension
        features["right_arm_extension"] = right_arm_extension
        
        return features
    
    def extract_hand_relationship_features(self, left_hand, right_hand) -> Dict[str, Any]:
        """Extract features that describe relationships between hands.
        
        Args:
            left_hand: MediaPipe left hand landmarks
            right_hand: MediaPipe right hand landmarks
            
        Returns:
            Dictionary of hand relationship features
        """
        from tracking.hand_tracker import HandLandmark
        
        features = {}
        
        # Calculate distance between hands (wrist to wrist)
        left_wrist = left_hand.landmark[HandLandmark.WRIST]
        right_wrist = right_hand.landmark[HandLandmark.WRIST]
        
        hand_distance = GeometryUtils.calculate_distance_3d(left_wrist, right_wrist)
        # Normalize to approximately 0-1 range based on typical arm span
        norm_distance = min(hand_distance / 0.5, 1.0)
        features["hands_distance"] = norm_distance
        
        # Calculate "hand mirroring" - how symmetrically the hands are positioned
        mirroring_x = 1.0 - min(abs(left_wrist.x - (1.0 - right_wrist.x)) * 5.0, 1.0)
        mirroring_y = 1.0 - min(abs(left_wrist.y - right_wrist.y) * 5.0, 1.0)
        
        features["hands_mirroring_x"] = mirroring_x
        features["hands_mirroring_y"] = mirroring_y
        
        return features
    
    def extract_face_hand_relationship_features(self, face_landmarks, left_hand, right_hand) -> Dict[str, Any]:
        """Extract features describing relationships between hands and face.
        
        Args:
            face_landmarks: MediaPipe face mesh landmarks
            left_hand: MediaPipe left hand landmarks (can be None)
            right_hand: MediaPipe right hand landmarks (can be None)
            
        Returns:
            Dictionary of face-hand relationship features
        """
        from tracking.hand_tracker import HandLandmark
        
        features = {}
        
        # Use nose tip as reference point for face
        nose_tip_idx = 4
        nose_tip = face_landmarks.landmark[nose_tip_idx]
        
        # Calculate distances between face and hands
        if left_hand:
            left_wrist = left_hand.landmark[HandLandmark.WRIST]
            left_hand_face_dist = GeometryUtils.calculate_distance_3d(nose_tip, left_wrist)
            # Normalize to approximately 0-1 range
            norm_left_dist = min(left_hand_face_dist / 0.5, 1.0)
            features["left_hand_face_distance"] = norm_left_dist
            
            # Check if left hand is near face (could be used for gestures like "thinking" or "talking")
            features["left_hand_near_face"] = 1.0 if left_hand_face_dist < 0.2 else 0.0
        
        if right_hand:
            right_wrist = right_hand.landmark[HandLandmark.WRIST]
            right_hand_face_dist = GeometryUtils.calculate_distance_3d(nose_tip, right_wrist)
            # Normalize to approximately 0-1 range
            norm_right_dist = min(right_hand_face_dist / 0.5, 1.0)
            features["right_hand_face_distance"] = norm_right_dist
            
            # Check if right hand is near face
            features["right_hand_near_face"] = 1.0 if right_hand_face_dist < 0.2 else 0.0
        
        return features
    
    def calculate_derivatives(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate velocity and acceleration for continuous features.
        
        Args:
            features: Current frame features
            
        Returns:
            Dictionary of velocity and acceleration features
        """
        derivatives = {}
        
        # Only process if we have previous features to compare with
        if not self.prev_features:
            return derivatives
            
        # Time delta since last frame
        time_delta = time.time() - self.prev_time
        if time_delta <= 0:
            return derivatives
            
        # Calculate velocities for relevant features
        for key, value in features.items():
            if key in self.prev_features and isinstance(value, (int, float)):
                # Only calculate for numerical values
                try:
                    velocity = (value - self.prev_features[key]) / time_delta
                    derivatives[f"{key}_velocity"] = velocity
                except (TypeError, ValueError):
                    pass
                    
        return derivatives