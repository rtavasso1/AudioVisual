import socket
import json
import time
from typing import Dict, Any, Optional

class TouchDesignerClient:
    """Client for sending tracking data to TouchDesigner via UDP."""
    
    def __init__(self, ip: str = "127.0.0.1", port: int = 7000):
        """Initialize TouchDesigner client with IP and port.
        
        Args:
            ip: The IP address of the TouchDesigner instance
            port: The UDP port to send data to
        """
        self.ip = ip
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.enabled = True
        
    def send_data(self, channel: str, data: Dict[str, Any]) -> bool:
        """Send JSON data to TouchDesigner on the specified channel.
        
        Args:
            channel: String identifier for the data type
            data: Dictionary of data to send
            
        Returns:
            True if send was successful, False otherwise
        """
        if not self.enabled:
            return False
            
        try:
            # Create message with channel prefix for TD routing
            message = {
                "channel": channel,
                "timestamp": time.time(),
                "data": data
            }
            
            # Convert to JSON and send
            json_data = json.dumps(message).encode('utf-8')
            self.socket.sendto(json_data, (self.ip, self.port))
            return True
        except Exception as e:
            print(f"Error sending data to TouchDesigner: {e}")
            return False
    
    def send_face_data(self, face_landmarks) -> bool:
        """Send face landmark data to TouchDesigner.
        
        Args:
            face_landmarks: MediaPipe face mesh landmarks
            
        Returns:
            True if send was successful, False otherwise
        """
        if not face_landmarks:
            return False
            
        # Extract landmark positions into a flat list for easier processing in TD
        # Format: [x1, y1, z1, x2, y2, z2, ...]
        positions = []
        for landmark in face_landmarks.landmark:
            positions.extend([landmark.x, landmark.y, landmark.z])
            
        return self.send_data("face", {
            "landmark_count": len(face_landmarks.landmark),
            "positions": positions
        })
    
    def send_hand_data(self, hand_type: str, hand_landmarks) -> bool:
        """Send hand landmark data to TouchDesigner.
        
        Args:
            hand_type: "left" or "right"
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            True if send was successful, False otherwise
        """
        if not hand_landmarks:
            return False
            
        # Extract landmark positions into a flat list
        positions = []
        for landmark in hand_landmarks.landmark:
            positions.extend([landmark.x, landmark.y, landmark.z])
            
        return self.send_data(f"hand_{hand_type}", {
            "landmark_count": len(hand_landmarks.landmark),
            "positions": positions
        })
    
    def send_body_data(self, body_landmarks) -> bool:
        """Send body landmark data to TouchDesigner.
        
        Args:
            body_landmarks: MediaPipe pose landmarks
            
        Returns:
            True if send was successful, False otherwise
        """
        if not body_landmarks:
            return False
            
        # Extract landmark positions into a flat list
        positions = []
        for landmark in body_landmarks.landmark:
            positions.extend([landmark.x, landmark.y, landmark.z])
            
        return self.send_data("body", {
            "landmark_count": len(body_landmarks.landmark),
            "positions": positions
        })
    
    def send_features(self, features: Dict[str, Any]) -> bool:
        """Send extracted features to TouchDesigner.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            True if send was successful, False otherwise
        """
        if not features:
            return False
            
        return self.send_data("features", features)