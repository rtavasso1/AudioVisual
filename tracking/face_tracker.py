import mediapipe as mp
import numpy as np
from typing import List, Tuple

class FaceTracker:
    def __init__(self, 
                static_image_mode=False, 
                max_num_faces=1, 
                refine_landmarks=True, 
                min_detection_confidence=0.5, 
                min_tracking_confidence=0.5):
        """Initialize the face mesh processor with MediaPipe"""
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            refine_landmarks=refine_landmarks,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def process(self, image: np.ndarray):
        """Process an image and return face landmarks"""
        return self.face_mesh.process(image)
    
    @staticmethod
    def extract_points(face_landmarks) -> np.ndarray:
        """Extract 3D points from face landmarks"""
        points_3d = []
        for lm in face_landmarks.landmark:
            x_3d = (lm.x - 0.5) * 3.0
            y_3d = -(lm.y - 0.5) * 2.0
            z_3d = -lm.z * 2.0 + 1.5
            points_3d.append([x_3d, y_3d, z_3d])
        return np.array(points_3d, dtype=np.float32)
    
    @staticmethod
    def densify_points(points: np.ndarray, connections: List[Tuple[int, int]], 
                        num_interpolations: int = 2) -> np.ndarray:
        """Create additional points by interpolating between connected landmarks"""
        densified = list(points)
        for i, j in connections:
            if i < len(points) and j < len(points):
                p1 = points[i]
                p2 = points[j]
                for k in range(1, num_interpolations + 1):
                    alpha = k / (num_interpolations + 1)
                    interp_point = (1 - alpha) * p1 + alpha * p2
                    densified.append(interp_point)
        return np.array(densified, dtype=np.float32)