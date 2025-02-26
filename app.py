# app.py

import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
from typing import Dict, List, Tuple, Optional, Any

from config import FINAL_WIDTH, FINAL_HEIGHT, WEBCAM_HEIGHT, DEFAULT_IP, DEFAULT_ABLETON_PORT, DEFAULT_TD_PORT
from image_utils import ImageUtils
from tracking.face_tracker import FaceTracker
from tracking.hand_tracker import HandTracker
from tracking.body_tracker import BodyTracker
from data_processors.feature_extractor import FeatureExtractor
from integration.ableton_controller import AbletonController
from integration.touchdesigner_client import TouchDesignerClient


class TrackingApp:
    """
    Main application class for tracking and controlling audio/visual systems.
    Combines video tracking with Ableton Live and TouchDesigner integration.
    """
    def __init__(self, config_path: Optional[str] = None, debug_display: bool = False):
        """
        Initialize the tracking application with optional configuration path.
        
        Args:
            config_path: Path to configuration file
            debug_display: Flag to enable visual display of all tracking metrics
        """
        # Debug display flag
        self.debug_display = debug_display
        
        # Initialize tracking components
        self.initialize_trackers()
        
        # Initialize external integrations
        self.initialize_integrations(config_path)
        
        # Initialize webcam
        self.setup_webcam()
        
        # Performance tracking
        self.prev_time = time.time()
        self.frame_count = 0
        self.fps = 0
        
        # Tracking state
        self.is_playing = False
        
        # Initialize tracking data structure
        self.tracking_data = {
            "face": None,
            "hands": {
                "left": None,
                "right": None
            },
            "body": None,
            "features": {},
            "timestamp": 0
        }

    def initialize_trackers(self):
        """Initialize all tracking components."""
        # Initialize MediaPipe components
        self.face_tracker = FaceTracker()
        self.hand_tracker = HandTracker()
        self.body_tracker = BodyTracker()
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor()
        
        # Get drawing utilities for preview
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def initialize_integrations(self, config_path: Optional[str] = None):
        """Initialize external system integrations."""
        # Initialize Ableton Live integration
        self.ableton = AbletonController(
            ip=DEFAULT_IP, 
            port=DEFAULT_ABLETON_PORT,
            config_path=config_path
        )
        
        # Initialize TouchDesigner integration
        self.td_client = TouchDesignerClient(
            ip=DEFAULT_IP, 
            port=DEFAULT_TD_PORT
        )

    def setup_webcam(self):
        """Set up webcam capture."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open webcam.")

    def update_tracking_data(self):
        """Update the main tracking_data structure with all processed results."""
        self.tracking_data["timestamp"] = time.time()
        
        # Check if we have any tracking data to process
        has_tracking_data = any([
            self.tracking_data["face"] is not None,
            self.tracking_data["hands"]["left"] is not None,
            self.tracking_data["hands"]["right"] is not None,
            self.tracking_data["body"] is not None
        ])
        
        # Extract meaningful features from raw tracking data
        if has_tracking_data:
            self.tracking_data["features"] = self.feature_extractor.process(
                face_data=self.tracking_data["face"],
                left_hand=self.tracking_data["hands"]["left"],
                right_hand=self.tracking_data["hands"]["right"],
                body_data=self.tracking_data["body"]
            )

    def process_frame(self, frame_rgb):
        """Process a single frame with all trackers."""
        # Process with face tracker
        face_results = self.face_tracker.process(frame_rgb)
        if face_results.multi_face_landmarks:
            self.tracking_data["face"] = face_results.multi_face_landmarks[0]
        else:
            self.tracking_data["face"] = None
        
        # Process with hand tracker
        hands_results = self.hand_tracker.process(frame_rgb)
        self.tracking_data["hands"]["left"] = None
        self.tracking_data["hands"]["right"] = None
        
        if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
            for hand_landmarks, handedness in zip(
                hands_results.multi_hand_landmarks, 
                hands_results.multi_handedness
            ):
                # Identify hand as left or right
                hand_type = handedness.classification[0].label
                
                if hand_type == "Left":
                    self.tracking_data["hands"]["left"] = hand_landmarks
                else:  # Right
                    self.tracking_data["hands"]["right"] = hand_landmarks
        
        # Process with body tracker
        body_results = self.body_tracker.process(frame_rgb)
        if body_results.pose_landmarks:
            self.tracking_data["body"] = body_results.pose_landmarks
        else:
            self.tracking_data["body"] = None

    def draw_landmarks(self, frame):
        """Draw landmarks on the preview frame."""
        # Draw face mesh if available
        if self.tracking_data["face"] is not None:
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=self.tracking_data["face"],
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
        
        # Draw hand landmarks if available
        for hand_type, hand_landmarks in self.tracking_data["hands"].items():
            if hand_landmarks is not None:
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=mp.solutions.hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Draw body landmarks if available
        if self.tracking_data["body"] is not None:
            self.mp_drawing.draw_landmarks(
                frame,
                self.tracking_data["body"],
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

    def draw_debug_info(self, frame):
        """Draw debug information on the frame showing all tracked features."""
        if not self.debug_display:
            return
            
        features = self.tracking_data["features"]
        if not features:
            return
            
        # Set up text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        text_color = (0, 255, 0)  # Green
        bg_color = (0, 0, 0)  # Black background
        padding = 5
        line_height = 20
        
        # Create categorized feature groups
        feature_groups = {
            "Left Hand": {},
            "Right Hand": {},
            "Face": {},
            "Body": {},
            "Relationships": {}
        }
        
        # Categorize features
        for key, value in features.items():
            if isinstance(value, (int, float)):
                if "left" in key and "hand" in key:
                    feature_groups["Left Hand"][key] = value
                elif "right" in key and "hand" in key:
                    feature_groups["Right Hand"][key] = value
                elif "face" in key:
                    feature_groups["Face"][key] = value
                elif "body" in key or "arm" in key:
                    feature_groups["Body"][key] = value
                elif "distance" in key or "mirroring" in key or "near" in key:
                    feature_groups["Relationships"][key] = value
        
        # Draw features by category
        y_offset = 30
        x_offset = 10
        max_width = frame.shape[1] - 20
        col_width = max_width // 2
        
        # Function to draw text with background
        def draw_text_with_bg(frame, text, x, y):
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            cv2.rectangle(frame, (x - padding, y - text_size[1] - padding), 
                         (x + text_size[0] + padding, y + padding), bg_color, -1)
            cv2.putText(frame, text, (x, y), font, font_scale, text_color, font_thickness)
            return text_size[1] + 2 * padding
        
        # Draw each category
        for group_idx, (category, category_features) in enumerate(feature_groups.items()):
            if not category_features:
                continue
            
            # Determine column
            col = group_idx % 2
            if group_idx > 0 and group_idx % 2 == 0:
                y_offset = 30  # Reset Y for new row
            
            x = x_offset + col * col_width
            
            # Draw category title
            title = f"{category} Features:"
            height = draw_text_with_bg(frame, title, x, y_offset)
            y_offset += height + 5
            
            # Draw features
            for key, value in sorted(category_features.items()):
                # Format the value
                if abs(value) < 0.001:
                    value_str = "0.000"
                else:
                    value_str = f"{value:.3f}"
                
                # Clean up the key name
                display_key = key.replace("_", " ").title()
                
                # Create feature text
                feature_text = f"{display_key}: {value_str}"
                
                height = draw_text_with_bg(frame, feature_text, x, y_offset)
                y_offset += height
                
                # Check if we need to move to next column
                if y_offset > frame.shape[0] - 30:
                    col += 1
                    y_offset = 30
                    x = x_offset + col * col_width
                    if col >= 2:  # Reset if we exceed columns
                        break
                        
            # Add space between categories
            y_offset += 15
            
        # Draw connection status
        connection_y = frame.shape[0] - 30
        if self.is_playing:
            status_text = "Ableton: Connected & Playing"
            status_color = (0, 255, 0)  # Green
        else:
            status_text = "Ableton: Connected (Not Playing)"
            status_color = (0, 255, 255)  # Yellow
            
        cv2.putText(frame, status_text, (10, connection_y), font, 0.6, status_color, 2)
        
        # Draw controls hint
        controls_text = "ESC: Exit | SPACE: Toggle Debug Display"
        cv2.putText(frame, controls_text, (10, connection_y + 20), font, 0.5, (255, 255, 255), 1)

    def send_data_to_touchdesigner(self):
        """Send all relevant tracking data to TouchDesigner."""
        # Send raw landmarks
        if self.tracking_data["face"] is not None:
            self.td_client.send_face_data(self.tracking_data["face"])
        
        for hand_type, hand_data in self.tracking_data["hands"].items():
            if hand_data is not None:
                self.td_client.send_hand_data(hand_type, hand_data)
            
        if self.tracking_data["body"] is not None:
            self.td_client.send_body_data(self.tracking_data["body"])
        
        # Send extracted features
        self.td_client.send_features(self.tracking_data["features"])

    def update_ableton(self):
        """Update Ableton based on tracking data and features."""
        features = self.tracking_data["features"]
        
        # Check hand presence
        left_hand_present = self.tracking_data["hands"]["left"] is not None
        right_hand_present = self.tracking_data["hands"]["right"] is not None
        any_hand_present = left_hand_present or right_hand_present
        
        if any_hand_present:
            self.handle_ableton_playback(True)
            self.update_ableton_track_mutes(left_hand_present, right_hand_present)
            self.update_ableton_parameters(features)
        else:
            self.handle_ableton_playback(False)

    def handle_ableton_playback(self, should_play: bool):
        """Handle Ableton playback state."""
        if should_play and not self.is_playing:
            # Start playback
            self.start_ableton_clips()
            self.ableton.start_song_playback()
            self.is_playing = True
        elif not should_play and self.is_playing:
            # Stop playback
            self.ableton.stop_song_playback()
            self.is_playing = False

    def start_ableton_clips(self):
        """Start clips in Ableton across tracks."""
        for track_name, track in self.ableton.tracks.items():
            if track.clips:
                first_clip_idx = track.clips[0].index
                self.ableton.play_clip(track.index, first_clip_idx)

    def update_ableton_track_mutes(self, left_hand_present: bool, right_hand_present: bool):
        """Update track mutes based on hand presence."""
        # Update track muting based on which hands are present
        self.ableton.set_track_mute("other", 0 if left_hand_present else 1)
        self.ableton.set_track_mute("vocal", 0 if left_hand_present else 1)
        self.ableton.set_track_mute("Bass", 0 if right_hand_present else 1)
        self.ableton.set_track_mute("drums", 0 if right_hand_present else 1)

    def update_ableton_parameters(self, features: Dict[str, Any]):
        """Update Ableton parameters based on extracted features."""
        # Map features to Ableton parameters
        feature_mappings = {
            "hand_pinch_left": lambda val: self.ableton.update_rack_macro("space_rack", val),
            "hand_pinch_right": lambda val: self.ableton.update_rack_macro("distort_rack", val),
            "hand_height_left": lambda val: self.ableton.update_track_volume("vocal", val),
            "hand_height_right": lambda val: self.ableton.update_track_volume("Bass", val),
            "body_rotation_y": lambda val: self.ableton.set_track_pan("other", (val + 1) / 2)
        }
        
        # Apply each feature mapping if the feature exists
        for feature_name, mapping_func in feature_mappings.items():
            if feature_name in features:
                mapping_func(features[feature_name])

    def calculate_fps(self):
        """Calculate and update FPS."""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.prev_time
        
        if elapsed > 1.0:  # Update FPS every second
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.prev_time = current_time

    def run(self):
        """Main application loop."""
        try:
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Flip horizontally for more intuitive mirroring
                frame = cv2.flip(frame, 1)
                
                # Convert for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame with all trackers
                self.process_frame(frame_rgb)
                
                # Draw landmarks on preview
                self.draw_landmarks(frame)
                
                # Process all tracking data to extract features
                self.update_tracking_data()
                
                # Draw debug information if enabled
                self.draw_debug_info(frame)
                
                # Send data to TouchDesigner
                self.send_data_to_touchdesigner()
                
                # Update Ableton based on tracking data
                self.update_ableton()
                
                # Calculate FPS
                self.calculate_fps()
                
                # Display FPS on preview
                cv2.putText(
                    frame, f"FPS: {self.fps:.1f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
                
                # Resize for preview
                preview = ImageUtils.resize_and_crop(frame, FINAL_WIDTH, WEBCAM_HEIGHT)
                
                # Show preview window
                cv2.imshow('Tracking Preview', preview)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    break
                elif key == 32:  # SPACE key
                    self.debug_display = not self.debug_display
                    
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, shutting down...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources before exiting."""
        self.cap.release()
        cv2.destroyAllWindows()
        print("Application shutdown complete")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Motion tracking and control application')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug display of tracking metrics')
    return parser.parse_args()


def main():
    """Application entry point."""
    args = parse_arguments()
    app = TrackingApp(config_path=args.config, debug_display=args.debug)
    app.run()


if __name__ == '__main__':
    main()