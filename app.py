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
                # Note: MediaPipe returns "Left"/"Right" based on image perspective
                # Since we're flipping the image horizontally, we need to flip handedness too
                hand_label = handedness.classification[0].label
                hand_type = "Right" if hand_label == "Left" else "Left"
                
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
        """Create and update a separate debug information window."""
        if not self.debug_display:
            # If debug window exists but debug is disabled, close it
            if hasattr(self, 'debug_window_created') and self.debug_window_created:
                cv2.destroyWindow('Debug Info')
                self.debug_window_created = False
                if hasattr(self, 'mouse_callback_set'):
                    self.mouse_callback_set = False
                return
        
        # Define all trackable features by category
        feature_categories = {
            "Left Hand": [
                "hand_pinch_left", 
                "hand_height_left", 
                "hand_openness_left",
                "hand_depth_velocity_left"
            ],
            "Right Hand": [
                "hand_pinch_right", 
                "hand_height_right", 
                "hand_openness_right",
                "hand_depth_velocity_right"
            ],
            "Face": [
                "face_rot_x",
                "face_rot_y",
                "face_rot_z",
                "face_mouth_openness",
                "face_eye_openness"
            ],
            "Body": [
                "body_rotation_y",
                "body_pos_y",
                "left_arm_extension",
                "right_arm_extension",  # Added right arm extension
                "body_depth_velocity"
            ],
            "Relationships": [
                "hands_distance",
                "hands_mirroring_x",
                "hands_mirroring_y",
                "left_hand_near_face",
                "right_hand_near_face"  # Added right hand near face
            ]
        }
        
        # Get current feature values
        features = self.tracking_data["features"]
        
        # Create a blank image for debug info
        debug_width = 600
        debug_height = 800
        debug_image = np.zeros((debug_height, debug_width, 3), dtype=np.uint8)
        
        # Set up text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        normal_color = (200, 200, 200)  # Light gray
        na_color = (0, 0, 255)  # Red for N/A values
        title_color = (100, 255, 100)  # Green for titles
        bg_color = (50, 50, 50)  # Dark gray for category headers
        highlight_color = (100, 100, 255)  # Highlight color for clickable areas
        padding = 8
        
        # Track collapsible states and y-positions of headers if not already initialized
        if not hasattr(self, 'category_collapsed'):
            self.category_collapsed = {category: False for category in feature_categories.keys()}
            self.debug_window_created = False
            self.header_y_positions = {}  # Store y-positions of headers for click detection
        
        # Function to draw text with background
        def draw_text_with_bg(image, text, x, y, color, bg=None, clickable=False):
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            if bg is not None:
                # Draw background rectangle
                cv2.rectangle(image, 
                            (x - padding, y - text_size[1] - padding), 
                            (x + text_size[0] + padding, y + padding), 
                            bg, -1)
                
                # Add border for clickable items
                if clickable:
                    cv2.rectangle(image, 
                                (x - padding, y - text_size[1] - padding), 
                                (x + text_size[0] + padding, y + padding), 
                                highlight_color, 1)
            
            cv2.putText(image, text, (x, y), font, font_scale, color, font_thickness)
            return text_size[1] + 2 * padding, text_size[0] + 2 * padding  # Return height and width
        
        # Draw title
        y_offset = 40
        title_font_scale = 0.8
        cv2.putText(debug_image, "Tracking Debug Information", 
                (20, y_offset), font, title_font_scale, 
                (255, 255, 255), 2)
        y_offset += 40
        
        # Draw connection status
        if self.is_playing:
            status_text = "Ableton: Connected & Playing"
            status_color = (0, 255, 0)  # Green
        else:
            status_text = "Ableton: Connected (Not Playing)"
            status_color = (0, 255, 255)  # Yellow
        
        height, _ = draw_text_with_bg(debug_image, status_text, 20, y_offset, status_color)
        y_offset += height + 10
        
        # Clear previous header positions
        self.header_y_positions = {}
        
        # Draw each category
        for category, feature_keys in feature_categories.items():
            # Draw category header (clickable)
            header_text = f"[{'+' if self.category_collapsed[category] else '-'}] {category}"
            
            # Store this header's y-position for click detection
            self.header_y_positions[category] = y_offset
            
            # Draw header with highlight to show it's clickable
            height, width = draw_text_with_bg(
                debug_image, header_text, 20, y_offset, 
                title_color, bg_color, clickable=True
            )
            
            y_offset += height + 5
            
            # If category is expanded, show features
            if not self.category_collapsed[category]:
                for key in feature_keys:
                    # Format the key name for display
                    display_key = key.replace("_", " ").title()
                    
                    # Get feature value if available
                    if key in features and features[key] is not None:
                        value = features[key]
                        if isinstance(value, (int, float)):
                            if abs(value) < 0.001:
                                value_str = "0.000"
                            else:
                                value_str = f"{value:.3f}"
                        else:
                            value_str = str(value)
                        color = normal_color
                    else:
                        value_str = "n/a"
                        color = na_color
                    
                    feature_text = f"  {display_key}: {value_str}"
                    height, _ = draw_text_with_bg(debug_image, feature_text, 40, y_offset, color)
                    y_offset += height
                
                # Add space after category
                y_offset += 10
            else:
                # For collapsed category, just add a small space
                y_offset += 10
        
        # Draw info about how to interact
        y_offset = debug_height - 60
        instruction_color = (200, 200, 100)
        cv2.putText(debug_image, "Click highlighted category headers to expand/collapse", 
                (20, y_offset), font, 0.5, instruction_color, 1)
        cv2.putText(debug_image, "ESC: Exit | SPACE: Toggle Debug Window", 
                (20, y_offset + 25), font, 0.5, instruction_color, 1)
        
        # Display the debug window
        cv2.imshow('Debug Info', debug_image)
        self.debug_window_created = True
        
        # Define mouse callback function outside of the condition
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Check each category header position
                for category, header_y in self.header_y_positions.items():
                    # Check if click is within the header area (with some margin)
                    header_height = 30  # Approximate header height
                    if 20 <= x <= 580 and header_y - 15 <= y <= header_y + 15:
                        # Toggle the category's collapsed state
                        self.category_collapsed[category] = not self.category_collapsed[category]
                        break
        
        # Handle mouse clicks for collapsing/expanding categories
        if not hasattr(self, 'mouse_callback_set') or not self.mouse_callback_set:
            cv2.setMouseCallback('Debug Info', mouse_callback)
            self.mouse_callback_set = True
        
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