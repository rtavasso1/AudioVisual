# ableton_controller.py

import time
import math
import json
import os
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import List, Tuple, Union, Any, Dict, Callable, Optional
from pythonosc import udp_client

class ClipState(Enum):
    STOPPED = 0
    PLAYING = 1
    TRIGGERED = 2

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

@dataclass
class Clip:
    name: str
    index: int
    state: ClipState = ClipState.STOPPED
    last_triggered: float = 0

@dataclass
class DeviceParameter:
    name: str
    index: int
    min_val: float = 0.0
    max_val: float = 127.0
    mapping_fn: Optional[Callable[[float], float]] = None
    current_value: float = 0.0
    smoothing_enabled: bool = False
    smoothing_method: str = "moving_average"
    smoothing_window_size: int = 5
    alpha: float = 0.2
    _value_history: deque = field(default_factory=lambda: deque(maxlen=20), init=False)

    def normalize_value(self, value: float) -> float:
        """Maps an incoming 0..1 to min_val..max_val, with optional custom mapping."""
        assert 0.0 <= value <= 1.0, f"Input must be between 0 and 1, got {value}"
        if self.mapping_fn:
            value = self.mapping_fn(value)
        return self.min_val + value * (self.max_val - self.min_val)

    def update_value(self, raw_value: float) -> float:
        normalized = self.normalize_value(raw_value)
        if not self.smoothing_enabled:
            self.current_value = normalized
            return self.current_value

        # If smoothing is enabled, choose the method
        if self.smoothing_method == "moving_average":
            self._value_history.append(normalized)
            self.current_value = sum(self._value_history) / len(self._value_history)
        elif self.smoothing_method == "exponential":
            self.current_value += self.alpha * (normalized - self.current_value)
        else:
            self.current_value = normalized
        return self.current_value

@dataclass
class Device:
    name: str
    index: int
    type: str  # "audio_rack", "midi_rack", or "other"
    parameters: Dict[str, DeviceParameter] = field(default_factory=dict)

    def ensure_parameter(self, param_name: str, index: int = 1) -> DeviceParameter:
        """Ensure a parameter exists, create it if it doesn't"""
        if param_name not in self.parameters:
            self.parameters[param_name] = DeviceParameter(
                name=param_name,
                index=index,
                min_val=0.0,
                max_val=127.0
            )
        return self.parameters[param_name]

@dataclass
class Track:
    name: str
    index: int
    devices: Dict[str, Device] = field(default_factory=dict)
    clips: List[Clip] = field(default_factory=list)
    active_clip: Optional[int] = None

    def add_clip(self, name: str, index: int) -> Clip:
        """Add a clip to the track"""
        clip = Clip(name=name, index=index)
        self.clips.append(clip)
        return clip

    def ensure_device(self, name: str, index: int, device_type: str = "audio_rack") -> Device:
        """Ensure a device exists, create it if it doesn't"""
        device_key = f"{name}_{index}"
        if device_key not in self.devices:
            self.devices[device_key] = Device(
                name=name,
                index=index,
                type=device_type
            )
        return self.devices[device_key]

    def get_device_by_index(self, index: int) -> Optional[Device]:
        """Get a device by its index"""
        for device in self.devices.values():
            if device.index == index:
                return device
        return None

    def get_rack_device(self, rack_key: str) -> Optional[Device]:
        """Get a rack device by its key name"""
        if rack_key in self.devices:
            return self.devices[rack_key]
        # Try finding a device where the name contains the rack_key
        for key, device in self.devices.items():
            if rack_key in key:
                return device
        return None


class AbletonProjectConfig:
    """Configuration class for Ableton project settings"""
    def __init__(self, config_path: Optional[str] = None):
        self.tracks: Dict[str, Dict] = {}
        self.global_racks: List[str] = []
        
        # Default configuration
        if not config_path:
            self._setup_default_config()
        else:
            self._load_config(config_path)
    
    def _setup_default_config(self):
        """Set up a default configuration with common rack names"""
        self.global_racks = ["distort_rack", "space_rack", "filter_rack"]
        self.tracks = {
            "Bass": {"index": 0},
            "Drums": {"index": 1},
            "Other": {"index": 2},
            "Vocal": {"index": 3}
        }
    
    def _load_config(self, config_path: str):
        """Load configuration from a JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.tracks = config.get("tracks", {})
                self.global_racks = config.get("global_racks", [])
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading config: {e}")
            self._setup_default_config()
    
    def save_config(self, config_path: str):
        """Save the current configuration to a JSON file"""
        config = {
            "tracks": self.tracks,
            "global_racks": self.global_racks
        }
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Config saved to {config_path}")
        except Exception as e:
            print(f"Error saving config: {e}")


class AbletonController:
    def __init__(self, ip: str, port: int, config_path: Optional[str] = None):
        """
        Initialize the Ableton Controller
        
        Args:
            ip: IP address for OSC communication
            port: Port for OSC communication
            config_path: Optional path to a JSON configuration file
        """
        self.client = udp_client.SimpleUDPClient(ip, port)
        self.config = AbletonProjectConfig(config_path)
        self.tracks: Dict[str, Track] = {}
        self._setup_tracks()
        
        self.current_scene = 0
        self.last_message_times = {}
        self.message_interval = 1.0 / 5.0  # rate-limit parameter messages to 5 messages per second
        
        # Default mapping function
        self.default_mapping_fn = lambda x: x
    
    def _setup_tracks(self):
        """Initialize tracks based on configuration"""
        for track_name, track_config in self.config.tracks.items():
            track = Track(
                name=track_name,
                index=track_config["index"]
            )
            
            # Add default clips if specified
            if "clips" in track_config:
                for clip_info in track_config["clips"]:
                    track.add_clip(clip_info["name"], clip_info["index"])
            
            # Set up default devices for each global rack type
            device_index = 0
            for rack_type in self.config.global_racks:
                device = track.ensure_device(
                    name=rack_type,
                    index=device_index,
                    device_type="audio_rack"
                )
                # Set up default macro parameter
                device.ensure_parameter("macro1", 1)
                device_index += 1
            
            self.tracks[track_name] = track

    def save_current_config(self, path: str):
        """Save the current configuration to a file"""
        self.config.save_config(path)

    #
    # --- Playback control methods ---
    #

    def start_song_playback(self):
        """Start the playback of the Ableton Live set"""
        self.client.send_message("/live/song/start_playing", [])

    def stop_song_playback(self):
        """Stop the playback of the Ableton Live set"""
        self.client.send_message("/live/song/stop_playing", [])

    def play_clip(self, track_idx: int, clip_idx: int) -> None:
        """
        Fires a specific clip in a track.
        
        Args:
            track_idx: Track index
            clip_idx: Clip index
        """
        self.client.send_message("/live/clip/fire", [track_idx, clip_idx])
        
        # Update internal track states
        for track in self.tracks.values():
            if track.index == track_idx:
                track.active_clip = clip_idx
                for clip in track.clips:
                    if clip.index == clip_idx:
                        clip.state = ClipState.PLAYING
                        clip.last_triggered = time.time()

    def stop_clip(self, track_idx: int, clip_idx: int) -> None:
        """
        Stops a specific clip in a track.
        
        Args:
            track_idx: Track index
            clip_idx: Clip index
        """
        self.client.send_message("/live/clip/stop", [track_idx, clip_idx])
        
        for track in self.tracks.values():
            if track.index == track_idx:
                for clip in track.clips:
                    if clip.index == clip_idx:
                        clip.state = ClipState.STOPPED
                if track.active_clip == clip_idx:
                    track.active_clip = None

    def stop_track(self, track_idx: int) -> None:
        """
        Stops all clips in a single track.
        
        Args:
            track_idx: Track index
        """
        self.client.send_message("/live/track/stop_all_clips", [track_idx])
        
        for track in self.tracks.values():
            if track.index == track_idx:
                for clip in track.clips:
                    clip.state = ClipState.STOPPED
                track.active_clip = None

    def play_track_clip(self, track_name: str, clip_idx: int) -> None:
        """
        Play a clip on a specific track by name.
        
        Args:
            track_name: Name of the track
            clip_idx: Clip index
        """
        if track_name in self.tracks:
            self.play_clip(self.tracks[track_name].index, clip_idx)
        else:
            print(f"Track {track_name} not found")

    #
    # --- Track-level controls ---
    #

    def set_track_mute(self, track_name: str, mute_state: int) -> None:
        """
        Mutes or unmutes a track.
        
        Args:
            track_name: Name of the track
            mute_state: 1 -> Mute, 0 -> Unmute
        """
        if track_name in self.tracks:
            track = self.tracks[track_name]
            self.client.send_message("/live/track/set/mute", [track.index, mute_state])
        else:
            print(f"Track {track_name} not found")

    def set_track_solo(self, track_name: str, solo_state: int) -> None:
        """
        Solos or un-solos a track.
        
        Args:
            track_name: Name of the track
            solo_state: 1 -> Solo, 0 -> Un-solo
        """
        if track_name in self.tracks:
            track = self.tracks[track_name]
            self.client.send_message("/live/track/set/solo", [track.index, solo_state])
        else:
            print(f"Track {track_name} not found")

    def update_track_volume(self, track_name: str, volume: float) -> None:
        """
        Sets a track's volume.
        
        Args:
            track_name: Name of the track
            volume: Volume value (0.0 to 1.0)
        """
        if track_name in self.tracks:
            track = self.tracks[track_name]
            # Map 0..1 to dB range (approximately -70dB to 0dB)
            db_value = -70.0 + (volume * 70.0)
            self.client.send_message("/live/track/set/volume", [track.index, db_value])
        else:
            print(f"Track {track_name} not found")

    def set_track_pan(self, track_name: str, pan: float) -> None:
        """
        Sets a track's panning.
        
        Args:
            track_name: Name of the track
            pan: Pan value (0.0 to 1.0, where 0.5 is center)
        """
        if track_name in self.tracks:
            track = self.tracks[track_name]
            # Map 0..1 to -1..1 pan range
            pan_value = (pan * 2.0) - 1.0
            self.client.send_message("/live/track/set/pan", [track.index, pan_value])
        else:
            print(f"Track {track_name} not found")

    def set_tempo(self, tempo: float) -> None:
        """
        Set the global BPM.
        
        Args:
            tempo: BPM value
        """
        self.client.send_message("/live/song/set/tempo", [float(tempo)])

    #
    # --- Parameter control (macros) ---
    #

    def set_parameter(self, track_index: int, device_index: int, param_index: int, value: float) -> None:
        """
        Sets a device parameter value, with rate-limiting to avoid spamming OSC.
        
        Args:
            track_index: Track index
            device_index: Device index
            param_index: Parameter index
            value: Parameter value (0.0 to 1.0)
        """
        key = (track_index, device_index, param_index)
        current_time = time.time()
        last_time = self.last_message_times.get(key, 0)
        
        if current_time - last_time < self.message_interval:
            return
            
        self.last_message_times[key] = current_time
        
        # Map 0..1 to parameter range
        self.client.send_message(
            "/live/device/set/parameter/value",
            [track_index, device_index, param_index, value]
        )

    def update_rack_macro(self, rack_key: str, value: float) -> None:
        """
        Update a macro parameter on a specific rack type across all tracks.
        
        Args:
            rack_key: Rack type key (e.g. "distort_rack")
            value: Parameter value (0.0 to 1.0)
        """
        print(f"Updating rack {rack_key} with value {value}")
        
        for track in self.tracks.values():
            rack_device = track.get_rack_device(rack_key)
            if rack_device:
                # Try to get macro1 parameter, create if it doesn't exist
                macro_param = rack_device.ensure_parameter("macro1", 1)
                smoothed_value = macro_param.update_value(value)
                
                self.set_parameter(
                    track.index,
                    rack_device.index,
                    macro_param.index,
                    smoothed_value
                )
    
    def update_all_racks(self, values_dict: Dict[str, float]) -> None:
        """
        Update multiple rack macros with a single call.
        
        Args:
            values_dict: Dictionary mapping rack keys to values
        """
        for rack_key, value in values_dict.items():
            self.update_rack_macro(rack_key, value)

    #
    # --- Utility methods ---
    #
            
    def register_track(self, name: str, index: int) -> Track:
        """
        Register a new track or update an existing one.
        
        Args:
            name: Track name
            index: Track index
            
        Returns:
            The track object
        """
        if name in self.tracks:
            # Update existing track
            self.tracks[name].index = index
        else:
            # Create new track
            self.tracks[name] = Track(name=name, index=index)
            
        # Update config
        self.config.tracks[name] = {"index": index}
        
        return self.tracks[name]
        
    def register_global_rack(self, rack_name: str) -> None:
        """
        Register a rack type as globally available.
        
        Args:
            rack_name: Name of the rack
        """
        if rack_name not in self.config.global_racks:
            self.config.global_racks.append(rack_name)
            
            # Add this rack type to all existing tracks
            for track in self.tracks.values():
                # Find the next available device index
                next_index = 0
                for device in track.devices.values():
                    if device.index >= next_index:
                        next_index = device.index + 1
                
                track.ensure_device(
                    name=rack_name,
                    index=next_index,
                    device_type="audio_rack"
                )