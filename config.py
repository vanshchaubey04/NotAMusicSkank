"""Configuration settings for AirBeat."""

# Camera Settings
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30
MIRROR_CAMERA = True  # Flip horizontally for intuitive selfie mirror view

# Visual Colors (BGR format for OpenCV)
COLOR_BG_DARK = (20, 20, 24)
COLOR_PRIMARY = (255, 128, 0)      # Neon Blue/Cyan in BGR
COLOR_SECONDARY = (0, 215, 255)    # Amber/Gold in BGR
COLOR_ACCENT = (147, 20, 255)      # Deep Pink/Purple in BGR
COLOR_SUCCESS = (80, 200, 120)     # Emerald Green in BGR
COLOR_TEXT = (245, 245, 245)
COLOR_TEXT_MUTED = (160, 160, 160)
COLOR_HIT_FLASH = (0, 255, 255)    # Yellow flash on drum trigger

# Musical Settings
DEFAULT_BPM = 120
MIN_BPM = 60
MAX_BPM = 200

MUSICAL_KEYS = ["C", "D", "E", "F", "G", "A", "B"]
DEFAULT_KEY = "C"

SCALES = {
    "Major": [0, 2, 4, 5, 7, 9, 11],
    "Minor": [0, 2, 3, 5, 7, 8, 10],
    "Pentatonic": [0, 2, 4, 7, 9],
    "Blues": [0, 3, 5, 6, 7, 10],
    "Dorian": [0, 2, 3, 5, 7, 9, 10]
}
DEFAULT_SCALE = "Major"

BACKING_TRACKS = [
    "None",
    "Acoustic Groove",
    "Lo-Fi Chill",
    "Synthwave 80s",
    "Funk Foundation"
]
DEFAULT_BACKING_TRACK = "None"

# Drum Hit Detection Configuration
DRUM_SOUNDS = ["Kick", "Snare", "Hi-Hat", "Crash"]

# Hit detection parameters
DRUM_VELOCITY_THRESHOLD = 1.2     # Downward y-velocity (norm units/sec) to arm a strike
DRUM_COOLDOWN_MS = 120            # Per-finger strike cooldown in milliseconds
DRUM_VELOCITY_SMOOTHING = 0.6     # Velocity EMA smoothing factor (0.0=sluggish, 1.0=raw)
DRUM_DECEL_RATIO = 0.60           # Deceleration fraction of peak velocity to trigger hit
DRUM_FLASH_DURATION_MS = 150      # Visual screen flash duration in milliseconds
DRUM_MAX_ARMED_DURATION_MS = 160  # Maximum time in armed state before forcing strike eval

# Audio Engine Settings
AUDIO_SAMPLE_RATE = 44100
AUDIO_BUFFER_SIZE = 256           # Low-latency buffer size (256 samples ~ 5.8ms)
AUDIO_NUM_CHANNELS = 16           # Polyphonic simultaneous playback channels
SOUNDS_DIR = "assets/sounds"      # Directory for custom wav samples

# Drum Mapping & Velocity Dynamics
# "finger": index finger = kick, middle finger = snare
# "zones":  horizontal zones (left = kick, middle = snare, right = hi-hat)
DRUM_MAPPING_MODE = "finger"
DRUM_ZONE_BOUNDARIES = (0.60, 0.80)  # (left_cutoff, right_cutoff) in normalized screen X
DRUM_MIN_VOLUME = 0.35            # Minimum volume at trigger threshold
DRUM_MAX_VOLUME = 1.0             # Maximum volume for hard strikes
DRUM_MAX_VELOCITY = 3.2           # Velocity saturation point for max volume

# Metronome / Master Clock Settings
METRONOME_ENABLED = True
METRONOME_CLICK_ENABLED = True    # Audible metronome woodblock/click
METRONOME_CLICK_VOLUME = 0.45     # Click volume (0.0 to 1.0)

# BPM Control Modes (Right Hand)
# "tap":    Tap tempo calculated from the average of the last 4 intervals between hits
# "height": Right hand height mapped to BPM 60-180 with exponential smoothing
BPM_CONTROL_MODE = "tap"
BPM_MIN = 60
BPM_MAX = 180
BPM_TAP_HISTORY_COUNT = 4         # Average of last 4 hit intervals
BPM_HEIGHT_MIN_Y = 0.20           # Normalized Y for BPM 180 (higher hand)
BPM_HEIGHT_MAX_Y = 0.80           # Normalized Y for BPM 60 (lower hand)
BPM_HEIGHT_SMOOTHING = 0.08       # Smoothing factor for height-based tempo (0.01 to 0.2)

# Hit Quantization Settings
# "none": Immediate hit playback with zero delay
# "1/8":  Quantize hits to the nearest 1/8 note grid
# "1/16": Quantize hits to the nearest 1/16 note grid
QUANTIZE_MODE = "none"

