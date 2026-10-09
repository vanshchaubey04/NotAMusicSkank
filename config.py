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

# Drum Configuration
DRUM_SOUNDS = ["Kick", "Snare", "Hi-Hat", "Crash"]
DRUM_HIT_DEBOUNCE_SECONDS = 0.15
DRUM_VELOCITY_THRESHOLD = 0.05  # Normalized downward delta
