# AirBeat: Gesture-Controlled Music Maker

**AirBeat** is a macOS desktop application that turns your webcam into an interactive, touchless musical instrument and performance station using real-time computer vision.

---

## 🎯 Vision & Core Mechanics

AirBeat tracks both hands simultaneously via the webcam and separates performance roles:

### 1. Right Hand: Air Drumming & Rhythm
* **Two-Finger Hits**: Strike gestures (using index and middle fingers) trigger dynamic drum samples (Kick, Snare, Hi-Hat, Toms, Cymbals).
* **Hit Detection & Velocity**: Tracks downward acceleration and finger positions relative to virtual spatial hit zones.
* **BPM & Tempo Control**: Detects rhythmic air-drumming cadence to dynamically calculate and adjust playback BPM (tap tempo).

### 2. Left Hand: Gesture-Driven UI & Accompaniment
* **Floating On-Screen GUI**: A gesture-controlled HUD overlaid directly onto the camera feed.
* **Key Selection**: Select root musical keys (e.g., C, D, E, F, G, A, B).
* **Scale / Mode Selection**: Switch musical scales (Major, Minor, Pentatonic, Dorian, Blues, etc.).
* **Backing Track & Chord Progression**: Choose and launch backing loops, synth chords, or basslines that stay in sync with the right-hand tempo.

---

## 🏗️ Project Architecture

```
NotAMusicSoftware/
├── PROJECT.md          # Project roadmap, design, and architecture
├── requirements.txt    # Project dependencies
├── config.py           # Configuration constants, layout, keys, scales, sound mapping
├── hand_tracker.py     # MediaPipe hand tracking & landmark extraction
├── drum_engine.py      # Air drumming hit detection, gesture logic & BPM calculation
├── audio.py            # Audio engine (pygame.mixer) & sound bank management
├── gui.py              # Real-time HUD and gesture-driven UI overlay
└── main.py             # Application entry point & camera loop with macOS permission handling
```

### Module Responsibilities

* **`config.py`**:
  * Global constants: camera index, frame dimensions, target FPS, color palette (BGR).
  * Musical presets: available keys, scales, backing track lists, default BPM.
  * Spatial trigger boundaries and hit sensitivity thresholds.

* **`hand_tracker.py`**:
  * Wrapper around MediaPipe Hands (`mp.solutions.hands`).
  * Distinguishes Left Hand vs. Right Hand (with mirror-mode calibration).
  * Landmark smoothing and spatial velocity calculations.

* **`drum_engine.py`**:
  * Evaluates right-hand fingertip trajectory for strike velocity and hit registration.
  * Prevents double-triggers using debounce/cooldown timers.
  * Calculates running BPM from strike intervals.

* **`audio.py`**:
  * Initializes low-latency audio via `pygame.mixer`.
  * Loads and triggers drum one-shots (Kick, Snare, Hi-Hat, etc.).
  * Plays looping backing tracks synchronized to the selected key and BPM.

* **`gui.py`**:
  * Renders visual feedback directly onto OpenCV frames.
  * Left-hand selection dial/menu with hover/pinch activation.
  * Right-hand visual drum pads with hit animation feedback.
  * Status banner displaying current Key, Scale, Backing Track, and BPM.

* **`main.py`**:
  * Initializes the video stream with macOS `AVFoundation` backend support.
  * Handles macOS camera permissions gracefully.
  * Main event loop integrating tracker, engines, GUI, and audio.

---

## 🍎 macOS Camera Permissions

On macOS, camera access is governed by Transparency, Consent, and Control (TCC):
1. When running for the first time, macOS will prompt to grant Camera access to Terminal / IDE / Python.
2. If access is denied or revoked, navigate to:
   **System Settings > Privacy & Security > Camera** and enable the toggle for your terminal application or IDE.
3. AirBeat uses OpenCV with the `cv2.CAP_AVFOUNDATION` backend for native, low-latency video capture on macOS.

---

## 🚀 Getting Started

1. **Create and activate Python 3.11 virtual environment:**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the camera test / app:**
   ```bash
   python main.py
   ```
