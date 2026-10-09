# AirBeat (NotAMusicSkank)

A gesture-controlled music maker for macOS using the webcam.

- **Right Hand**: Air drumming (two-finger downward hits trigger drum sounds, velocity scaling, tap tempo or hand height BPM control, optional quantization)
- **Left Hand**: Gesture-driven GUI (left index fingertip cursor, pinch-to-click with hysteresis, dwell-to-select 1.0s fallback with circular progress ring, controls Key, Scale, Backing Style, and Play/Pause)

See [PROJECT.md](PROJECT.md) for full architecture and development details.

## 🖐️ Gesture Controls

### Left Hand (Menu & Musical Settings)
* **Index Fingertip Cursor**: Point with your left index finger to move the on-screen cursor over buttons.
* **Pinch to Click**: Bring thumb and index fingertips together to click buttons with hysteresis debounce.
* **Dwell to Select**: Hover over any button for 1.0 second without pinching to trigger dwell selection. A winding circular progress ring tracks your dwell countdown.
* **Buttons Available**:
  - `BACKING TRACK`: Start / Stop generative backing track playback.
  - `KEY SELECTOR`: Cycle root key (C through B).
  - `SCALE SELECTOR`: Cycle scale (Major, Minor, Pentatonic, Dorian).
  - `STYLE SELECTOR`: Cycle backing style (Lo-Fi Chill, Synthwave 80s, Funk Groove, None).

### Right Hand (Air Drumming & Rhythm)
* **Strike Motion**: Swift downward strikes of your index and middle fingertips trigger drum hits.
* **Drum Mapping Modes** (Press `M` to toggle):
  - `finger`: Index = Kick, Middle = Snare.
  - `zones`: Spatial horizontal zones (Left = Kick, Middle = Snare, Right = Hi-Hat).
* **BPM Controls** (Press `B` to toggle):
  - `tap`: Dynamic tap tempo averaged from your last 4 hit intervals.
  - `height`: Right-hand height mapped smoothly to 60–180 BPM.
* **Quantization** (Press `G` to toggle): None, 1/8 note, or 1/16 note beat grid.

## 🚀 Quick Start

1. **Activate Python 3.11 virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   python main.py
   ```

4. **Run unit tests:**
   ```bash
   python test_gui.py
   ```
