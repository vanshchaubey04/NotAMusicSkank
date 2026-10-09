# AirBeat (NotAMusicSkank)

A gesture-controlled music maker and touchless air drum studio for macOS using the webcam.

- **Right Hand (Green Hand)**: 5-Finger Air Drum Kit (every finger triggers a distinct drum sound, high sensitivity calibrated for subtle movements, velocity dynamics, tap tempo, hand height BPM control, optional quantization)
- **Left Hand (Cyan Hand)**: Gesture-driven GUI (left index fingertip cursor, pinch-to-click with hysteresis, dwell-to-select 1.0s fallback with circular progress ring, controls Key, Scale, Backing Style, Play/Pause, and Guided Tutorial)
- **Interactive Demo Mode**: Step-by-step interactive guided tutorial explaining every drum effect, gesture, and musical feature with live action detection and skip options.

See [PROJECT.md](PROJECT.md) for full architecture and development details.

## 🖐️ Gesture Controls

### Right Hand (Green Hand: 5-Piece Air Drum Kit)
Every finger of your right hand has its own dedicated drum sound:
* **👍 Thumb**: Deep 808 **Kick Drum** (punchy analog low-end foundation).
* **☝️ Index**: Acoustic **Snare Drum** (crisp wire snap).
* **🖕 Middle**: Closed **Hi-Hat Cymbal** (metallic rhythm driver).
* **💍 Ring**: Resonant **Tom Drum** (warm tone for drum fills).
* **🤙 Pinky**: Shimmering **Crash Cymbal** (long metallic wash for accents).
* **High Sensitivity**: Calibrated for effortless slight finger flicks without needing excessive physical force.
* **Dynamic Volume**: Strike velocity scales playback loudness from subtle ghost notes to loud hits.
* **BPM Controls** (Press `B` to toggle):
  - `tap`: Dynamic tap tempo averaged from your last 4 hit intervals.
  - `height`: Right-hand height mapped smoothly to 60–180 BPM.
* **Quantization** (Press `G` to toggle): None, 1/8 note, or 1/16 note beat grid.

### Left Hand (Cyan Hand: Menu & Musical Settings)
* **Index Fingertip Cursor**: Point with your left index finger to move the on-screen crosshair over buttons.
* **Pinch to Click**: Bring thumb and index fingertips together to click buttons with hysteresis debounce.
* **Dwell to Select**: Hover over any button for 1.0 second without pinching to trigger dwell selection. A winding circular progress ring tracks your dwell countdown.
* **Buttons Available**:
  - `BACKING TRACK`: Start / Stop generative backing track playback (`P`).
  - `KEY SELECTOR`: Cycle root key C through B (`K`).
  - `SCALE SELECTOR`: Cycle scale Major, Minor, Pentatonic, Dorian (`S`).
  - `STYLE SELECTOR`: Cycle backing style Lo-Fi Chill, Synthwave 80s, Funk Groove (`T`).
  - `TUTORIAL / DEMO`: Open the step-by-step interactive walkthrough (`D`).

### 🎓 Interactive Demo / Tutorial Mode
* Click **`TUTORIAL / DEMO`** or press **`D`** to launch the interactive walkthrough.
* **Guided Steps**: Explains every finger drum sound, velocity scaling, left-hand cursor/pinch/dwell, backing tracks, key transposition, and metronome modes.
* **Try It Live or Skip**: Each step has a live action detection prompt. Perform the gesture to see real-time emerald green verification, or click `[SKIP]` (`Tab`/`J`) if not working or if you want to advance immediately!
* **Keyboard Shortcuts in Demo**:
  - `[D]`: Toggle demo mode on / off
  - `[N]` or `[Space]`: Next step
  - `[Tab]` or `[J]`: Skip current step
  - `[` or `[Backspace]`: Previous step
  - `[X]` or `[ESC]`: Exit demo mode

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
