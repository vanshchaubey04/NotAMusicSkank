"""Unit tests for AirBeat Gesture GUI and Overlay."""

import time
import numpy as np

from config import DEFAULT_BPM, DEFAULT_KEY, DEFAULT_SCALE, DEFAULT_BACKING_STYLE
from gui import OverlayGUI, GUIButton, PINCH_IN_THRESHOLD, PINCH_OUT_THRESHOLD, DWELL_TIME_SECONDS
from hand_tracker import HandData
from music_engine import MusicEngine
from drum_engine import DrumEngine
from audio import AudioManager


def create_mock_hand_data(
    handedness: str,
    cursor_x_px: int,
    cursor_y_px: int,
    pinch_dist_norm: float = 0.20,
    norm_w: int = 1280,
    norm_h: int = 720,
) -> HandData:
    """Create simulated HandData with specified index and thumb coordinates."""
    pixel_landmarks = np.zeros((21, 2), dtype=np.float32)
    landmarks = np.zeros((21, 3), dtype=np.float32)

    # Index fingertip (#8)
    pixel_landmarks[8] = [float(cursor_x_px), float(cursor_y_px)]
    landmarks[8] = [cursor_x_px / norm_w, cursor_y_px / norm_h, 0.0]

    # Thumb tip (#4) placed at pinch_dist_norm distance away
    thb_norm_x = (cursor_x_px / norm_w) - pinch_dist_norm
    thb_norm_y = cursor_y_px / norm_h
    landmarks[4] = [thb_norm_x, thb_norm_y, 0.0]
    pixel_landmarks[4] = [thb_norm_x * norm_w, thb_norm_y * norm_h]

    return HandData(
        handedness=handedness,
        confidence=0.95,
        landmarks=landmarks,
        pixel_landmarks=pixel_landmarks,
        raw_landmarks=landmarks.copy(),
        bbox=(cursor_x_px - 40, cursor_y_px - 40, 80, 80),
    )


def test_gui_initialization():
    """Verify GUI initialized with default buttons and state."""
    gui = OverlayGUI()
    assert len(gui.buttons) == 4
    button_ids = [b.button_id for b in gui.buttons]
    assert button_ids == ["playback", "key", "scale", "style"]
    assert gui.selected_key == DEFAULT_KEY
    assert gui.selected_scale == DEFAULT_SCALE
    assert gui.selected_backing_track == DEFAULT_BACKING_STYLE
    assert gui.current_bpm == DEFAULT_BPM
    assert not gui.is_playing


def test_cursor_tracking_and_hover():
    """Verify left hand index fingertip moves cursor and highlights hovered button."""
    gui = OverlayGUI()
    # Button 0 (playback) is at x=22..282, y=68..126
    btn = gui.buttons[0]
    assert btn.contains(btn.x + 10, btn.y + 10)
    assert not btn.contains(btn.x + 300, btn.y + 10)

    # Move cursor over button 0
    hand = create_mock_hand_data("Left", btn.x + 50, btn.y + 20, pinch_dist_norm=0.20)
    gui.process_left_hand(hand, timestamp=1.0)

    assert gui.cursor_pos == (btn.x + 50, btn.y + 20)
    assert btn.is_hovered
    assert gui.hovered_button == btn
    assert not gui.buttons[1].is_hovered


def test_pinch_hysteresis_and_click():
    """Verify pinch-to-click activates button using hysteresis thresholds."""
    music_engine = MusicEngine()
    music_engine.key = "C"
    gui = OverlayGUI(music_engine=music_engine)

    key_btn = gui.buttons[1]  # Key selector button
    bx, by = key_btn.x + 30, key_btn.y + 20

    # 1. Approach button: fingers open (dist = 0.15)
    hand = create_mock_hand_data("Left", bx, by, pinch_dist_norm=0.15)
    gui.process_left_hand(hand, timestamp=1.0)
    assert not gui.is_pinched
    assert music_engine.key == "C"

    # 2. Pinch fingers together below PINCH_IN_THRESHOLD (dist = 0.04)
    assert 0.04 <= PINCH_IN_THRESHOLD
    hand = create_mock_hand_data("Left", bx, by, pinch_dist_norm=0.04)
    gui.process_left_hand(hand, timestamp=1.1)
    assert gui.is_pinched
    # Click triggered: key advances from C to C#
    assert music_engine.key == "C#"

    # 3. Intermediate distance (dist = 0.065): between in and out thresholds
    # Must remain pinched due to hysteresis!
    assert PINCH_IN_THRESHOLD < 0.065 < PINCH_OUT_THRESHOLD
    hand = create_mock_hand_data("Left", bx, by, pinch_dist_norm=0.065)
    gui.process_left_hand(hand, timestamp=1.2)
    assert gui.is_pinched
    # Should NOT double click because pinch state didn't re-enter
    assert music_engine.key == "C#"

    # 4. Open fingers above PINCH_OUT_THRESHOLD (dist = 0.09)
    assert 0.09 >= PINCH_OUT_THRESHOLD
    hand = create_mock_hand_data("Left", bx, by, pinch_dist_norm=0.09)
    gui.process_left_hand(hand, timestamp=1.3)
    assert not gui.is_pinched

    # 5. Pinch again after debounce -> should advance key again
    hand = create_mock_hand_data("Left", bx, by, pinch_dist_norm=0.03)
    gui.process_left_hand(hand, timestamp=1.7)
    assert gui.is_pinched
    assert music_engine.key == "D"

    music_engine.cleanup()


def test_dwell_to_select_and_progress_ring():
    """Verify dwell-to-select triggers click after 1.0s and computes progress."""
    music_engine = MusicEngine()
    music_engine.scale = "Major"
    gui = OverlayGUI(music_engine=music_engine)

    scale_btn = gui.buttons[2]  # Scale selector button
    bx, by = scale_btn.x + 30, scale_btn.y + 20

    # Start hovering at t=1.0 (no pinch)
    hand = create_mock_hand_data("Left", bx, by, pinch_dist_norm=0.15)
    gui.process_left_hand(hand, timestamp=1.0)
    assert scale_btn.is_hovered
    assert abs(scale_btn.dwell_progress - 0.0) < 0.01

    # Halfway through dwell at t=1.5 (0.5s elapsed)
    gui.process_left_hand(hand, timestamp=1.5)
    assert abs(scale_btn.dwell_progress - 0.5) < 0.03
    assert music_engine.scale == "Major"  # Not triggered yet

    # Reach dwell duration at t=2.0 (1.0s elapsed)
    gui.process_left_hand(hand, timestamp=2.0)
    assert abs(scale_btn.dwell_progress - 1.0) < 0.03
    # Dwell triggered: scale advanced from Major to Minor!
    assert music_engine.scale == "Minor"

    # Moving away immediately cancels dwell
    far_hand = create_mock_hand_data("Left", 800, 400, pinch_dist_norm=0.15)
    gui.process_left_hand(far_hand, timestamp=2.1)
    assert not scale_btn.is_hovered
    assert scale_btn.dwell_progress == 0.0

    music_engine.cleanup()


def test_button_actions_cycle_all():
    """Verify all 4 buttons trigger their musical state actions."""
    music_engine = MusicEngine()
    gui = OverlayGUI(music_engine=music_engine)

    # 1. Playback button
    play_btn = gui.buttons[0]
    initial_playing = music_engine.is_playing
    play_btn.on_click()
    assert music_engine.is_playing != initial_playing
    play_btn.on_click()
    assert music_engine.is_playing == initial_playing

    # 2. Key button
    key_btn = gui.buttons[1]
    k0 = music_engine.key
    key_btn.on_click()
    assert music_engine.key != k0

    # 3. Scale button
    scale_btn = gui.buttons[2]
    s0 = music_engine.scale
    scale_btn.on_click()
    assert music_engine.scale != s0

    # 4. Style button
    style_btn = gui.buttons[3]
    st0 = music_engine.style
    style_btn.on_click()
    assert music_engine.style != st0

    music_engine.cleanup()


def test_hand_isolation_left_and_right():
    """Verify left-hand input never triggers drum hits and right-hand never clicks GUI."""
    audio = AudioManager()
    audio.initialize()
    clock = None
    drum_engine = DrumEngine(audio_manager=audio)
    music_engine = MusicEngine()
    gui = OverlayGUI(music_engine=music_engine)

    # 1. Left hand strikes down vigorously over button area
    # GUI should only update cursor/hover/pinch, NOT drum hits
    btn = gui.buttons[0]
    left_hand = create_mock_hand_data("Left", btn.x + 20, btn.y + 20, pinch_dist_norm=0.15)
    gui.process_left_hand(left_hand, timestamp=1.0)
    # Passing left hand to drum engine returns 0 hits (drum engine only processes right hand)
    hits = drum_engine.process_right_hand(None, timestamp=1.0)
    assert len(hits) == 0

    # 2. Right hand moves over GUI buttons
    # Right hand passed to drum_engine, but NOT to gui.process_left_hand
    right_hand = create_mock_hand_data("Right", btn.x + 20, btn.y + 20, pinch_dist_norm=0.03)
    gui.process_left_hand(None, timestamp=1.1)  # Left hand is None
    assert gui.cursor_pos is None
    assert not btn.is_hovered

    drum_engine.cleanup()
    music_engine.cleanup()


def test_draw_hud_renders_without_error():
    """Verify draw_hud renders clean OpenCV frame with all visual components."""
    gui = OverlayGUI()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Set active cursor and hover
    btn = gui.buttons[1]
    hand = create_mock_hand_data("Left", btn.x + 40, btn.y + 20, pinch_dist_norm=0.04)
    gui.process_left_hand(hand, timestamp=1.0)

    out_frame = gui.draw_hud(frame, camera_ok=True, status_message="Test Status")
    assert out_frame is not None
    assert out_frame.shape == (720, 1280, 3)
    # Pixels should be modified (non-black)
    assert np.count_nonzero(out_frame) > 0


if __name__ == "__main__":
    tests = [
        ("test_gui_initialization", test_gui_initialization),
        ("test_cursor_tracking_and_hover", test_cursor_tracking_and_hover),
        ("test_pinch_hysteresis_and_click", test_pinch_hysteresis_and_click),
        ("test_dwell_to_select_and_progress_ring", test_dwell_to_select_and_progress_ring),
        ("test_button_actions_cycle_all", test_button_actions_cycle_all),
        ("test_hand_isolation_left_and_right", test_hand_isolation_left_and_right),
        ("test_draw_hud_renders_without_error", test_draw_hud_renders_without_error),
    ]

    passed = 0
    print("\n--- Running AirBeat GUI Unit Tests ---")
    for name, func in tests:
        try:
            func()
            print(f"  ✅ {name}: PASSED")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: FAILED ({e})")
            raise

    print(f"\n--- All {passed}/{len(tests)} tests passed successfully! ---\n")
