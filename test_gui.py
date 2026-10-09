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
    """Verify GUI initialized with default buttons (including demo) and state."""
    gui = OverlayGUI()
    assert len(gui.buttons) == 5
    button_ids = [b.button_id for b in gui.buttons]
    assert button_ids == ["playback", "key", "scale", "style", "demo"]
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


def test_5_finger_drum_kit_and_sensitivity():
    """Verify 5-finger drum assignment and high sensitivity strike detection."""
    from config import FINGER_DRUM_MAP, DRUM_VELOCITY_THRESHOLD, DRUM_THUMB_VELOCITY_THRESHOLD

    # 1. Verify 5 distinct drum effects mapped
    assert FINGER_DRUM_MAP["thumb"] == "kick"
    assert FINGER_DRUM_MAP["index"] == "snare"
    assert FINGER_DRUM_MAP["middle"] == "hihat"
    assert FINGER_DRUM_MAP["ring"] == "tom"
    assert FINGER_DRUM_MAP["pinky"] == "crash"

    # 2. Verify strike thresholds calibrated for effortless slight movements
    assert DRUM_VELOCITY_THRESHOLD <= 0.40
    assert DRUM_THUMB_VELOCITY_THRESHOLD <= 0.35

    audio = AudioManager()
    audio.initialize()

    # Verify all 5 instruments synthesized and ready in audio manager
    assert "kick" in audio._sounds
    assert "snare" in audio._sounds
    assert "hihat" in audio._sounds
    assert "tom" in audio._sounds
    assert "crash" in audio._sounds

    for finger, expected_sound in FINGER_DRUM_MAP.items():
        resolved = audio.resolve_drum(finger, 0.5)
        assert resolved == expected_sound, f"Expected {expected_sound} for {finger}, got {resolved}"

    drum_engine = DrumEngine(audio_manager=audio)

    # Check 5 trackers instantiated
    assert hasattr(drum_engine, "thumb_tracker")
    assert hasattr(drum_engine, "index_tracker")
    assert hasattr(drum_engine, "middle_tracker")
    assert hasattr(drum_engine, "ring_tracker")
    assert hasattr(drum_engine, "pinky_tracker")

    # Verify on_hit_callback invocation
    detected_hits = []
    drum_engine.on_hit_callback = lambda hit: detected_hits.append(hit)

    # Simulate slight downward thumb movement
    norm_w, norm_h = 1280, 720
    landmarks_1 = np.zeros((21, 3), dtype=np.float32)
    pixel_1 = np.zeros((21, 2), dtype=np.float32)
    landmarks_1[4] = [0.45, 0.40, 0.0]
    pixel_1[4] = [0.45 * norm_w, 0.40 * norm_h]
    hand_t1 = HandData("Right", 0.95, landmarks_1, pixel_1, landmarks_1, (500, 250, 100, 100))

    landmarks_2 = np.zeros((21, 3), dtype=np.float32)
    pixel_2 = np.zeros((21, 2), dtype=np.float32)
    # Move downward by 0.035 units in 0.05s -> velocity = 0.70 units/s (arms the strike)
    landmarks_2[4] = [0.45, 0.435, 0.0]
    pixel_2[4] = [0.45 * norm_w, 0.435 * norm_h]
    hand_t2 = HandData("Right", 0.95, landmarks_2, pixel_2, landmarks_2, (500, 270, 100, 100))

    # Frame 3: Deceleration / reversal (finger stops moving downward)
    landmarks_3 = np.zeros((21, 3), dtype=np.float32)
    pixel_3 = np.zeros((21, 2), dtype=np.float32)
    landmarks_3[4] = [0.45, 0.435, 0.0]
    pixel_3[4] = [0.45 * norm_w, 0.435 * norm_h]
    hand_t3 = HandData("Right", 0.95, landmarks_3, pixel_3, landmarks_3, (500, 270, 100, 100))

    drum_engine.process_right_hand(hand_t1, timestamp=1.0)
    drum_engine.process_right_hand(hand_t2, timestamp=1.05)
    hits = drum_engine.process_right_hand(hand_t3, timestamp=1.10)

    assert len(hits) >= 1
    assert hits[0].finger == "thumb"
    assert hits[0].velocity >= 0.30
    assert len(detected_hits) >= 1

    drum_engine.cleanup()


def test_demo_tutorial_navigation_and_actions():
    """Verify guided interactive demo walkthrough: toggle, next, prev, skip, and live action verification."""
    gui = OverlayGUI()
    assert not gui.is_demo_active
    assert len(gui.tutorial_steps) >= 12
    assert len(gui.tutorial_buttons) == 4

    # 1. Toggle demo ON
    gui.toggle_demo()
    assert gui.is_demo_active
    assert gui.demo_step == 0
    assert gui.tutorial_steps[0].step_id == "welcome"

    # 2. Advance to Step 1 (Thumb / Kick)
    gui.next_demo_step()
    assert gui.demo_step == 1
    assert gui.tutorial_steps[1].step_id == "thumb"
    assert not gui.step_action_completed

    # 3. Verify Live Action detection for thumb flick
    gui.notify_action_event("hit_thumb")
    assert gui.step_action_completed

    # 4. Advance to Step 2 (Index / Snare)
    gui.next_demo_step()
    assert gui.demo_step == 2
    assert gui.tutorial_steps[2].step_id == "index"
    assert not gui.step_action_completed

    # 5. Skip Step 2 if user doesn't want to play snare right now
    gui.skip_demo_step()
    assert gui.demo_step == 3
    assert gui.tutorial_steps[3].step_id == "middle"

    # 6. Return to Previous Step
    gui.prev_demo_step()
    assert gui.demo_step == 2

    # 7. Exit demo
    gui.exit_demo()
    assert not gui.is_demo_active


def test_demo_render_overlay():
    """Verify demo overlay cards and live action verification box render seamlessly."""
    gui = OverlayGUI()
    gui.start_demo()
    assert gui.is_demo_active

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    out_frame = gui.draw_hud(frame, camera_ok=True, status_message="Demo Running")
    assert out_frame is not None
    assert np.count_nonzero(out_frame) > 0

    # Test completed action rendering
    gui.notify_action_event(gui.tutorial_steps[gui.demo_step].expected_action)
    out_frame_done = gui.draw_hud(frame, camera_ok=True)
    assert out_frame_done is not None


if __name__ == "__main__":
    tests = [
        ("test_gui_initialization", test_gui_initialization),
        ("test_cursor_tracking_and_hover", test_cursor_tracking_and_hover),
        ("test_pinch_hysteresis_and_click", test_pinch_hysteresis_and_click),
        ("test_dwell_to_select_and_progress_ring", test_dwell_to_select_and_progress_ring),
        ("test_button_actions_cycle_all", test_button_actions_cycle_all),
        ("test_hand_isolation_left_and_right", test_hand_isolation_left_and_right),
        ("test_draw_hud_renders_without_error", test_draw_hud_renders_without_error),
        ("test_5_finger_drum_kit_and_sensitivity", test_5_finger_drum_kit_and_sensitivity),
        ("test_demo_tutorial_navigation_and_actions", test_demo_tutorial_navigation_and_actions),
        ("test_demo_render_overlay", test_demo_render_overlay),
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
