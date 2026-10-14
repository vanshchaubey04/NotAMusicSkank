"""AirBeat - Gesture-Controlled Music Maker.

Main entry point and camera capture loop.
Handles macOS camera permissions and initializes core subsystems.
"""

import sys
import time
import argparse
from typing import Optional, Tuple

import cv2
import numpy as np

from config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    TARGET_FPS,
    MIRROR_CAMERA,
)
from gui import OverlayGUI
from hand_tracker import HandTracker
from drum_engine import DrumEngine
from audio import AudioManager
from music_engine import MusicEngine
from looper import LiveLooper, JamRecorder
from rhythm_hero import RhythmHeroGame


def print_macos_camera_troubleshooting() -> None:
    """Print actionable troubleshooting steps for macOS camera permission errors."""
    print("\n" + "=" * 60)
    print("⚠️  macOS Camera Permission / Device Access Diagnostic")
    print("=" * 60)
    print("AirBeat could not access your webcam. Please check:")
    print("1. macOS Privacy Settings:")
    print("   Open System Settings -> Privacy & Security -> Camera.")
    print("   Ensure access is enabled for your Terminal / IDE / Python.")
    print("2. Camera In Use:")
    print("   Ensure no other application (FaceTime, Zoom, Photo Booth) is locking the camera.")
    print("3. External Camera / Device Index:")
    print("   If using an external webcam or Continuity Camera, try another index:")
    print("   python main.py --camera-index 1")
    print("=" * 60 + "\n")


def open_camera(camera_index: int = CAMERA_INDEX) -> Tuple[Optional[cv2.VideoCapture], str]:
    """Attempt to open the camera using macOS AVFoundation backend with fallback.

    Args:
        camera_index: Index of the camera to open.

    Returns:
        Tuple of (cv2.VideoCapture or None, status_message).
    """
    backends = []
    # On macOS, cv2.CAP_AVFOUNDATION is the native capture backend
    if hasattr(cv2, "CAP_AVFOUNDATION"):
        backends.append(("AVFoundation", cv2.CAP_AVFOUNDATION))
    backends.append(("Default", cv2.CAP_ANY))

    for name, backend in backends:
        print(f"[AirBeat] Attempting to open camera {camera_index} with {name} backend...")
        try:
            cap = cv2.VideoCapture(camera_index, backend)
            if cap.isOpened():
                # Configure resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

                # Test reading an initial frame
                ret, test_frame = cap.read()
                if ret and test_frame is not None and test_frame.size > 0:
                    h, w = test_frame.shape[:2]
                    print(f"✅ [AirBeat] Camera {camera_index} successfully opened via {name} ({w}x{h}).")
                    return cap, f"Camera active ({name} {w}x{h})"
                else:
                    print(f"⚠️ [AirBeat] Camera opened with {name}, but frame read failed.")
                    cap.release()
            else:
                cap.release()
        except Exception as e:
            print(f"⚠️ [AirBeat] Backend {name} error: {e}")

    return None, "Camera unavailable"


def run_camera_test(camera_index: int = CAMERA_INDEX, max_frames: Optional[int] = None) -> bool:
    """Scaffold camera verification test.

    Opens camera, verifies frame capture, draws HUD scaffold, and handles exit.

    Args:
        camera_index: Index of webcam.
        max_frames: If set, terminates after processing N frames (useful for automated testing).

    Returns:
        bool: True if camera opened and captured frames successfully.
    """
    cap, status_msg = open_camera(camera_index)
    if cap is None:
        print_macos_camera_troubleshooting()
        return False

    tracker = HandTracker(mirrored=MIRROR_CAMERA, use_one_euro_filter=True)
    tracker.initialize()
    audio = AudioManager()
    audio.initialize()
    drum_engine = DrumEngine(audio_manager=audio)
    music_engine = MusicEngine(bpm=drum_engine.bpm)
    looper = LiveLooper()
    jam_recorder = JamRecorder()
    rhythm_hero = RhythmHeroGame()

    # Connect live looper, jam session recorder, and rhythm hero to drum engine
    drum_engine.looper = looper
    drum_engine.jam_recorder = jam_recorder
    drum_engine.rhythm_game = rhythm_hero

    # Wire looper loop playback callback
    def on_looper_playback(drum_name: str, volume: float) -> None:
        audio.play_drum(drum_name, volume=volume)
        if drum_engine.ar_pads_enabled and drum_name in drum_engine.ar_pads:
            pad = drum_engine.ar_pads[drum_name]
            pad["hit_glow"] = 0.85
            pad["compression"] = 0.75

    looper.on_trigger_callback = on_looper_playback

    gui = OverlayGUI(
        music_engine=music_engine,
        audio_manager=audio,
        drum_engine=drum_engine,
        looper=looper,
        jam_recorder=jam_recorder,
        rhythm_hero=rhythm_hero,
    )

    # Wire drum hit callback to interactive tutorial action verification
    drum_engine.on_hit_callback = lambda hit: (
        gui.notify_action_event(f"hit_{hit.finger}"),
        gui.notify_action_event("hit_any")
    )

    window_name = "AirBeat - Camera Test"
    window_supported = True

    try:
        frame_count = 0
        print("\n[AirBeat] Starting video feed.")
        print("  Controls: [P] Play/Pause Backing | [K] Key | [S] Scale | [T] Style | [M] Drum Map")
        print("            [V] Drum Kit | [E] Sensitivity | [L] Beat Looper | [R] Jam Recorder")
        print("            [J] Rhythm Hero Game | [H] AR Drum Pads | [Y] 2-Hand Drums")
        print("            [B] BPM Mode | [G] Quantize | [C] Metronome Click | [D] Tutorial/Demo | [Q/ESC] Quit\n")

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠️ [AirBeat] Failed to grab frame from camera.")
                break

            frame_count += 1

            # Mirror horizontally for natural selfie perspective
            if MIRROR_CAMERA:
                frame = cv2.flip(frame, 1)

            now = time.time()

            # Process hand landmarks (left and right hands separated, 1€ smoothed)
            left_hand, right_hand = tracker.process_frame(frame, timestamp=now)

            # Clock synchronization for looper and rhythm hero
            current_beat = (
                drum_engine.clock.get_continuous_beat()
                if drum_engine.clock
                else 0.0
            )
            looper.update_clock(current_beat)
            if rhythm_hero.is_active:
                rhythm_hero.update_clock(current_beat=current_beat, bpm=drum_engine.bpm, current_time=now)

            # Process left hand GUI gestures (cursor, pinch-to-click, dwell-to-select)
            gui.process_left_hand(left_hand, music_engine=music_engine, timestamp=now)

            # Two-handed drumming: process left hand strikes when outside GUI menu
            left_hits = drum_engine.process_left_hand_drums(left_hand, timestamp=now)

            # Process drum hit detection on right hand and trigger sounds
            hits = drum_engine.process_right_hand(right_hand, timestamp=now)

            # Sync BPM to music engine
            music_engine.set_bpm(drum_engine.bpm)

            # Draw hand landmarks, bones, and labels
            frame = tracker.draw_debug(
                frame, left_hand, right_hand, show_fps=False, show_hud=False
            )

            # Draw drum tuning overlay (live velocity meters, zones, fingertip tags, flash, AR pads)
            frame = drum_engine.draw_debug_overlay(
                frame, right_hand, current_time=now
            )

            # Update GUI musical state and render HUD
            gui.update_state(
                key=music_engine.key,
                scale=music_engine.scale,
                style=music_engine.style,
                bpm=drum_engine.bpm,
                is_playing=music_engine.is_playing,
            )
            fps_val = tracker.fps
            track_tag = f"Track: {music_engine.style[:8]} ({'ON' if music_engine.is_playing else 'OFF'}) ['p']"
            status_text = (
                f"FPS: {fps_val:.1f} | {track_tag} | Key: {music_engine.key} ['k'] | "
                f"Kit: {audio.current_kit} ['v'] | Sens: {drum_engine.sensitivity_preset} ['e'] | "
                f"Looper: {looper.state} ['l'] | Rec: {'ON' if jam_recorder.is_recording else 'OFF'} ['r']"
            )
            frame = gui.draw_hud(
                frame,
                camera_ok=True,
                status_message=status_text,
            )

            # Try displaying window
            if window_supported:
                try:
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):  # ESC or 'q'
                        if key == 27 and gui.is_demo_active:
                            gui.exit_demo()
                        else:
                            print("[AirBeat] Quit signal received.")
                            break
                    elif key in (ord("d"), ord("D")):
                        gui.toggle_demo()
                    elif gui.is_demo_active and key in (ord("n"), ord("N")):
                        gui.next_demo_step()
                    elif gui.is_demo_active and key in (ord("x"), ord("X")):
                        gui.exit_demo()
                    elif gui.is_demo_active and key in (9, ord("j"), ord("J")):  # Tab or 'j'
                        gui.skip_demo_step()
                    elif gui.is_demo_active and key in (ord("["), 8):  # '[' or Backspace
                        gui.prev_demo_step()
                    elif key in (ord("p"), ord("P"), 32):  # 'p' or space
                        music_engine.toggle_playback()
                        gui.notify_action_event("toggle_track")
                    elif key in (ord("k"), ord("K")):
                        music_engine.cycle_key()
                        gui.notify_action_event("cycle_key")
                    elif key in (ord("s"), ord("S")):
                        music_engine.cycle_scale()
                        gui.notify_action_event("cycle_scale")
                    elif key in (ord("t"), ord("T")):
                        music_engine.cycle_style()
                        gui.notify_action_event("cycle_style")
                    elif key in (ord("v"), ord("V")):
                        audio.cycle_kit()
                        gui.notify_action_event("cycle_kit")
                    elif key in (ord("e"), ord("E")):
                        drum_engine.cycle_sensitivity()
                        gui.notify_action_event("sensitivity")
                    elif key in (ord("l"), ord("L")):
                        looper.toggle()
                        gui.notify_action_event("looper")
                    elif key in (ord("r"), ord("R")):
                        jam_recorder.toggle()
                        gui.notify_action_event("recorder")
                    elif key in (ord("j"), ord("J")) and not gui.is_demo_active:
                        rhythm_hero.toggle()
                        gui.notify_action_event("rhythm_game")
                    elif key in (ord("h"), ord("H")):
                        drum_engine.toggle_ar_pads()
                        gui.notify_action_event("ar_pads")
                    elif key in (ord("y"), ord("Y")):
                        drum_engine.toggle_two_hand_drums()
                    elif key in (ord("m"), ord("M")):
                        drum_engine.toggle_mapping_mode()
                    elif key in (ord("b"), ord("B")):
                        drum_engine.toggle_bpm_mode()
                        gui.notify_action_event("tempo")
                    elif key in (ord("g"), ord("G")):
                        drum_engine.toggle_quantize_mode()
                        gui.notify_action_event("tempo")
                    elif key in (ord("c"), ord("C")):
                        drum_engine.toggle_metronome_click()
                except cv2.error as cv_err:
                    print(f"[AirBeat] Window display unavailable: {cv_err}")
                    window_supported = False

            if max_frames is not None and frame_count >= max_frames:
                print(f"[AirBeat] Reached test frame limit ({max_frames} frames).")
                break

            time.sleep(0.01)

        print(f"✅ [AirBeat] Video loop completed ({frame_count} frames processed).")
        return True

    finally:
        cap.release()
        if window_supported:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        tracker.release()
        music_engine.cleanup()
        drum_engine.cleanup()
        if jam_recorder.is_recording:
            jam_recorder.stop()
        print("[AirBeat] Resources cleanly released.")


def main() -> None:
    parser = argparse.ArgumentParser(description="AirBeat: Gesture-Controlled Music Maker")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=CAMERA_INDEX,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--test-frames",
        type=int,
        default=None,
        help="Limit execution to N frames for testing",
    )
    args = parser.parse_args()

    success = run_camera_test(
        camera_index=args.camera_index, max_frames=args.test_frames
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
