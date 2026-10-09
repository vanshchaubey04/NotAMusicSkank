"""Hand tracking module for AirBeat using MediaPipe.

Wraps MediaPipe Hands, returning smoothed landmarks for left and right hands separately.
Correctly handles left/right swap for mirrored camera feeds, applies a 1€ (One Euro) filter
for jitter reduction, and provides real-time debug visualization with FPS counter.
"""

from dataclasses import dataclass
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# MediaPipe hand landmark connection indices
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm base connections
    (5, 9), (9, 13), (13, 17)
]

FINGERTIP_INDICES = [4, 8, 12, 16, 20]


class OneEuroFilter:
    """1€ Filter: Adaptive low-pass filter for real-time jitter reduction.

    Dynamically adjusts the cutoff frequency based on movement velocity:
    - Low velocity (subtle gestures / hovering): Low cutoff -> filters out high-frequency jitter.
    - High velocity (fast drum strikes / swipes): High cutoff -> eliminates lag.
    """

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 0.08,
        d_cutoff: float = 1.0,
    ) -> None:
        """Initialize One Euro Filter.

        Args:
            min_cutoff: Minimum cutoff frequency in Hz for low-speed jitter suppression.
            beta: Speed coefficient. Higher values reduce lag during fast movements.
            d_cutoff: Cutoff frequency for derivative calculation.
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self._x_prev: Optional[np.ndarray] = None
        self._dx_prev: Optional[np.ndarray] = None
        self._last_time: Optional[float] = None

    def _alpha(self, dt: float, cutoff: np.ndarray | float) -> np.ndarray | float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        """Filter input array with timestamp.

        Args:
            x: Input landmark coordinates array of shape (N, D).
            timestamp: Current timestamp in seconds (defaults to time.time()).

        Returns:
            Filtered landmark coordinates array of shape (N, D).
        """
        x_val = np.asarray(x, dtype=np.float32)
        if timestamp is None:
            timestamp = time.time()

        if self._last_time is None or (timestamp - self._last_time) <= 0:
            dt = 1.0 / 30.0
        else:
            dt = timestamp - self._last_time

        # If gap between frames is too large (> 0.4s), reset filter
        if dt > 0.4:
            self.reset()
            dt = 1.0 / 30.0

        self._last_time = timestamp

        if self._x_prev is None:
            self._x_prev = x_val.copy()
            self._dx_prev = np.zeros_like(x_val)
            return self._x_prev.copy()

        # Compute derivative (speed)
        dx = (x_val - self._x_prev) / dt
        alpha_d = self._alpha(dt, self.d_cutoff)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self._dx_prev
        self._dx_prev = dx_hat

        # Dynamic cutoff frequency based on velocity
        speed = np.abs(dx_hat)
        cutoff = self.min_cutoff + self.beta * speed

        # Filter the position signal
        alpha = self._alpha(dt, cutoff)
        x_hat = alpha * x_val + (1.0 - alpha) * self._x_prev
        self._x_prev = x_hat

        return x_hat.copy()

    def reset(self) -> None:
        """Reset internal filter state."""
        self._x_prev = None
        self._dx_prev = None
        self._last_time = None


@dataclass
class HandData:
    """Structured container for single-hand detection and tracking results."""

    handedness: str  # "Left" or "Right" (physical perspective of user)
    confidence: float
    landmarks: np.ndarray  # (21, 3) smoothed normalized coordinates [0..1]
    pixel_landmarks: np.ndarray  # (21, 2) smoothed pixel (x, y) coordinates
    raw_landmarks: np.ndarray  # (21, 3) raw unsmoothed normalized coordinates
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, width, height) in pixels


class HandTracker:
    """Wrapper around MediaPipe Hands with handedness correction and 1€ smoothing."""

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        mirrored: bool = True,
        use_one_euro_filter: bool = True,
        filter_min_cutoff: float = 1.5,
        filter_beta: float = 0.15,
    ) -> None:
        """Initialize HandTracker.

        Args:
            max_num_hands: Maximum number of hands to detect concurrently (1 or 2).
            min_detection_confidence: MediaPipe detection confidence threshold.
            min_tracking_confidence: MediaPipe tracking confidence threshold.
            mirrored: True if input video feed is horizontally flipped (selfie mirror view).
            use_one_euro_filter: True to apply 1€ filter for landmark jitter smoothing.
            filter_min_cutoff: 1€ filter minimum cutoff frequency.
            filter_beta: 1€ filter speed coefficient.
        """
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mirrored = mirrored
        self.use_one_euro_filter = use_one_euro_filter

        # Separate filters for left and right hands
        self._filters: Dict[str, OneEuroFilter] = {
            "Left": OneEuroFilter(min_cutoff=filter_min_cutoff, beta=filter_beta),
            "Right": OneEuroFilter(min_cutoff=filter_min_cutoff, beta=filter_beta),
        }
        self._last_seen: Dict[str, float] = {"Left": 0.0, "Right": 0.0}

        # FPS calculation state
        self._prev_frame_time: float = 0.0
        self._fps_smoothed: float = 0.0

        # MediaPipe handles
        self._hands = None
        self._mp_hands = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize MediaPipe Hands resources."""
        if self._initialized and self._hands is not None:
            return True

        try:
            import mediapipe as mp  # type: ignore

            self._mp_hands = mp.solutions.hands
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                model_complexity=1,
            )
            self._initialized = True
            return True
        except Exception as e:
            print(f"[HandTracker] MediaPipe initialization error: {e}")
            self._initialized = False
            return False

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> Tuple[Optional[HandData], Optional[HandData]]:
        """Process a video frame and extract separate left and right hand landmarks.

        Handles mirroring inversion:
        - When mirrored=True (selfie camera), MediaPipe's "Left" classification
          corresponds to the user's physical RIGHT hand, and vice versa.

        Args:
            frame: OpenCV BGR frame.
            timestamp: Frame timestamp in seconds (optional).

        Returns:
            Tuple of (left_hand, right_hand) as Optional[HandData].
        """
        if not self._initialized and not self.initialize():
            return None, None

        if timestamp is None:
            timestamp = time.time()

        # Update FPS
        if self._prev_frame_time > 0:
            dt = timestamp - self._prev_frame_time
            if dt > 0:
                current_fps = 1.0 / dt
                self._fps_smoothed = (
                    0.9 * self._fps_smoothed + 0.1 * current_fps
                    if self._fps_smoothed > 0
                    else current_fps
                )
        self._prev_frame_time = timestamp

        h, w = frame.shape[:2]

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self._hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        left_hand: Optional[HandData] = None
        right_hand: Optional[HandData] = None

        if results.multi_hand_landmarks and results.multi_handedness:
            detected_candidates: List[Tuple[str, float, np.ndarray]] = []

            for hand_lms, handedness_info in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                mp_label = handedness_info.classification[0].label
                confidence = float(handedness_info.classification[0].score)

                # Correct handedness for mirrored camera feeds:
                # MediaPipe assumes a 3rd-person observer. In selfie/mirrored view,
                # the user's physical right hand looks like a left hand to the model.
                if self.mirrored:
                    user_handedness = "Right" if mp_label == "Left" else "Left"
                else:
                    user_handedness = mp_label

                raw_coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark],
                    dtype=np.float32,
                )
                detected_candidates.append((user_handedness, confidence, raw_coords))

            # Disambiguate if two hands are detected with the same label
            if len(detected_candidates) == 2 and detected_candidates[0][0] == detected_candidates[1][0]:
                # In mirrored view, Left hand is on the left side of frame (smaller x)
                # and Right hand is on the right side of frame (larger x)
                h0_wrist_x = detected_candidates[0][2][0, 0]
                h1_wrist_x = detected_candidates[1][2][0, 0]
                if h0_wrist_x < h1_wrist_x:
                    left_idx, right_idx = 0, 1
                else:
                    left_idx, right_idx = 1, 0

                detected_candidates[left_idx] = (
                    "Left",
                    detected_candidates[left_idx][1],
                    detected_candidates[left_idx][2],
                )
                detected_candidates[right_idx] = (
                    "Right",
                    detected_candidates[right_idx][1],
                    detected_candidates[right_idx][2],
                )

            for label, conf, raw in detected_candidates:
                # Apply 1€ filter for jitter reduction
                if self.use_one_euro_filter:
                    smoothed = self._filters[label].filter(raw, timestamp)
                else:
                    smoothed = raw.copy()

                self._last_seen[label] = timestamp

                # Compute pixel coordinates
                pixel_coords = np.zeros((21, 2), dtype=np.int32)
                pixel_coords[:, 0] = np.clip(smoothed[:, 0] * w, 0, w - 1).astype(np.int32)
                pixel_coords[:, 1] = np.clip(smoothed[:, 1] * h, 0, h - 1).astype(np.int32)

                # Compute bounding box
                x_min, y_min = np.min(pixel_coords, axis=0)
                x_max, y_max = np.max(pixel_coords, axis=0)
                pad_x = int(0.04 * w)
                pad_y = int(0.04 * h)
                bbox_x = max(0, x_min - pad_x)
                bbox_y = max(0, y_min - pad_y)
                bbox_w = min(w, x_max + pad_x) - bbox_x
                bbox_h = min(h, y_max + pad_y) - bbox_y

                hand_obj = HandData(
                    handedness=label,
                    confidence=conf,
                    landmarks=smoothed,
                    pixel_landmarks=pixel_coords,
                    raw_landmarks=raw,
                    bbox=(bbox_x, bbox_y, bbox_w, bbox_h),
                )

                if label == "Left":
                    left_hand = hand_obj
                else:
                    right_hand = hand_obj

        # Reset filter if hand has been missing for > 0.4s
        for label, last_t in self._last_seen.items():
            if timestamp - last_t > 0.4:
                self._filters[label].reset()

        return left_hand, right_hand

    @property
    def fps(self) -> float:
        """Current smoothed frames-per-second."""
        return self._fps_smoothed

    def draw_debug(
        self,
        frame: np.ndarray,
        left_hand: Optional[HandData],
        right_hand: Optional[HandData],
        show_fps: bool = True,
        show_connections: bool = True,
        show_hud: bool = True,
    ) -> np.ndarray:
        """Render debug window overlays: landmarks, skeletons, labels, and FPS.

        Args:
            frame: OpenCV BGR frame.
            left_hand: Detected Left Hand data or None.
            right_hand: Detected Right Hand data or None.
            show_fps: Whether to render the FPS counter.
            show_connections: Whether to draw skeleton bones between landmarks.
            show_hud: Whether to render the top header bar and status badges.

        Returns:
            Frame with annotations drawn.
        """
        # Styling parameters
        # Left Hand: Electric Blue / Cyan
        color_left_bone = (255, 190, 0)
        color_left_joint = (255, 120, 0)
        color_left_tip = (255, 255, 100)
        color_left_tag = (200, 100, 0)

        # Right Hand: Emerald Green / Lime
        color_right_bone = (80, 220, 120)
        color_right_joint = (40, 255, 140)
        color_right_tip = (0, 255, 255)
        color_right_tag = (40, 160, 80)

        hands_to_draw = []
        if left_hand is not None:
            hands_to_draw.append(
                (left_hand, color_left_bone, color_left_joint, color_left_tip, color_left_tag)
            )
        if right_hand is not None:
            hands_to_draw.append(
                (right_hand, color_right_bone, color_right_joint, color_right_tip, color_right_tag)
            )

        for hand, c_bone, c_joint, c_tip, c_tag in hands_to_draw:
            pts = hand.pixel_landmarks

            # 1. Draw Skeleton Bones
            if show_connections:
                for idx_a, idx_b in HAND_CONNECTIONS:
                    pt_a = tuple(pts[idx_a])
                    pt_b = tuple(pts[idx_b])
                    cv2.line(frame, pt_a, pt_b, c_bone, 2, cv2.LINE_AA)

            # 2. Draw Joints
            for i, pt in enumerate(pts):
                pt_tuple = tuple(pt)
                if i in FINGERTIP_INDICES:
                    # Fingertips: larger pulsating circle
                    cv2.circle(frame, pt_tuple, 7, c_tip, -1, cv2.LINE_AA)
                    cv2.circle(frame, pt_tuple, 9, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.circle(frame, pt_tuple, 4, c_joint, -1, cv2.LINE_AA)

            # 3. Draw Bounding Box & Pill Label
            bx, by, bw, bh = hand.bbox
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), c_bone, 1, cv2.LINE_AA)

            # Pill Header Tag
            label_text = f"{hand.handedness} Hand ({int(hand.confidence * 100)}%)"
            tag_y = max(24, by - 8)
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

            # Tag background
            cv2.rectangle(
                frame,
                (bx, tag_y - th - 6),
                (bx + tw + 14, tag_y + 4),
                c_tag,
                -1,
            )
            cv2.putText(
                frame,
                label_text,
                (bx + 7, tag_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 4. Top Status Header & FPS HUD
        if show_hud:
            h, w = frame.shape[:2]
            header_h = 44
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, header_h), (18, 18, 22), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Title
        cv2.putText(
            frame,
            "AirBeat Hand Tracker",
            (16, 28),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            (255, 160, 50),
            1,
            cv2.LINE_AA,
        )

        # Hand detection badges
        left_status = "Left: ON" if left_hand else "Left: --"
        left_color = color_left_joint if left_hand else (120, 120, 120)
        cv2.putText(
            frame,
            left_status,
            (250, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            left_color,
            2 if left_hand else 1,
            cv2.LINE_AA,
        )

        right_status = "Right: ON" if right_hand else "Right: --"
        right_color = color_right_joint if right_hand else (120, 120, 120)
        cv2.putText(
            frame,
            right_status,
            (370, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            right_color,
            2 if right_hand else 1,
            cv2.LINE_AA,
        )

        # Filter info
        filter_label = "Filter: 1€ Filter (Active)" if self.use_one_euro_filter else "Filter: Off"
        cv2.putText(
            frame,
            filter_label,
            (490, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # FPS Counter
        if show_fps:
            fps_val = self._fps_smoothed
            fps_text = f"FPS: {fps_val:.1f}"
            (fw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
            cv2.putText(
                frame,
                fps_text,
                (w - fw - 20, 29),
                cv2.FONT_HERSHEY_DUPLEX,
                0.65,
                (0, 255, 180),
                1,
                cv2.LINE_AA,
            )

        return frame

    def release(self) -> None:
        """Release MediaPipe resources."""
        if self._hands is not None:
            self._hands.close()
            self._hands = None
        self._initialized = False


def run_debug_viewer(
    camera_index: int = 0,
    max_frames: Optional[int] = None,
) -> None:
    """Run standalone debug viewer demonstrating hand tracking, smoothing, and FPS."""
    print("[AirBeat] Initializing HandTracker debug viewer...")
    tracker = HandTracker(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        mirrored=True,
        use_one_euro_filter=True,
    )

    if not tracker.initialize():
        print("❌ [AirBeat] Failed to initialize MediaPipe Hands.")
        return

    # Attempt opening camera with AVFoundation backend
    cap = None
    if hasattr(cv2, "CAP_AVFOUNDATION"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"❌ [AirBeat] Could not open webcam at index {camera_index}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "AirBeat - Hand Tracker Debug (1€ Filter & FPS)"
    window_supported = True
    frame_count = 0

    print("✅ [AirBeat] Tracker running. Press 'q' or ESC in window to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠️ [AirBeat] Failed to grab frame.")
                break

            frame_count += 1

            # Mirror horizontally for selfie view
            frame = cv2.flip(frame, 1)

            # Process frame and extract left & right hand landmarks separately
            left_hand, right_hand = tracker.process_frame(frame)

            # Draw debug overlay with landmarks, "Left"/"Right" labels, and FPS
            annotated_frame = tracker.draw_debug(
                frame, left_hand, right_hand, show_fps=True
            )

            if window_supported:
                try:
                    cv2.imshow(window_name, annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                except cv2.error as e:
                    print(f"[AirBeat] Window display unavailable: {e}")
                    window_supported = False

            if max_frames is not None and frame_count >= max_frames:
                print(f"[AirBeat] Reached frame limit: {max_frames}")
                break

    finally:
        cap.release()
        if window_supported:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        tracker.release()
        print(f"✅ [AirBeat] Completed {frame_count} frames. Resources released.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AirBeat Hand Tracker Debug Viewer")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--test-frames", type=int, default=None, help="Stop after N frames")
    args = parser.parse_args()

    run_debug_viewer(camera_index=args.camera_index, max_frames=args.test_frames)
