"""Hand tracking module for AirBeat using MediaPipe.

Tracks left and right hands, extracts landmarks, and identifies gestures
for drum triggering (right hand) and GUI control (left hand).
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class HandTracker:
    """Wrapper around MediaPipe Hands for detecting hands and extracting gestures."""

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """Initialize the hand tracker.

        Args:
            max_num_hands: Maximum number of hands to detect concurrently.
            min_detection_confidence: Confidence threshold for initial detection.
            min_tracking_confidence: Confidence threshold for landmark tracking.
        """
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self._hands = None
        self._mp_hands = None
        self._mp_draw = None

    def initialize(self) -> bool:
        """Lazy-initialize MediaPipe resources.

        Returns:
            bool: True if initialization was successful.
        """
        try:
            import mediapipe as mp  # type: ignore

            self._mp_hands = mp.solutions.hands
            self._mp_draw = mp.solutions.drawing_utils
            self._hands = self._mp_hands.Hands(
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            return True
        except ImportError:
            return False

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Detect and classify hands in the given frame.

        Args:
            frame: OpenCV BGR image frame.

        Returns:
            Tuple of (left_hand_data, right_hand_data).
            Each hand data contains landmarks and metadata, or None if not detected.
        """
        # Scaffold stub: will process RGB frame with self._hands.process()
        # and parse multi_hand_landmarks & multi_handedness.
        return None, None

    def draw_landmarks(self, frame: np.ndarray, hand_data: Dict[str, Any]) -> None:
        """Draw landmarks and connections onto the frame for debugging / HUD."""
        # Scaffold stub
        pass

    def release(self) -> None:
        """Release MediaPipe resources."""
        if self._hands is not None:
            self._hands.close()
            self._hands = None
