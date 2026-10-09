"""Gesture-driven GUI and HUD overlay for AirBeat.

Handles left-hand interactive menu (Key, Scale, Backing Track selection)
and right-hand drum pad visual zones and hit animations.
"""

from typing import Any, Dict, List, Optional
import cv2  # type: ignore
import numpy as np

from config import (
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_ACCENT,
    COLOR_SUCCESS,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    COLOR_BG_DARK,
    MUSICAL_KEYS,
    SCALES,
    BACKING_TRACKS,
    DEFAULT_KEY,
    DEFAULT_SCALE,
    DEFAULT_BACKING_TRACK,
    DEFAULT_BPM,
)


class OverlayGUI:
    """Renders the real-time HUD and gesture-controlled menus onto OpenCV frames."""

    def __init__(self) -> None:
        self.selected_key: str = DEFAULT_KEY
        self.selected_scale: str = DEFAULT_SCALE
        self.selected_backing_track: str = DEFAULT_BACKING_TRACK
        self.current_bpm: int = DEFAULT_BPM

        # Active menu category on left side: "Key", "Scale", "Track"
        self.active_category: str = "Key"

    def process_left_hand(self, left_hand_data: Optional[Dict[str, Any]]) -> None:
        """Process left hand position and pinch/hover gestures to navigate menus.

        Args:
            left_hand_data: Landmark data from HandTracker.
        """
        if left_hand_data is None:
            return
        # Scaffold stub: hover/pinch detection for selecting items

    def draw_hud(
        self,
        frame: np.ndarray,
        camera_ok: bool = True,
        status_message: str = "",
    ) -> np.ndarray:
        """Draw the status banner and controls onto the frame.

        Args:
            frame: OpenCV BGR frame.
            camera_ok: Whether camera stream is active and operational.
            status_message: Optional status or diagnostic message.

        Returns:
            Frame with HUD drawn.
        """
        h, w, _ = frame.shape

        # Top Header Bar
        header_height = 50
        overlay = frame.copy()
        cv2.rectangle(
            overlay, (0, 0), (w, header_height), COLOR_BG_DARK, -1
        )
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Title
        cv2.putText(
            frame,
            "AirBeat",
            (20, 34),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            COLOR_PRIMARY,
            2,
            cv2.LINE_AA,
        )

        # Status Info (Key, Scale, Track, BPM)
        status_text = (
            f"Key: {self.selected_key}  |  Scale: {self.selected_scale}  |  "
            f"Track: {self.selected_backing_track}  |  BPM: {self.current_bpm}"
        )
        cv2.putText(
            frame,
            status_text,
            (170, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )

        # Left Hand GUI zone indicator
        self._draw_left_menu_scaffold(frame, h)

        # Right Hand Drum zones indicator
        self._draw_drum_zones_scaffold(frame, w, h)

        # Extra status message or camera diagnostic banner
        if status_message:
            cv2.putText(
                frame,
                status_message,
                (20, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                COLOR_SECONDARY,
                1,
                cv2.LINE_AA,
            )

        return frame

    def _draw_left_menu_scaffold(self, frame: np.ndarray, h: int) -> None:
        """Render scaffold guide for left hand menu zone."""
        cv2.putText(
            frame,
            "[Left Hand: Menu Controls]",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_ACCENT,
            1,
            cv2.LINE_AA,
        )

    def _draw_drum_zones_scaffold(self, frame: np.ndarray, w: int, h: int) -> None:
        """Render scaffold guide for right hand drum hit zones."""
        pass
