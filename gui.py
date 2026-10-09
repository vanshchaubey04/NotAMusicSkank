"""Gesture-driven GUI and full-screen HUD overlay for AirBeat.

Features:
- Left hand index fingertip (landmark #8) acts as the interactive cursor.
- Pinch-to-click: Thumb tip (#4) to index tip (#8) distance with hysteresis.
- Dwell-to-select fallback: Hovering over any button for 1.0 second triggers click.
- Circular progress ring drawn around cursor during dwell countdown.
- Interactive Buttons:
  1. Key Selector (C through B)
  2. Scale Selector (Major, Minor, Pentatonic, Dorian)
  3. Backing Track Selector (Lo-Fi Chill, Synthwave 80s, Funk Groove, None)
  4. Start / Stop Playback Toggle
- Hover highlights, click animations, and full-screen status bar.
- Strict isolation: Left-hand GUI inputs and right-hand air drumming never interfere.
"""

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    COLOR_ACCENT,
    COLOR_BG_DARK,
    COLOR_HIT_FLASH,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SUCCESS,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    DEFAULT_BACKING_STYLE,
    DEFAULT_BPM,
    DEFAULT_KEY,
    DEFAULT_SCALE,
    MUSICAL_KEYS,
    SCALES,
)


# Gesture Tuning Constants
PINCH_IN_THRESHOLD: float = 0.052    # Normalized Euclidean distance to enter pinch
PINCH_OUT_THRESHOLD: float = 0.076   # Normalized Euclidean distance to exit pinch
DWELL_TIME_SECONDS: float = 1.00     # Dwell duration to trigger fallback click
CLICK_DEBOUNCE_SECONDS: float = 0.28 # Minimum cooldown between consecutive clicks


@dataclass
class GUIButton:
    """Represents an on-screen interactive glassmorphic button."""

    button_id: str
    title: str
    x: int
    y: int
    w: int
    h: int
    get_value_text: Callable[[], str]
    on_click: Callable[[], None]
    accent_color: Tuple[int, int, int] = COLOR_PRIMARY
    is_hovered: bool = False
    dwell_progress: float = 0.0
    last_click_time: float = 0.0

    def contains(self, px: int, py: int) -> bool:
        """Check if pixel coordinate is inside button bounds."""
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h


class OverlayGUI:
    """Full-screen interactive OpenCV overlay driven by left-hand gestures."""

    def __init__(self, music_engine: Optional[Any] = None) -> None:
        self.music_engine = music_engine

        # State cache
        self.selected_key: str = DEFAULT_KEY
        self.selected_scale: str = DEFAULT_SCALE
        self.selected_backing_track: str = DEFAULT_BACKING_STYLE
        self.current_bpm: int = DEFAULT_BPM
        self.is_playing: bool = False

        # Left hand cursor state
        self.cursor_pos: Optional[Tuple[int, int]] = None
        self.cursor_norm_pos: Optional[Tuple[float, float]] = None
        self.thumb_pos: Optional[Tuple[int, int]] = None
        self.pinch_distance: float = 1.0
        self.is_pinched: bool = False
        self.last_click_timestamp: float = 0.0

        # Dwell state
        self.hovered_button: Optional[GUIButton] = None
        self.hover_start_time: float = 0.0
        self.dwell_triggered: bool = False

        # Visual click flash queue: (x, y, timestamp, color)
        self._click_flashes: List[Dict[str, Any]] = []

        # Initialize interactive buttons (left side panel)
        self.buttons: List[GUIButton] = []
        self._init_buttons()

    def _init_buttons(self) -> None:
        """Create buttons laid out ergonomically on the left edge of the screen."""
        btn_w = 260
        btn_h = 58
        start_x = 22
        start_y = 68
        gap_y = 12

        # 1. Start / Stop Playback Button
        self.buttons.append(
            GUIButton(
                button_id="playback",
                title="BACKING TRACK",
                x=start_x,
                y=start_y,
                w=btn_w,
                h=btn_h,
                get_value_text=lambda: "▶ START TRACK" if not self.is_playing else "⏹ STOP TRACK",
                on_click=self._action_toggle_playback,
                accent_color=COLOR_SUCCESS if not self.is_playing else (60, 60, 240),
            )
        )

        # 2. Key Selector Button
        self.buttons.append(
            GUIButton(
                button_id="key",
                title="KEY SELECTOR",
                x=start_x,
                y=start_y + (btn_h + gap_y),
                w=btn_w,
                h=btn_h,
                get_value_text=lambda: f"Key: [ {self.selected_key} ]",
                on_click=self._action_cycle_key,
                accent_color=COLOR_PRIMARY,
            )
        )

        # 3. Scale Selector Button
        self.buttons.append(
            GUIButton(
                button_id="scale",
                title="SCALE SELECTOR",
                x=start_x,
                y=start_y + 2 * (btn_h + gap_y),
                w=btn_w,
                h=btn_h,
                get_value_text=lambda: f"Scale: [ {self.selected_scale} ]",
                on_click=self._action_cycle_scale,
                accent_color=COLOR_SECONDARY,
            )
        )

        # 4. Backing Style Selector Button
        self.buttons.append(
            GUIButton(
                button_id="style",
                title="STYLE SELECTOR",
                x=start_x,
                y=start_y + 3 * (btn_h + gap_y),
                w=btn_w,
                h=btn_h,
                get_value_text=lambda: f"Style: [ {self.selected_backing_track} ]",
                on_click=self._action_cycle_style,
                accent_color=COLOR_ACCENT,
            )
        )

    def set_music_engine(self, engine: Any) -> None:
        """Attach music engine reference."""
        self.music_engine = engine

    # --- Button Action Handlers ---
    def _action_toggle_playback(self) -> None:
        if self.music_engine:
            self.music_engine.toggle_playback()
            self.is_playing = self.music_engine.is_playing

    def _action_cycle_key(self) -> None:
        if self.music_engine:
            self.selected_key = self.music_engine.cycle_key(1)

    def _action_cycle_scale(self) -> None:
        if self.music_engine:
            self.selected_scale = self.music_engine.cycle_scale(1)

    def _action_cycle_style(self) -> None:
        if self.music_engine:
            self.selected_backing_track = self.music_engine.cycle_style(1)

    def update_state(
        self,
        key: str,
        scale: str,
        style: str,
        bpm: int,
        is_playing: bool = False,
    ) -> None:
        """Synchronize musical state from MusicEngine."""
        self.selected_key = key
        self.selected_scale = scale
        self.selected_backing_track = style
        self.current_bpm = bpm
        self.is_playing = is_playing

        # Dynamically synchronize playback button accent and state
        for btn in self.buttons:
            if btn.button_id == "playback":
                btn.accent_color = (60, 60, 240) if is_playing else COLOR_SUCCESS

    def process_left_hand(
        self,
        left_hand_data: Optional[Any],
        music_engine: Optional[Any] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Process left hand position, pinch-to-click, and dwell selection.

        Strictly operates on left hand data to ensure complete isolation from right-hand air drumming.

        Args:
            left_hand_data: HandData instance from HandTracker (or None).
            music_engine: Optional MusicEngine reference.
            timestamp: Frame timestamp in seconds.
        """
        if music_engine is not None:
            self.music_engine = music_engine

        if timestamp is None:
            timestamp = time.time()

        if left_hand_data is None:
            # Left hand lost: reset cursor, pinch, and dwell state
            self.cursor_pos = None
            self.cursor_norm_pos = None
            self.thumb_pos = None
            self.is_pinched = False
            self.hovered_button = None
            for b in self.buttons:
                b.is_hovered = False
                b.dwell_progress = 0.0
            return

        landmarks = left_hand_data.landmarks
        pixel_landmarks = left_hand_data.pixel_landmarks

        # Left index fingertip (landmark #8) is the cursor
        idx_px = (int(pixel_landmarks[8, 0]), int(pixel_landmarks[8, 1]))
        idx_norm = (float(landmarks[8, 0]), float(landmarks[8, 1]))
        self.cursor_pos = idx_px
        self.cursor_norm_pos = idx_norm

        # Left thumb tip (landmark #4)
        thb_px = (int(pixel_landmarks[4, 0]), int(pixel_landmarks[4, 1]))
        thb_norm = (float(landmarks[4, 0]), float(landmarks[4, 1]))
        self.thumb_pos = thb_px

        # Compute normalized Euclidean distance for pinch detection
        dx = idx_norm[0] - thb_norm[0]
        dy = idx_norm[1] - thb_norm[1]
        dist = math.sqrt(dx * dx + dy * dy)
        self.pinch_distance = dist

        # -------------------------------------------------------------
        # 1. Pinch Detection with Hysteresis
        # -------------------------------------------------------------
        click_triggered = False

        if not self.is_pinched:
            # Enter pinch state
            if dist <= PINCH_IN_THRESHOLD:
                self.is_pinched = True
                if (timestamp - self.last_click_timestamp) >= CLICK_DEBOUNCE_SECONDS:
                    click_triggered = True
                    self.last_click_timestamp = timestamp
        else:
            # Exit pinch state
            if dist >= PINCH_OUT_THRESHOLD:
                self.is_pinched = False

        # -------------------------------------------------------------
        # 2. Hover & Dwell-to-Select Logic
        # -------------------------------------------------------------
        current_hover: Optional[GUIButton] = None
        cx, cy = idx_px

        for btn in self.buttons:
            if btn.contains(cx, cy):
                current_hover = btn
                btn.is_hovered = True
            else:
                btn.is_hovered = False
                btn.dwell_progress = 0.0

        if current_hover is not None:
            if self.hovered_button == current_hover:
                # Still hovering over the same button: increment dwell
                elapsed_dwell = timestamp - self.hover_start_time
                if elapsed_dwell < 0:
                    current_hover.dwell_progress = 0.0
                    self.dwell_triggered = False
                else:
                    progress = min(1.0, elapsed_dwell / DWELL_TIME_SECONDS)
                    current_hover.dwell_progress = progress

                    if progress >= 1.0 and not self.dwell_triggered:
                        if (timestamp - self.last_click_timestamp) >= CLICK_DEBOUNCE_SECONDS:
                            click_triggered = True
                            self.dwell_triggered = True
                            self.last_click_timestamp = timestamp
                            # Add a brief 0.4s pause before next dwell starts
                            self.hover_start_time = timestamp + 0.4
            else:
                # Newly hovered button: start dwell timer
                self.hovered_button = current_hover
                self.hover_start_time = timestamp
                self.dwell_triggered = False
                current_hover.dwell_progress = 0.0
        else:
            self.hovered_button = None
            self.dwell_triggered = False

        # -------------------------------------------------------------
        # 3. Execute Click Action
        # -------------------------------------------------------------
        if click_triggered:
            if current_hover is not None:
                current_hover.on_click()
                current_hover.last_click_time = timestamp
                self._register_click_flash(cx, cy, current_hover.accent_color, timestamp)
                print(f"[OverlayGUI] Button '{current_hover.button_id}' clicked (val: {current_hover.get_value_text()})")
            else:
                # Pinch in empty space: subtle visual confirmation ripple
                self._register_click_flash(cx, cy, (160, 160, 180), timestamp)

    def _register_click_flash(
        self, cx: int, cy: int, color: Tuple[int, int, int], timestamp: float
    ) -> None:
        """Add visual ripple effect at click coordinates."""
        self._click_flashes.append(
            {"x": cx, "y": cy, "color": color, "time": timestamp}
        )

    def draw_hud(
        self,
        frame: np.ndarray,
        camera_ok: bool = True,
        status_message: str = "",
    ) -> np.ndarray:
        """Render the complete full-screen overlay: top status bar, buttons, cursor, and progress ring.

        Args:
            frame: OpenCV BGR frame.
            camera_ok: Whether video feed is healthy.
            status_message: Optional bottom status message.

        Returns:
            Frame with interactive GUI rendered.
        """
        now = time.time()
        h, w = frame.shape[:2]

        # 1. Render Top Header Bar & Status
        self._render_header_bar(frame, w, camera_ok)

        # 2. Render Interactive Left-Hand Buttons
        self._render_buttons(frame, now)

        # 3. Render Click Ripple Animations
        self._render_click_flashes(frame, now)

        # 4. Render Left-Hand Cursor & Dwell Progress Ring
        if self.cursor_pos is not None:
            self._render_cursor(frame, now)

        # 5. Render Bottom Diagnostic Banner
        if status_message:
            cv2.putText(
                frame,
                status_message,
                (20, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                COLOR_SECONDARY,
                1,
                cv2.LINE_AA,
            )

        return frame

    def _render_header_bar(self, frame: np.ndarray, w: int, camera_ok: bool) -> None:
        """Draw glassmorphic top header with status badges."""
        header_h = 48
        h = frame.shape[0]
        if h >= header_h:
            sub = frame[0:header_h, 0:w]
            overlay = sub.copy()
            overlay[:] = COLOR_BG_DARK
            cv2.addWeighted(overlay, 0.85, sub, 0.15, 0, sub)

        # Title
        cv2.putText(
            frame,
            "AirBeat",
            (20, 32),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            COLOR_PRIMARY,
            2,
            cv2.LINE_AA,
        )

        # Track play badge
        play_badge = "▶ PLAY" if self.is_playing else "⏹ STOP"
        play_color = COLOR_SUCCESS if self.is_playing else (120, 120, 140)

        # Musical status info
        status_text = (
            f"Key: {self.selected_key}  |  Scale: {self.selected_scale}  |  "
            f"Track: {self.selected_backing_track}  |  BPM: {self.current_bpm}"
        )
        cv2.putText(
            frame,
            status_text,
            (155, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )

        # Playback status pill on top right
        (tw, _), _ = cv2.getTextSize(play_badge, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        pill_x = max(660, w - tw - 340)
        cv2.rectangle(frame, (pill_x - 6, 12), (pill_x + tw + 6, 36), play_color, 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            play_badge,
            (pill_x, 29),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            play_color,
            1,
            cv2.LINE_AA,
        )

        # Camera health status indicator
        cam_text = "● CAM" if camera_ok else "● NO CAM"
        cam_color = COLOR_SUCCESS if camera_ok else (60, 60, 240)
        (cw, _), _ = cv2.getTextSize(cam_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cam_x = max(pill_x + tw + 20, w - cw - 20)
        cv2.putText(
            frame,
            cam_text,
            (cam_x, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            cam_color,
            1,
            cv2.LINE_AA,
        )

    def _render_buttons(self, frame: np.ndarray, now: float) -> None:
        """Draw interactive cards on the left edge with hover and click animations."""
        h, w = frame.shape[:2]
        for btn in self.buttons:
            bx, by, bw, bh = btn.x, btn.y, btn.w, btn.h
            if by + bh > h or bx + bw > w:
                continue

            # Check click flash state
            click_elapsed = now - btn.last_click_time
            is_click_flash = click_elapsed < 0.20

            # Sub-frame glassmorphic background
            sub_frame = frame[by : by + bh, bx : bx + bw]
            overlay = sub_frame.copy()
            bg_color = (40, 40, 52) if btn.is_hovered else (22, 22, 28)
            overlay[:] = bg_color

            # Click flash overlay
            if is_click_flash:
                overlay[:] = COLOR_HIT_FLASH

            alpha = 0.88 if not btn.is_hovered else 0.94
            cv2.addWeighted(overlay, alpha, sub_frame, 1.0 - alpha, 0, sub_frame)

            # Border
            if btn.is_hovered:
                border_color = (0, 255, 200)
                border_thick = 2
            else:
                border_color = (65, 65, 80)
                border_thick = 1

            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), border_color, border_thick, cv2.LINE_AA)

            # Left accent stripe
            cv2.rectangle(
                frame,
                (bx, by),
                (bx + 4, by + bh),
                btn.accent_color if not btn.is_hovered else (0, 255, 220),
                -1,
            )

            # Title
            cv2.putText(
                frame,
                btn.title,
                (bx + 14, by + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                COLOR_TEXT_MUTED if not btn.is_hovered else (0, 255, 220),
                1,
                cv2.LINE_AA,
            )

            # Value text
            val_text = btn.get_value_text()
            val_color = COLOR_TEXT if not btn.is_hovered else (255, 255, 255)
            cv2.putText(
                frame,
                val_text,
                (bx + 14, by + 44),
                cv2.FONT_HERSHEY_DUPLEX,
                0.52,
                val_color,
                1,
                cv2.LINE_AA,
            )

            # Hover chevron indicator
            if btn.is_hovered:
                cv2.putText(
                    frame,
                    "▶",
                    (bx + bw - 22, by + 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 200),
                    1,
                    cv2.LINE_AA,
                )

    def _render_cursor(self, frame: np.ndarray, now: float) -> None:
        """Render left index fingertip cursor, pinch state, and dwell progress ring."""
        cx, cy = self.cursor_pos
        is_hover = self.hovered_button is not None
        dwell_prog = self.hovered_button.dwell_progress if self.hovered_button else 0.0

        # Draw subtle guide line between thumb tip and index cursor
        if self.thumb_pos is not None:
            tx, ty = self.thumb_pos
            line_color = COLOR_HIT_FLASH if self.is_pinched else (120, 120, 140)
            line_thick = 2 if self.is_pinched else 1
            cv2.line(frame, (cx, cy), (tx, ty), line_color, line_thick, cv2.LINE_AA)
            cv2.circle(frame, (tx, ty), 4, line_color, -1, cv2.LINE_AA)

        # Cursor color scheme
        if self.is_pinched:
            c_ring = COLOR_HIT_FLASH
            c_dot = (255, 255, 255)
            radius = 12
        elif is_hover:
            c_ring = (0, 255, 200)  # Cyan/Emerald
            c_dot = (0, 255, 200)
            radius = 14
        else:
            c_ring = COLOR_PRIMARY  # Electric Blue/Cyan
            c_dot = (255, 255, 255)
            radius = 13

        # Base cursor crosshair & circles
        cv2.circle(frame, (cx, cy), radius, c_ring, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 3, c_dot, -1, cv2.LINE_AA)

        # Crosshair ticks
        cv2.line(frame, (cx - radius - 4, cy), (cx - radius + 2, cy), c_ring, 1, cv2.LINE_AA)
        cv2.line(frame, (cx + radius - 2, cy), (cx + radius + 4, cy), c_ring, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - radius - 4), (cx, cy - radius + 2), c_ring, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy + radius - 2), (cx, cy + radius + 4), c_ring, 1, cv2.LINE_AA)

        # Dwell progress ring: winding arc from -90° to 270°
        if dwell_prog > 0.01:
            ring_radius = radius + 9
            end_angle = int(-90 + 360 * dwell_prog)
            cv2.ellipse(
                frame,
                (cx, cy),
                (ring_radius, ring_radius),
                0,
                -90,
                end_angle,
                (0, 255, 255),
                3,
                cv2.LINE_AA,
            )
            # Dwell percentage text
            pct_text = f"{int(dwell_prog * 100)}%"
            cv2.putText(
                frame,
                pct_text,
                (cx + radius + 12, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Pinch tag if pinched
        if self.is_pinched:
            cv2.putText(
                frame,
                "[PINCH CLICK]",
                (cx - 36, cy - radius - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                COLOR_HIT_FLASH,
                1,
                cv2.LINE_AA,
            )

    def _render_click_flashes(self, frame: np.ndarray, now: float) -> None:
        """Render expanding ripple circles on click events."""
        active = []
        for fl in self._click_flashes:
            elapsed = now - fl["time"]
            if elapsed < 0.25:
                active.append(fl)
                prog = elapsed / 0.25
                alpha = 1.0 - prog
                radius = int(14 + 50 * prog)
                c = fl["color"]
                f_color = (int(c[0] * alpha), int(c[1] * alpha), int(c[2] * alpha))
                cv2.circle(frame, (fl["x"], fl["y"]), radius, f_color, max(1, int(3 * alpha)), cv2.LINE_AA)
        self._click_flashes = active
