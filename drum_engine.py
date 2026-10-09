"""Drum Engine module for AirBeat.

Detects air drumming hits from the right hand's index and middle fingertips:
- A hit is registered when high downward y-velocity is followed by deceleration or reversal.
- Enforces an independent per-finger cooldown (configurable in config.py, ~120ms).
- Plays low-latency drum samples (Kick, Snare, Hi-Hat) scaled by hit velocity.
- Supports two mappings:
  1. "finger": Index finger -> Kick, Middle finger -> Snare.
  2. "zones":  Horizontal hand zones (Left -> Kick, Middle -> Snare, Right -> Hi-Hat).
- Prints "HIT index velocity=X" to console.
- Renders an interactive screen flash and live velocity debug overlay for threshold tuning.
"""

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from audio import AudioManager
from config import (
    COLOR_BG_DARK,
    COLOR_HIT_FLASH,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SUCCESS,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    DEFAULT_BPM,
    DRUM_COOLDOWN_MS,
    DRUM_DECEL_RATIO,
    DRUM_FLASH_DURATION_MS,
    DRUM_MAPPING_MODE,
    DRUM_MAX_ARMED_DURATION_MS,
    DRUM_SOUNDS,
    DRUM_VELOCITY_SMOOTHING,
    DRUM_VELOCITY_THRESHOLD,
    DRUM_ZONE_BOUNDARIES,
)


INDEX_TIP_IDX = 8
MIDDLE_TIP_IDX = 12


@dataclass
class HitEvent:
    """Represents a registered drum hit."""

    finger: str  # "index" or "middle"
    velocity: float  # Peak strike downward velocity (norm units/sec)
    timestamp: float
    pixel_pos: Tuple[int, int]  # (x, y) coordinates of the fingertip at hit
    norm_x: float = 0.5  # Normalized x coordinate of fingertip at hit
    drum: str = "kick"  # Triggered drum sound ("kick", "snare", "hihat")
    volume: float = 1.0  # Dynamic playback volume [0.0, 1.0]


class FingerStrikeTracker:
    """Tracks motion and hit state for a single fingertip."""

    def __init__(
        self,
        name: str,
        landmark_idx: int,
        velocity_threshold: float = DRUM_VELOCITY_THRESHOLD,
        cooldown_ms: float = DRUM_COOLDOWN_MS,
        smoothing: float = DRUM_VELOCITY_SMOOTHING,
        decel_ratio: float = DRUM_DECEL_RATIO,
        max_armed_ms: float = DRUM_MAX_ARMED_DURATION_MS,
    ) -> None:
        self.name = name
        self.landmark_idx = landmark_idx
        self.velocity_threshold = velocity_threshold
        self.cooldown_ms = cooldown_ms
        self.smoothing = smoothing
        self.decel_ratio = decel_ratio
        self.max_armed_ms = max_armed_ms

        # Motion state
        self.prev_y: Optional[float] = None
        self.prev_time: Optional[float] = None
        self.raw_velocity: float = 0.0
        self.velocity: float = 0.0
        self.peak_velocity: float = 0.0

        # State machine: "IDLE", "ARMED", "COOLDOWN"
        self.state: str = "IDLE"
        self.arm_time: float = 0.0
        self.last_hit_time: float = 0.0
        self.last_hit_velocity: float = 0.0

        # Pixel & normalized position cache
        self.current_pixel_pos: Tuple[int, int] = (0, 0)
        self.current_norm_x: float = 0.5

    def reset(self) -> None:
        """Reset motion state when hand tracking is lost."""
        self.prev_y = None
        self.prev_time = None
        self.raw_velocity = 0.0
        self.velocity = 0.0
        self.peak_velocity = 0.0
        self.state = "IDLE"

    def update(
        self,
        norm_x: float,
        norm_y: float,
        pixel_pos: Tuple[int, int],
        timestamp: float,
    ) -> Optional[HitEvent]:
        """Update tracker with new frame coordinates and detect hits.

        Args:
            norm_x: Normalized x coordinate (0=left, 1=right).
            norm_y: Normalized y coordinate (0=top, 1=bottom; downward is positive).
            pixel_pos: (x, y) pixel coordinates.
            timestamp: Frame timestamp in seconds.

        Returns:
            HitEvent if a hit was triggered on this frame, else None.
        """
        self.current_pixel_pos = pixel_pos
        self.current_norm_x = norm_x

        if self.prev_y is None or self.prev_time is None:
            self.prev_y = norm_y
            self.prev_time = timestamp
            return None

        dt = timestamp - self.prev_time
        if dt <= 0.001 or dt > 0.3:
            dt = 1.0 / 30.0

        # Downward motion in image space means norm_y is increasing (dy > 0)
        dy = norm_y - self.prev_y
        raw_v = dy / dt

        # Exponential moving average smoothing
        smoothed_v = (
            self.smoothing * raw_v + (1.0 - self.smoothing) * self.velocity
        )
        self.raw_velocity = raw_v
        self.velocity = smoothed_v
        self.prev_y = norm_y
        self.prev_time = timestamp

        hit_event: Optional[HitEvent] = None
        time_since_hit_ms = (timestamp - self.last_hit_time) * 1000.0

        # Cooldown check
        if time_since_hit_ms < self.cooldown_ms:
            self.state = "COOLDOWN"
            return None

        if self.state == "COOLDOWN":
            self.state = "IDLE"

        # State: IDLE -> check for fast downward velocity
        if self.state == "IDLE":
            if self.velocity >= self.velocity_threshold:
                self.state = "ARMED"
                self.arm_time = timestamp
                self.peak_velocity = self.velocity

        # State: ARMED -> track downward strike peak and trigger on deceleration/reversal
        elif self.state == "ARMED":
            if self.velocity > self.peak_velocity:
                self.peak_velocity = self.velocity

            armed_duration_ms = (timestamp - self.arm_time) * 1000.0

            # Conditions for hit completion:
            # 1. Velocity reversal: velocity drops <= 0 (rebound / bounce back upward)
            # 2. Deceleration: velocity drops below decel_ratio of peak_velocity or raw drops sharply
            # 3. Timeout in armed state: exceeded max_armed_ms while slowing down
            is_reversal = (self.velocity <= 0.0) or (self.raw_velocity <= 0.0)
            is_decel = (
                (self.velocity <= (self.decel_ratio * self.peak_velocity))
                or (self.raw_velocity <= (0.40 * self.peak_velocity))
            )
            is_timeout = (
                armed_duration_ms >= self.max_armed_ms
                and self.velocity < self.peak_velocity
            )

            if is_reversal or is_decel or is_timeout:
                # Registered HIT!
                hit_velocity = self.peak_velocity
                self.last_hit_time = timestamp
                self.last_hit_velocity = hit_velocity
                self.state = "COOLDOWN"

                hit_event = HitEvent(
                    finger=self.name,
                    velocity=hit_velocity,
                    timestamp=timestamp,
                    pixel_pos=pixel_pos,
                    norm_x=norm_x,
                )

        return hit_event


class DrumEngine:
    """Manages right-hand air drumming hit detection, visual flash, audio triggers, and debug overlays."""

    def __init__(
        self,
        audio_manager: Optional[AudioManager] = None,
        bpm: int = DEFAULT_BPM,
        velocity_threshold: float = DRUM_VELOCITY_THRESHOLD,
        cooldown_ms: float = DRUM_COOLDOWN_MS,
        velocity_smoothing: float = DRUM_VELOCITY_SMOOTHING,
        decel_ratio: float = DRUM_DECEL_RATIO,
        flash_duration_ms: float = DRUM_FLASH_DURATION_MS,
        mapping_mode: str = DRUM_MAPPING_MODE,
        zone_boundaries: Tuple[float, float] = DRUM_ZONE_BOUNDARIES,
    ) -> None:
        self.bpm = bpm
        self.velocity_threshold = velocity_threshold
        self.cooldown_ms = cooldown_ms
        self.velocity_smoothing = velocity_smoothing
        self.decel_ratio = decel_ratio
        self.flash_duration_ms = flash_duration_ms
        self.mapping_mode = mapping_mode
        self.zone_boundaries = zone_boundaries

        # Audio Manager
        if audio_manager is None:
            self.audio = AudioManager(mapping_mode=self.mapping_mode, zone_boundaries=self.zone_boundaries)
            self.audio.initialize()
        else:
            self.audio = audio_manager

        # Per-finger trackers
        self.index_tracker = FingerStrikeTracker(
            name="index",
            landmark_idx=INDEX_TIP_IDX,
            velocity_threshold=self.velocity_threshold,
            cooldown_ms=self.cooldown_ms,
            smoothing=self.velocity_smoothing,
            decel_ratio=self.decel_ratio,
        )
        self.middle_tracker = FingerStrikeTracker(
            name="middle",
            landmark_idx=MIDDLE_TIP_IDX,
            velocity_threshold=self.velocity_threshold,
            cooldown_ms=self.cooldown_ms,
            smoothing=self.velocity_smoothing,
            decel_ratio=self.decel_ratio,
        )

        # Active flash events for visual screen flash
        self._active_flashes: List[Dict[str, Any]] = []

        # BPM estimation tap timestamps
        self._tap_timestamps: List[float] = []

    def set_mapping_mode(self, mode: str) -> None:
        """Switch between 'finger' and 'zones' mapping modes."""
        if mode in ("finger", "zones"):
            self.mapping_mode = mode
            if self.audio:
                self.audio.set_mapping_mode(mode)
            print(f"[DrumEngine] Mapping mode changed to: {self.mapping_mode.upper()}")

    def toggle_mapping_mode(self) -> str:
        """Toggle between 'finger' and 'zones' modes."""
        new_mode = "zones" if self.mapping_mode == "finger" else "finger"
        self.set_mapping_mode(new_mode)
        return self.mapping_mode

    def process_right_hand(
        self,
        right_hand_data: Optional[Any],
        timestamp: Optional[float] = None,
    ) -> List[HitEvent]:
        """Process right hand landmarks to detect index and middle finger hits.

        Args:
            right_hand_data: HandData instance from HandTracker (or None).
            timestamp: Frame timestamp (defaults to time.time()).

        Returns:
            List of HitEvent objects detected in this frame.
        """
        if timestamp is None:
            timestamp = time.time()

        if right_hand_data is None:
            self.index_tracker.reset()
            self.middle_tracker.reset()
            return []

        # Extract landmarks: normalized (21, 3) and pixel (21, 2)
        landmarks = right_hand_data.landmarks
        pixel_landmarks = right_hand_data.pixel_landmarks

        index_norm_x = float(landmarks[INDEX_TIP_IDX, 0])
        index_norm_y = float(landmarks[INDEX_TIP_IDX, 1])
        index_px = (
            int(pixel_landmarks[INDEX_TIP_IDX, 0]),
            int(pixel_landmarks[INDEX_TIP_IDX, 1]),
        )

        middle_norm_x = float(landmarks[MIDDLE_TIP_IDX, 0])
        middle_norm_y = float(landmarks[MIDDLE_TIP_IDX, 1])
        middle_px = (
            int(pixel_landmarks[MIDDLE_TIP_IDX, 0]),
            int(pixel_landmarks[MIDDLE_TIP_IDX, 1]),
        )

        hits: List[HitEvent] = []

        # Update index finger tracker
        index_hit = self.index_tracker.update(
            index_norm_x, index_norm_y, index_px, timestamp
        )
        if index_hit is not None:
            self._handle_hit_trigger(index_hit)
            hits.append(index_hit)

        # Update middle finger tracker
        middle_hit = self.middle_tracker.update(
            middle_norm_x, middle_norm_y, middle_px, timestamp
        )
        if middle_hit is not None:
            self._handle_hit_trigger(middle_hit)
            hits.append(middle_hit)

        return hits

    def _handle_hit_trigger(self, hit: HitEvent) -> None:
        """Trigger audio sample, scale volume, log to console, and register screen flash."""
        drum_name, volume = self.audio.trigger_hit(
            finger_name=hit.finger,
            velocity=hit.velocity,
            norm_x=hit.norm_x,
            mapping_mode=self.mapping_mode,
        )
        hit.drum = drum_name
        hit.volume = volume

        # Print exact requested console output with drum and volume details
        print(f"HIT {hit.finger} velocity={hit.velocity:.2f} [{drum_name.upper()} vol={int(volume * 100)}%]")

        self._register_flash(hit)
        self._update_bpm_tap(hit.timestamp)

    def _register_flash(self, hit: HitEvent) -> None:
        """Register a visual screen flash event."""
        self._active_flashes.append(
            {
                "finger": hit.finger,
                "drum": hit.drum,
                "velocity": hit.velocity,
                "volume": hit.volume,
                "timestamp": hit.timestamp,
                "pixel_pos": hit.pixel_pos,
            }
        )

    def _update_bpm_tap(self, timestamp: float) -> None:
        """Update estimated BPM from rhythmic hit cadence."""
        self._tap_timestamps.append(timestamp)
        # Keep recent taps within 3 seconds
        self._tap_timestamps = [t for t in self._tap_timestamps if timestamp - t <= 3.0]
        if len(self._tap_timestamps) >= 3:
            intervals = [
                self._tap_timestamps[i] - self._tap_timestamps[i - 1]
                for i in range(1, len(self._tap_timestamps))
            ]
            avg_interval = sum(intervals) / len(intervals)
            if 0.2 <= avg_interval <= 1.5:
                estimated_bpm = int(60.0 / avg_interval)
                self.bpm = max(60, min(200, estimated_bpm))

    def draw_debug_overlay(
        self,
        frame: np.ndarray,
        right_hand_data: Optional[Any],
        current_time: Optional[float] = None,
    ) -> np.ndarray:
        """Render live velocity meters, horizontal zones, fingertip vectors, and screen flash effects.

        Args:
            frame: OpenCV BGR image frame.
            right_hand_data: HandData from HandTracker (or None).
            current_time: Current timestamp in seconds.

        Returns:
            Frame with debug graphics and hit flashes rendered.
        """
        if current_time is None:
            current_time = time.time()

        h, w = frame.shape[:2]

        # 1. Render Horizontal Hand Zones if in "zones" mapping mode
        if self.mapping_mode == "zones":
            self._render_horizontal_zones(frame, w, h, right_hand_data)

        # 2. Render Screen Flash Effect if any hits are active
        frame = self._render_screen_flashes(frame, current_time)

        # 3. Render Live Fingertip Velocity Indicators on Hand
        if right_hand_data is not None:
            self._render_fingertip_indicators(frame)

        # 4. Render Tuning HUD Panel (top right)
        self._render_tuning_hud(frame, w, h, current_time)

        return frame

    def _render_horizontal_zones(
        self,
        frame: np.ndarray,
        w: int,
        h: int,
        right_hand_data: Optional[Any],
    ) -> None:
        """Draw vertical zone dividers and labels for Kick, Snare, and Hi-Hat."""
        b1_x = int(self.zone_boundaries[0] * w)
        b2_x = int(self.zone_boundaries[1] * w)

        hand_x = -1.0
        if right_hand_data is not None:
            hand_x = float(right_hand_data.landmarks[INDEX_TIP_IDX, 0])

        overlay = frame.copy()

        # Zone 1: Kick (< b1_x)
        is_z1 = 0 <= hand_x < self.zone_boundaries[0]
        c1 = (50, 120, 255) if is_z1 else (30, 40, 50)
        cv2.rectangle(overlay, (0, 50), (b1_x, h), c1, -1)

        # Zone 2: Snare (b1_x <= x < b2_x)
        is_z2 = self.zone_boundaries[0] <= hand_x < self.zone_boundaries[1]
        c2 = (40, 200, 100) if is_z2 else (25, 45, 35)
        cv2.rectangle(overlay, (b1_x, 50), (b2_x, h), c2, -1)

        # Zone 3: Hi-Hat (>= b2_x)
        is_z3 = hand_x >= self.zone_boundaries[1]
        c3 = (200, 180, 40) if is_z3 else (40, 40, 30)
        cv2.rectangle(overlay, (b2_x, 50), (w, h), c3, -1)

        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # Divider lines
        cv2.line(frame, (b1_x, 50), (b1_x, h), (120, 120, 140), 1, cv2.LINE_AA)
        cv2.line(frame, (b2_x, 50), (b2_x, h), (120, 120, 140), 1, cv2.LINE_AA)

        # Zone header tags at the bottom
        y_label = h - 16
        cv2.putText(
            frame,
            "[ ZONE: KICK ]",
            (20, y_label),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255) if is_z1 else (160, 160, 160),
            2 if is_z1 else 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "[ ZONE: SNARE ]",
            (b1_x + 20, y_label),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 255, 120) if is_z2 else (160, 160, 160),
            2 if is_z2 else 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "[ ZONE: HI-HAT ]",
            (b2_x + 20, y_label),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 230, 80) if is_z3 else (160, 160, 160),
            2 if is_z3 else 1,
            cv2.LINE_AA,
        )

    def _render_screen_flashes(
        self, frame: np.ndarray, current_time: float
    ) -> np.ndarray:
        """Draw border flash and shockwave ripples for active hit events."""
        active = []
        flash_duration_sec = self.flash_duration_ms / 1000.0

        for fl in self._active_flashes:
            elapsed = current_time - fl["timestamp"]
            if elapsed < flash_duration_sec:
                active.append(fl)
                progress = elapsed / flash_duration_sec
                alpha = max(0.0, 1.0 - progress)

                # Screen perimeter flash border
                h, w = frame.shape[:2]
                border_thick = int(14 * alpha)
                if border_thick > 0:
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), COLOR_HIT_FLASH, border_thick)
                    cv2.addWeighted(overlay, alpha * 0.7, frame, 1.0 - (alpha * 0.7), 0, frame)

                # Expanding ripple circle at fingertip
                px, py = fl["pixel_pos"]
                radius = int(20 + 70 * progress)
                ripple_alpha = alpha
                ripple_color = (
                    int(COLOR_HIT_FLASH[0] * ripple_alpha),
                    int(COLOR_HIT_FLASH[1] * ripple_alpha),
                    int(COLOR_HIT_FLASH[2] * ripple_alpha),
                )
                cv2.circle(frame, (px, py), radius, ripple_color, max(1, int(3 * alpha)), cv2.LINE_AA)

                # Hit text tag
                vol_pct = int(fl.get("volume", 1.0) * 100)
                tag_text = f"HIT {fl['drum'].upper()}! (v={fl['velocity']:.2f}, vol={vol_pct}%)"
                (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
                cv2.putText(
                    frame,
                    tag_text,
                    (max(10, px - tw // 2), max(th + 10, py - radius - 8)),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.7,
                    COLOR_HIT_FLASH,
                    2,
                    cv2.LINE_AA,
                )

        self._active_flashes = active
        return frame

    def _render_fingertip_indicators(self, frame: np.ndarray) -> None:
        """Render floating velocity badges next to the index and middle fingertips."""
        trackers = [self.index_tracker, self.middle_tracker]

        for tr in trackers:
            px, py = tr.current_pixel_pos
            if px == 0 and py == 0:
                continue

            v = tr.velocity
            sign = "+" if v >= 0 else ""
            target_drum = self.audio.resolve_drum(tr.name, tr.current_norm_x, self.mapping_mode)
            label = f"{tr.name[:3]}->{target_drum[:4].upper()}: {sign}{v:.2f}"

            # Color coding: Green if moving down fast, Red if moving up, White otherwise
            if tr.state == "ARMED":
                badge_bg = (0, 165, 255)  # Orange/Amber
                badge_fg = (255, 255, 255)
            elif tr.state == "COOLDOWN":
                badge_bg = (100, 100, 100)
                badge_fg = (200, 200, 200)
            elif v >= self.velocity_threshold:
                badge_bg = (40, 200, 100)
                badge_fg = (0, 0, 0)
            else:
                badge_bg = (40, 40, 48)
                badge_fg = (220, 220, 220)

            # Draw small floating pill
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            box_x = px + 12
            box_y = py - 8
            cv2.rectangle(
                frame,
                (box_x, box_y - th - 4),
                (box_x + tw + 8, box_y + 4),
                badge_bg,
                -1,
            )
            cv2.rectangle(
                frame,
                (box_x, box_y - th - 4),
                (box_x + tw + 8, box_y + 4),
                (180, 180, 180),
                1,
            )
            cv2.putText(
                frame,
                label,
                (box_x + 4, box_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                badge_fg,
                1,
                cv2.LINE_AA,
            )

            # Draw downward velocity vector arrow if striking
            if v > 0.3:
                arrow_len = min(60, int(v * 20))
                cv2.arrowedLine(
                    frame,
                    (px, py),
                    (px, py + arrow_len),
                    COLOR_HIT_FLASH if tr.state == "ARMED" else (0, 220, 255),
                    2,
                    tipLength=0.35,
                )

    def _render_tuning_hud(
        self,
        frame: np.ndarray,
        w: int,
        h: int,
        current_time: float,
    ) -> None:
        """Render velocity tuning meters panel in the top-right corner."""
        panel_w = 330
        panel_h = 185
        panel_x = w - panel_w - 15
        panel_y = 60

        # Semi-transparent dark background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            COLOR_BG_DARK,
            -1,
        )
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (70, 70, 85),
            1,
        )
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Panel Title
        cv2.putText(
            frame,
            "AIRBEAT DRUM & AUDIO ENGINE",
            (panel_x + 12, panel_y + 22),
            cv2.FONT_HERSHEY_DUPLEX,
            0.48,
            COLOR_SECONDARY,
            1,
            cv2.LINE_AA,
        )

        # Mapping mode badge
        mode_desc = "FINGER (Idx=Kick, Mid=Snare)" if self.mapping_mode == "finger" else "ZONES (L=Kick, M=Snare, R=Hat)"
        cv2.putText(
            frame,
            f"Mode: {mode_desc}",
            (panel_x + 12, panel_y + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )

        # Config stats line
        config_info = (
            f"Thresh: {self.velocity_threshold:.1f} | "
            f"Cool: {int(self.cooldown_ms)}ms | "
            f"BPM: {self.bpm} | [M]ode"
        )
        cv2.putText(
            frame,
            config_info,
            (panel_x + 12, panel_y + 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            COLOR_TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )

        # Render meter for each finger
        self._render_finger_meter(
            frame,
            tracker=self.index_tracker,
            display_name="INDEX (tip #8)",
            x=panel_x + 12,
            y=panel_y + 66,
            bar_w=panel_w - 24,
            current_time=current_time,
        )

        self._render_finger_meter(
            frame,
            tracker=self.middle_tracker,
            display_name="MIDDLE (tip #12)",
            x=panel_x + 12,
            y=panel_y + 124,
            bar_w=panel_w - 24,
            current_time=current_time,
        )

    def _render_finger_meter(
        self,
        frame: np.ndarray,
        tracker: FingerStrikeTracker,
        display_name: str,
        x: int,
        y: int,
        bar_w: int,
        current_time: float,
    ) -> None:
        """Render a single finger's live velocity bar, threshold mark, and state badge."""
        bar_h = 12
        max_meter_v = max(3.5, self.velocity_threshold * 2.0)

        # Target drum sound
        target_drum = self.audio.resolve_drum(tracker.name, tracker.current_norm_x, self.mapping_mode)
        dyn_vol = int(self.audio.velocity_to_volume(tracker.velocity) * 100)

        # Name, drum, and velocity reading
        sign = "+" if tracker.velocity >= 0 else ""
        vel_text = f"{display_name} -> {target_drum.upper()}: {sign}{tracker.velocity:.2f} ({dyn_vol}%)"
        cv2.putText(
            frame,
            vel_text,
            (x, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )

        # State pill badge
        state_label = tracker.state
        if tracker.state == "COOLDOWN":
            remaining_ms = int(
                max(0.0, tracker.cooldown_ms - (current_time - tracker.last_hit_time) * 1000.0)
            )
            state_label = f"COOL ({remaining_ms}ms)"
            badge_color = (110, 110, 110)
        elif tracker.state == "ARMED":
            badge_color = (0, 180, 255)  # Orange
        elif tracker.velocity >= self.velocity_threshold:
            badge_color = COLOR_SUCCESS
        else:
            badge_color = (60, 60, 70)

        (sw, _), _ = cv2.getTextSize(state_label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        badge_x = x + bar_w - sw - 10
        cv2.rectangle(
            frame,
            (badge_x, y),
            (badge_x + sw + 8, y + 16),
            badge_color,
            -1,
        )
        cv2.putText(
            frame,
            state_label,
            (badge_x + 4, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Bar background
        bar_y = y + 20
        cv2.rectangle(
            frame,
            (x, bar_y),
            (x + bar_w, bar_y + bar_h),
            (35, 35, 42),
            -1,
        )

        # Fill bar proportional to positive downward velocity
        norm_val = max(0.0, min(1.0, tracker.velocity / max_meter_v))
        fill_w = int(norm_val * bar_w)
        if fill_w > 0:
            if tracker.state == "ARMED":
                fill_color = COLOR_HIT_FLASH
            elif tracker.velocity >= self.velocity_threshold:
                fill_color = COLOR_SUCCESS
            else:
                fill_color = COLOR_PRIMARY

            cv2.rectangle(
                frame,
                (x, bar_y),
                (x + fill_w, bar_y + bar_h),
                fill_color,
                -1,
            )

        # Threshold marker line
        thresh_x = int(x + (self.velocity_threshold / max_meter_v) * bar_w)
        cv2.line(
            frame,
            (thresh_x, bar_y - 3),
            (thresh_x, bar_y + bar_h + 3),
            (0, 0, 255),  # Red threshold marker line
            2,
            cv2.LINE_AA,
        )

        # Frame border
        cv2.rectangle(
            frame,
            (x, bar_y),
            (x + bar_w, bar_y + bar_h),
            (80, 80, 95),
            1,
        )


def run_drum_engine_test(
    camera_index: int = 0,
    max_frames: Optional[int] = None,
) -> None:
    """Standalone live test runner for drum engine hit detection, audio synthesis, and tuning."""
    from hand_tracker import HandTracker

    print("[AirBeat] Initializing Drum Engine live test...")
    tracker = HandTracker(mirrored=True, use_one_euro_filter=True)
    if not tracker.initialize():
        print("❌ [AirBeat] Failed to initialize HandTracker.")
        return

    audio_manager = AudioManager()
    if not audio_manager.initialize():
        print("❌ [AirBeat] Failed to initialize AudioManager.")
        return

    drum_engine = DrumEngine(audio_manager=audio_manager)

    cap = None
    if hasattr(cv2, "CAP_AVFOUNDATION"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"❌ [AirBeat] Could not open camera at index {camera_index}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "AirBeat - Drum Engine & Audio Playback"
    window_supported = True
    frame_count = 0

    print("=" * 60)
    print("🥁 AirBeat Drum Engine & Audio Live")
    print(f"   Mapping Mode:       {drum_engine.mapping_mode.upper()} (press 'm' in window to toggle)")
    print(f"   Velocity Threshold: {drum_engine.velocity_threshold:.2f} norm/s")
    print(f"   Cooldown:           {drum_engine.cooldown_ms:.0f} ms")
    print("   Strike downward with Right Hand Index or Middle finger to play drums!")
    print("   Press 'q' or ESC in window to exit.")
    print("=" * 60 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_count += 1
            frame = cv2.flip(frame, 1)
            now = time.time()

            # Process hand tracking
            left_hand, right_hand = tracker.process_frame(frame, timestamp=now)

            # Process drum hit detection on right hand and trigger sounds
            hits = drum_engine.process_right_hand(right_hand, timestamp=now)

            # Draw hand skeleton and labels
            frame = tracker.draw_debug(
                frame, left_hand, right_hand, show_fps=True, show_hud=True
            )

            # Draw drum tuning overlay (meters, zones, fingertip tags, flash)
            frame = drum_engine.draw_debug_overlay(
                frame, right_hand, current_time=now
            )

            if window_supported:
                try:
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                    elif key in (ord("m"), ord("M")):
                        # Toggle mapping mode
                        new_m = drum_engine.toggle_mapping_mode()
                except cv2.error as e:
                    print(f"[AirBeat] Window display unavailable: {e}")
                    window_supported = False

            if max_frames is not None and frame_count >= max_frames:
                break

    finally:
        cap.release()
        if window_supported:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        tracker.release()
        audio_manager.cleanup()
        print(f"\n✅ [AirBeat] Drum test completed ({frame_count} frames processed).")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AirBeat Drum Engine Live Tuning")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--test-frames", type=int, default=None, help="Stop after N frames")
    args = parser.parse_args()

    run_drum_engine_test(camera_index=args.camera_index, max_frames=args.test_frames)
