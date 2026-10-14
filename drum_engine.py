"""Drum Engine module for AirBeat.

Detects air drumming hits from the right hand's index and middle fingertips:
- A hit is registered when high downward y-velocity is followed by deceleration or reversal.
- Enforces an independent per-finger cooldown (configurable in config.py, ~120ms).
- Plays low-latency drum samples (Kick, Snare, Hi-Hat) scaled by hit velocity.
- Supports two drum mappings:
  1. "finger": Index finger -> Kick, Middle finger -> Snare.
  2. "zones":  Horizontal hand zones (Left -> Kick, Middle -> Snare, Right -> Hi-Hat).
- Threaded metronome/clock with adjustable BPM:
  1. Mode (a) "tap": Tap tempo from time between hits (average of last 4 intervals).
  2. Mode (b) "height": Right hand height mapped to BPM 60-180 with exponential smoothing.
- Optional hit quantization to 1/8 or 1/16 note beat grid.
- Live on-screen BPM display, beat pulse animation, and velocity tuning meters.
"""

from collections import deque
from dataclasses import dataclass
import math
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from audio import AudioManager
from config import (
    AR_PADS_ENABLED,
    BPM_CONTROL_MODE,
    BPM_HEIGHT_MAX_Y,
    BPM_HEIGHT_MIN_Y,
    BPM_HEIGHT_SMOOTHING,
    BPM_MAX,
    BPM_MIN,
    BPM_TAP_HISTORY_COUNT,
    COLOR_BG_DARK,
    COLOR_HIT_FLASH,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SUCCESS,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    DEFAULT_BPM,
    DEFAULT_DRUM_KIT,
    DEFAULT_SENSITIVITY,
    DRUM_COOLDOWN_MS,
    DRUM_DECEL_RATIO,
    DRUM_FLASH_DURATION_MS,
    DRUM_KITS,
    DRUM_MAPPING_MODE,
    DRUM_MAX_ARMED_DURATION_MS,
    DRUM_PINKY_VELOCITY_THRESHOLD,
    DRUM_RING_VELOCITY_THRESHOLD,
    DRUM_SOUNDS,
    DRUM_THUMB_VELOCITY_THRESHOLD,
    DRUM_VELOCITY_SMOOTHING,
    DRUM_VELOCITY_THRESHOLD,
    DRUM_ZONE_BOUNDARIES,
    LEFT_HAND_DRUM_MAP,
    METRONOME_CLICK_ENABLED,
    METRONOME_CLICK_VOLUME,
    QUANTIZE_MODE,
    SENSITIVITY_PRESETS,
    TWO_HAND_DRUMS_ENABLED,
)
from metronome import MetronomeClock


# Right hand 5-finger fingertip and knuckle base landmark indices
THUMB_TIP_IDX = 4
THUMB_BASE_IDX = 2    # Thumb MCP
INDEX_TIP_IDX = 8
INDEX_BASE_IDX = 5    # Index MCP
MIDDLE_TIP_IDX = 12
MIDDLE_BASE_IDX = 9   # Middle MCP
RING_TIP_IDX = 16
RING_BASE_IDX = 13    # Ring MCP
PINKY_TIP_IDX = 20
PINKY_BASE_IDX = 17   # Pinky MCP


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
    quantized: bool = False  # Whether playback was scheduled to a beat grid


class FingerStrikeTracker:
    """Tracks motion and hit state for a single fingertip."""

    def __init__(
        self,
        name: str,
        landmark_idx: int,
        base_landmark_idx: Optional[int] = None,
        velocity_threshold: float = DRUM_VELOCITY_THRESHOLD,
        cooldown_ms: float = DRUM_COOLDOWN_MS,
        smoothing: float = DRUM_VELOCITY_SMOOTHING,
        decel_ratio: float = DRUM_DECEL_RATIO,
        max_armed_ms: float = DRUM_MAX_ARMED_DURATION_MS,
    ) -> None:
        self.name = name
        self.landmark_idx = landmark_idx
        self.base_landmark_idx = base_landmark_idx
        self.velocity_threshold = velocity_threshold
        self.cooldown_ms = cooldown_ms
        self.smoothing = smoothing
        self.decel_ratio = decel_ratio
        self.max_armed_ms = max_armed_ms

        # Motion state
        self.prev_x: Optional[float] = None
        self.prev_y: Optional[float] = None
        self.prev_base_y: Optional[float] = None
        self.prev_time: Optional[float] = None
        self.raw_velocity: float = 0.0
        self.velocity: float = 0.0
        self.peak_velocity: float = 0.0

        # State machine: "IDLE", "ARMED", "COOLDOWN"
        self.state: str = "IDLE"
        self.arm_time: float = 0.0
        self.last_hit_time: float = 0.0
        self.last_hit_velocity: float = 0.0

        # Position caches
        self.current_pixel_pos: Tuple[int, int] = (0, 0)
        self.current_norm_x: float = 0.5

    def reset(self) -> None:
        """Reset motion state when hand tracking is lost."""
        self.prev_x = None
        self.prev_y = None
        self.prev_base_y = None
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
        base_norm_y: Optional[float] = None,
    ) -> Optional[HitEvent]:
        """Update tracker with new frame coordinates and detect hits.

        Args:
            norm_x: Normalized x coordinate (0=left, 1=right).
            norm_y: Normalized y coordinate (0=top, 1=bottom; downward is positive).
            pixel_pos: (x, y) pixel coordinates.
            timestamp: Frame timestamp in seconds.
            base_norm_y: Optional normalized y coordinate of knuckle MCP base for relative flexion.

        Returns:
            HitEvent if a hit was triggered on this frame, else None.
        """
        self.current_pixel_pos = pixel_pos
        self.current_norm_x = norm_x

        if self.prev_y is None or self.prev_time is None:
            self.prev_x = norm_x
            self.prev_y = norm_y
            self.prev_base_y = base_norm_y
            self.prev_time = timestamp
            return None

        dt = timestamp - self.prev_time
        if dt <= 0.001 or dt > 0.3:
            dt = 1.0 / 30.0

        # Absolute displacement in normalized screen coordinates (downward is positive)
        dy_abs = norm_y - self.prev_y
        dx = norm_x - self.prev_x if self.prev_x is not None else 0.0

        # Relative flexion displacement (fingertip moving downward relative to knuckle)
        dy_rel = 0.0
        if base_norm_y is not None and self.prev_base_y is not None:
            rel_curr = norm_y - base_norm_y
            rel_prev = self.prev_y - self.prev_base_y
            dy_rel = rel_curr - rel_prev
        self.prev_base_y = base_norm_y

        if self.name == "thumb":
            # Thumb taps pivot in a downward-inward arc: combine downward flexion & lateral flick
            flick_y = max(dy_abs, dy_rel, 0.0)
            flick_mag = flick_y + 0.60 * abs(dx)
            raw_v = flick_mag / dt if (flick_y > 0.001 or abs(dx) > 0.003) else (dy_abs / dt)
        else:
            # Combine absolute motion (entire hand downward) and knuckle-relative finger flexion
            if dy_abs > 0 and dy_rel > 0:
                dy_eff = max(dy_abs, dy_rel, 0.70 * dy_abs + 0.60 * dy_rel)
            elif dy_abs > 0:
                dy_eff = dy_abs
            elif dy_rel > 0:
                # Finger flicked down even if palm/wrist moved slightly upward
                dy_eff = dy_rel
            else:
                dy_eff = min(dy_abs, dy_rel)
            raw_v = dy_eff / dt

        # Fast-response exponential moving average
        prev_smoothed_v = self.velocity
        smoothed_v = (
            self.smoothing * raw_v + (1.0 - self.smoothing) * self.velocity
        )
        self.raw_velocity = raw_v
        self.velocity = smoothed_v
        self.prev_x = norm_x
        self.prev_y = norm_y
        self.prev_time = timestamp

        # Instantaneous effective velocity ensures quick flicks aren't dampened
        effective_v = max(self.velocity, self.raw_velocity)

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
            if effective_v >= self.velocity_threshold:
                self.state = "ARMED"
                self.arm_time = timestamp
                self.peak_velocity = effective_v

        # State: ARMED -> track downward strike peak and trigger on deceleration/reversal
        elif self.state == "ARMED":
            if effective_v > self.peak_velocity:
                self.peak_velocity = effective_v

            armed_duration_ms = (timestamp - self.arm_time) * 1000.0

            # Conditions for hit completion:
            # 1. Any downward speed drop or reversal: raw drops below peak or drops below zero
            # 2. Deceleration: dropped below decel_ratio of peak_velocity or smoothed begins to fall
            # 3. Timeout in armed state: exceeded max_armed_ms while slowing down
            is_reversal = (self.raw_velocity <= 0.0) or (self.velocity <= 0.0)
            is_decel = (
                (self.raw_velocity <= (self.decel_ratio * self.peak_velocity))
                or (self.velocity < prev_smoothed_v)
            )
            is_timeout = (
                armed_duration_ms >= self.max_armed_ms
                and effective_v < self.peak_velocity
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
    """Manages right-hand hit detection, threaded metronome, BPM modes, audio, and visual overlays."""

    def __init__(
        self,
        audio_manager: Optional[AudioManager] = None,
        clock: Optional[MetronomeClock] = None,
        bpm: int = DEFAULT_BPM,
        velocity_threshold: float = DRUM_VELOCITY_THRESHOLD,
        cooldown_ms: float = DRUM_COOLDOWN_MS,
        velocity_smoothing: float = DRUM_VELOCITY_SMOOTHING,
        decel_ratio: float = DRUM_DECEL_RATIO,
        flash_duration_ms: float = DRUM_FLASH_DURATION_MS,
        mapping_mode: str = DRUM_MAPPING_MODE,
        zone_boundaries: Tuple[float, float] = DRUM_ZONE_BOUNDARIES,
        bpm_control_mode: str = BPM_CONTROL_MODE,
        quantize_mode: str = QUANTIZE_MODE,
    ) -> None:
        self.bpm = bpm
        self.velocity_threshold = velocity_threshold
        self.cooldown_ms = cooldown_ms
        self.velocity_smoothing = velocity_smoothing
        self.decel_ratio = decel_ratio
        self.flash_duration_ms = flash_duration_ms
        self.mapping_mode = mapping_mode
        self.zone_boundaries = zone_boundaries
        self.bpm_mode = bpm_control_mode
        self.quantize_mode = quantize_mode

        # Audio Manager
        if audio_manager is None:
            self.audio = AudioManager(
                mapping_mode=self.mapping_mode, zone_boundaries=self.zone_boundaries
            )
            self.audio.initialize()
        else:
            self.audio = audio_manager

        # Threaded Metronome Clock
        if clock is None:
            self.clock = MetronomeClock(
                bpm=self.bpm,
                click_enabled=METRONOME_CLICK_ENABLED,
                click_volume=METRONOME_CLICK_VOLUME,
                quantize_mode=self.quantize_mode,
            )
            self.clock.start()
        else:
            self.clock = clock

        # Per-finger trackers (All 5 fingers of the right hand / green hand)
        self.thumb_tracker = FingerStrikeTracker(
            name="thumb",
            landmark_idx=THUMB_TIP_IDX,
            base_landmark_idx=THUMB_BASE_IDX,
            velocity_threshold=DRUM_THUMB_VELOCITY_THRESHOLD,
            cooldown_ms=self.cooldown_ms,
            smoothing=self.velocity_smoothing,
            decel_ratio=self.decel_ratio,
        )
        self.index_tracker = FingerStrikeTracker(
            name="index",
            landmark_idx=INDEX_TIP_IDX,
            base_landmark_idx=INDEX_BASE_IDX,
            velocity_threshold=self.velocity_threshold,
            cooldown_ms=self.cooldown_ms,
            smoothing=self.velocity_smoothing,
            decel_ratio=self.decel_ratio,
        )
        self.middle_tracker = FingerStrikeTracker(
            name="middle",
            landmark_idx=MIDDLE_TIP_IDX,
            base_landmark_idx=MIDDLE_BASE_IDX,
            velocity_threshold=self.velocity_threshold,
            cooldown_ms=self.cooldown_ms,
            smoothing=self.velocity_smoothing,
            decel_ratio=self.decel_ratio,
        )
        self.ring_tracker = FingerStrikeTracker(
            name="ring",
            landmark_idx=RING_TIP_IDX,
            base_landmark_idx=RING_BASE_IDX,
            velocity_threshold=DRUM_RING_VELOCITY_THRESHOLD,
            cooldown_ms=self.cooldown_ms,
            smoothing=self.velocity_smoothing,
            decel_ratio=self.decel_ratio,
        )
        self.pinky_tracker = FingerStrikeTracker(
            name="pinky",
            landmark_idx=PINKY_TIP_IDX,
            base_landmark_idx=PINKY_BASE_IDX,
            velocity_threshold=DRUM_PINKY_VELOCITY_THRESHOLD,
            cooldown_ms=self.cooldown_ms,
            smoothing=self.velocity_smoothing,
            decel_ratio=self.decel_ratio,
        )
        self.finger_trackers = [
            self.thumb_tracker,
            self.index_tracker,
            self.middle_tracker,
            self.ring_tracker,
            self.pinky_tracker,
        ]

        # Two-Handed Drumming trackers (Left Hand: green/cyan dual strike)
        self.two_hand_drums_enabled: bool = TWO_HAND_DRUMS_ENABLED
        self.left_thumb_tracker = FingerStrikeTracker(
            name="thumb",
            landmark_idx=THUMB_TIP_IDX,
            base_landmark_idx=THUMB_BASE_IDX,
            velocity_threshold=DRUM_THUMB_VELOCITY_THRESHOLD,
            cooldown_ms=self.cooldown_ms,
            smoothing=self.velocity_smoothing,
            decel_ratio=self.decel_ratio,
        )
        self.left_index_tracker = FingerStrikeTracker(
            name="index",
            landmark_idx=INDEX_TIP_IDX,
            base_landmark_idx=INDEX_BASE_IDX,
            velocity_threshold=self.velocity_threshold,
            cooldown_ms=self.cooldown_ms,
            smoothing=self.velocity_smoothing,
            decel_ratio=self.decel_ratio,
        )
        self.left_middle_tracker = FingerStrikeTracker(
            name="middle",
            landmark_idx=MIDDLE_TIP_IDX,
            base_landmark_idx=MIDDLE_BASE_IDX,
            velocity_threshold=self.velocity_threshold,
            cooldown_ms=self.cooldown_ms,
            smoothing=self.velocity_smoothing,
            decel_ratio=self.decel_ratio,
        )
        self.left_ring_tracker = FingerStrikeTracker(
            name="ring",
            landmark_idx=RING_TIP_IDX,
            base_landmark_idx=RING_BASE_IDX,
            velocity_threshold=DRUM_RING_VELOCITY_THRESHOLD,
            cooldown_ms=self.cooldown_ms,
            smoothing=self.velocity_smoothing,
            decel_ratio=self.decel_ratio,
        )
        self.left_pinky_tracker = FingerStrikeTracker(
            name="pinky",
            landmark_idx=PINKY_TIP_IDX,
            base_landmark_idx=PINKY_BASE_IDX,
            velocity_threshold=DRUM_PINKY_VELOCITY_THRESHOLD,
            cooldown_ms=self.cooldown_ms,
            smoothing=self.velocity_smoothing,
            decel_ratio=self.decel_ratio,
        )
        self.left_finger_trackers = [
            self.left_thumb_tracker,
            self.left_index_tracker,
            self.left_middle_tracker,
            self.left_ring_tracker,
            self.left_pinky_tracker,
        ]

        # Sensitivity Presets: apply default preset
        self.sensitivity_preset: str = DEFAULT_SENSITIVITY
        self.set_sensitivity_preset(self.sensitivity_preset)

        # Holographic AR Drum Pads (5 pads in semi-circular ergonomic arc)
        self.ar_pads_enabled: bool = AR_PADS_ENABLED
        self.ar_pads: Dict[str, Dict[str, Any]] = {
            "kick": {
                "display": "KICK",
                "finger": "THB",
                "norm_pos": (0.42, 0.77),
                "radius": 46,
                "color": (255, 140, 0),    # Cyan/Blue
                "hit_glow": 0.0,
                "compression": 0.0,
            },
            "snare": {
                "display": "SNARE",
                "finger": "IDX",
                "norm_pos": (0.54, 0.73),
                "radius": 44,
                "color": (80, 240, 120),   # Emerald green
                "hit_glow": 0.0,
                "compression": 0.0,
            },
            "hihat": {
                "display": "HI-HAT",
                "finger": "MID",
                "norm_pos": (0.66, 0.68),
                "radius": 42,
                "color": (0, 240, 255),    # Golden yellow
                "hit_glow": 0.0,
                "compression": 0.0,
            },
            "tom": {
                "display": "TOM",
                "finger": "RNG",
                "norm_pos": (0.78, 0.70),
                "radius": 44,
                "color": (0, 160, 255),    # Vibrant Amber
                "hit_glow": 0.0,
                "compression": 0.0,
            },
            "crash": {
                "display": "CRASH",
                "finger": "PNK",
                "norm_pos": (0.89, 0.64),
                "radius": 44,
                "color": (220, 40, 255),   # Magenta
                "hit_glow": 0.0,
                "compression": 0.0,
            },
        }

        # Particle and shockwave visual FX
        self.shockwaves: List[Dict[str, Any]] = []
        self.neon_particles: List[Dict[str, Any]] = []

        # Subsystem references for live looper, jam recorder, and rhythm hero game
        self.looper: Optional[Any] = None
        self.jam_recorder: Optional[Any] = None
        self.rhythm_game: Optional[Any] = None

        # Optional listener hook for live demo/tutorial feedback
        self.on_hit_callback: Optional[Callable[[HitEvent], None]] = None

        # Active flash events for visual screen flash
        self._active_flashes: List[Dict[str, Any]] = []

        # BPM Mode (a): Tap tempo state (average of last 4 intervals)
        self._tap_hit_times: deque = deque(maxlen=8)
        self._tap_intervals: deque = deque(maxlen=BPM_TAP_HISTORY_COUNT)

        # BPM Mode (b): Hand height state
        self._smooth_bpm: float = float(bpm)
        self.current_hand_height_frac: float = 0.5
        self.target_height_bpm: float = float(bpm)

    def set_bpm(self, new_bpm: int) -> None:
        """Update BPM across engine and threaded metronome clock."""
        clamped = max(BPM_MIN, min(BPM_MAX, int(new_bpm)))
        self.bpm = clamped
        if self.clock:
            self.clock.set_bpm(clamped)

    def set_mapping_mode(self, mode: str) -> None:
        """Switch between 'finger' and 'zones' mapping modes."""
        if mode in ("finger", "zones"):
            self.mapping_mode = mode
            if self.audio:
                self.audio.set_mapping_mode(mode)
            print(f"[DrumEngine] Mapping mode changed to: {self.mapping_mode.upper()}")

    def toggle_mapping_mode(self) -> str:
        new_mode = "zones" if self.mapping_mode == "finger" else "finger"
        self.set_mapping_mode(new_mode)
        return self.mapping_mode

    def toggle_bpm_mode(self) -> str:
        """Toggle between (a) tap tempo and (b) hand height BPM control."""
        self.bpm_mode = "height" if self.bpm_mode == "tap" else "tap"
        print(f"[DrumEngine] BPM control mode: {self.bpm_mode.upper()}")
        return self.bpm_mode

    def toggle_quantize_mode(self) -> str:
        """Cycle hit quantization between 'none', '1/8', and '1/16'."""
        if self.clock:
            self.quantize_mode = self.clock.toggle_quantize()
        else:
            modes = ["none", "1/8", "1/16"]
            idx = modes.index(self.quantize_mode) if self.quantize_mode in modes else 0
            self.quantize_mode = modes[(idx + 1) % len(modes)]
        print(f"[DrumEngine] Quantize mode: {self.quantize_mode.upper()}")
        return self.quantize_mode

    def toggle_metronome_click(self) -> bool:
        """Toggle metronome audio click sound."""
        if self.clock:
            state = self.clock.toggle_click()
            print(f"[DrumEngine] Metronome click: {'ON' if state else 'OFF'}")
            return state
        return False

    def set_sensitivity_preset(self, preset_name: str) -> str:
        """Apply strike velocity sensitivity preset across right and left hand trackers."""
        if preset_name not in SENSITIVITY_PRESETS:
            return self.sensitivity_preset

        preset = SENSITIVITY_PRESETS[preset_name]
        self.sensitivity_preset = preset_name

        # Update right-hand finger trackers
        self.thumb_tracker.velocity_threshold = preset["thumb"]
        self.index_tracker.velocity_threshold = preset["index"]
        self.middle_tracker.velocity_threshold = preset["middle"]
        self.ring_tracker.velocity_threshold = preset["ring"]
        self.pinky_tracker.velocity_threshold = preset["pinky"]

        # Update left-hand finger trackers
        if hasattr(self, "left_thumb_tracker"):
            self.left_thumb_tracker.velocity_threshold = preset["thumb"]
            self.left_index_tracker.velocity_threshold = preset["index"]
            self.left_middle_tracker.velocity_threshold = preset["middle"]
            self.left_ring_tracker.velocity_threshold = preset["ring"]
            self.left_pinky_tracker.velocity_threshold = preset["pinky"]

        print(
            f"[DrumEngine] Sensitivity preset set to: {preset_name.upper()} "
            f"(Index={preset['index']}, Thumb={preset['thumb']})"
        )
        return self.sensitivity_preset

    def cycle_sensitivity(self) -> str:
        """Cycle sensitivity preset: High -> Ultra -> Normal -> High."""
        order = ["High", "Ultra", "Normal"]
        idx = order.index(self.sensitivity_preset) if self.sensitivity_preset in order else 0
        new_preset = order[(idx + 1) % len(order)]
        return self.set_sensitivity_preset(new_preset)

    def toggle_ar_pads(self) -> bool:
        """Toggle holographic AR drum pads on/off."""
        self.ar_pads_enabled = not self.ar_pads_enabled
        print(f"[DrumEngine] AR Drum Pads: {'ENABLED' if self.ar_pads_enabled else 'DISABLED'}")
        return self.ar_pads_enabled

    def toggle_two_hand_drums(self) -> bool:
        """Toggle two-handed drumming capability on/off."""
        self.two_hand_drums_enabled = not self.two_hand_drums_enabled
        print(f"[DrumEngine] Two-Handed Drumming: {'ENABLED' if self.two_hand_drums_enabled else 'DISABLED'}")
        return self.two_hand_drums_enabled

    def process_left_hand_drums(
        self,
        left_hand_data: Optional[Any],
        timestamp: Optional[float] = None,
    ) -> List[HitEvent]:
        """Process left hand landmarks for two-handed drumming when outside GUI menu.

        Args:
            left_hand_data: HandData instance from HandTracker (or None).
            timestamp: Frame timestamp (defaults to time.time()).

        Returns:
            List of HitEvent objects detected from left hand.
        """
        if timestamp is None:
            timestamp = time.time()

        if not self.two_hand_drums_enabled or left_hand_data is None:
            for tr in self.left_finger_trackers:
                tr.reset()
            return []

        # If left hand is inside the GUI menu zone (norm_x < 0.24 or pixel_x < 290),
        # reserve it exclusively for GUI navigation and disable drumming strikes
        wrist_norm_x = float(left_hand_data.landmarks[0, 0])
        idx_px_x = float(left_hand_data.pixel_landmarks[8, 0])
        if wrist_norm_x < 0.24 or idx_px_x < 290:
            for tr in self.left_finger_trackers:
                tr.reset()
            return []

        landmarks = left_hand_data.landmarks
        pixel_landmarks = left_hand_data.pixel_landmarks
        hits: List[HitEvent] = []

        for tr in self.left_finger_trackers:
            idx = tr.landmark_idx
            base_idx = tr.base_landmark_idx
            norm_x = float(landmarks[idx, 0])
            norm_y = float(landmarks[idx, 1])
            base_norm_y = (
                float(landmarks[base_idx, 1])
                if (base_idx is not None and landmarks.shape[0] > base_idx)
                else None
            )
            px = (int(pixel_landmarks[idx, 0]), int(pixel_landmarks[idx, 1]))

            hit = tr.update(norm_x, norm_y, px, timestamp, base_norm_y=base_norm_y)
            if hit is not None:
                hit.finger = f"L_{tr.name}"
                self._handle_hit_trigger(hit)
                hits.append(hit)
                if self.on_hit_callback:
                    try:
                        self.on_hit_callback(hit)
                    except Exception as cb_err:
                        print(f"⚠️ [DrumEngine] Left hit callback error: {cb_err}")

        return hits

    def process_right_hand(
        self,
        right_hand_data: Optional[Any],
        timestamp: Optional[float] = None,
    ) -> List[HitEvent]:
        """Process right hand landmarks: update BPM and detect index/middle hits.

        Args:
            right_hand_data: HandData instance from HandTracker (or None).
            timestamp: Frame timestamp (defaults to time.time()).

        Returns:
            List of HitEvent objects detected in this frame.
        """
        if timestamp is None:
            timestamp = time.time()

        if right_hand_data is None:
            for tr in self.finger_trackers:
                tr.reset()
            return []

        # BPM Control Mode (b): Hand height mapped to 60-180 BPM with smoothing
        if self.bpm_mode == "height":
            self._update_bpm_from_height(right_hand_data)

        # Extract landmarks: normalized (21, 3) and pixel (21, 2)
        landmarks = right_hand_data.landmarks
        pixel_landmarks = right_hand_data.pixel_landmarks

        hits: List[HitEvent] = []

        # Update all 5 finger strike trackers (Thumb, Index, Middle, Ring, Pinky)
        for tr in self.finger_trackers:
            idx = tr.landmark_idx
            base_idx = tr.base_landmark_idx
            norm_x = float(landmarks[idx, 0])
            norm_y = float(landmarks[idx, 1])
            base_norm_y = (
                float(landmarks[base_idx, 1])
                if (base_idx is not None and landmarks.shape[0] > base_idx)
                else None
            )
            px = (int(pixel_landmarks[idx, 0]), int(pixel_landmarks[idx, 1]))

            hit = tr.update(norm_x, norm_y, px, timestamp, base_norm_y=base_norm_y)
            if hit is not None:
                self._handle_hit_trigger(hit)
                hits.append(hit)
                if self.on_hit_callback:
                    try:
                        self.on_hit_callback(hit)
                    except Exception as cb_err:
                        print(f"⚠️ [DrumEngine] Hit callback error: {cb_err}")

        return hits

    def _update_bpm_from_height(self, right_hand_data: Any) -> None:
        """Map right hand vertical position to BPM 60-180 with EMA smoothing."""
        # Use wrist (landmark 0) for stable hand elevation
        wrist_y = float(right_hand_data.landmarks[0, 1])

        # In image coordinates, y=0 is top, y=1 is bottom
        # High hand (y <= BPM_HEIGHT_MIN_Y) -> 180 BPM
        # Low hand (y >= BPM_HEIGHT_MAX_Y) -> 60 BPM
        y_clamped = max(BPM_HEIGHT_MIN_Y, min(BPM_HEIGHT_MAX_Y, wrist_y))
        height_span = BPM_HEIGHT_MAX_Y - BPM_HEIGHT_MIN_Y
        norm_height = (BPM_HEIGHT_MAX_Y - y_clamped) / height_span
        self.current_hand_height_frac = norm_height

        target = BPM_MIN + norm_height * (BPM_MAX - BPM_MIN)
        self.target_height_bpm = target

        # Exponential moving average smoothing
        self._smooth_bpm = (
            BPM_HEIGHT_SMOOTHING * target
            + (1.0 - BPM_HEIGHT_SMOOTHING) * self._smooth_bpm
        )
        smoothed_int = int(round(self._smooth_bpm))
        if smoothed_int != self.bpm:
            self.set_bpm(smoothed_int)

    def _update_bpm_from_tap(self, timestamp: float) -> None:
        """Update tap tempo from time between hits (average of last 4 intervals)."""
        if self._tap_hit_times:
            dt = timestamp - self._tap_hit_times[-1]
            # Valid rhythmic interval between 150ms and 2.0s
            if 0.15 <= dt <= 2.0:
                self._tap_intervals.append(dt)
                if len(self._tap_intervals) >= BPM_TAP_HISTORY_COUNT:
                    avg_dt = sum(self._tap_intervals) / float(len(self._tap_intervals))
                    calc_bpm = int(round(60.0 / avg_dt))
                    new_bpm = max(BPM_MIN, min(BPM_MAX, calc_bpm))
                    self.set_bpm(new_bpm)
                    print(
                        f"[DrumEngine] Tap tempo updated: {new_bpm} BPM "
                        f"(avg of {len(self._tap_intervals)} intervals: {avg_dt:.3f}s)"
                    )
            else:
                # Reset interval chain if pause is too long
                self._tap_intervals.clear()

        self._tap_hit_times.append(timestamp)

    def _handle_hit_trigger(self, hit: HitEvent) -> None:
        """Process hit event: resolve drum, scale volume, quantize/play, record, and flash."""
        if hit.finger.startswith("L_"):
            clean_finger = hit.finger[2:]
            drum_name = LEFT_HAND_DRUM_MAP.get(clean_finger, "snare")
        else:
            drum_name = self.audio.resolve_drum(hit.finger, hit.norm_x, self.mapping_mode)

        volume = self.audio.velocity_to_volume(hit.velocity)
        hit.drum = drum_name
        hit.volume = volume

        # BPM Control Mode (a): Tap tempo interval calculation
        if self.bpm_mode == "tap" and not hit.finger.startswith("L_"):
            self._update_bpm_from_tap(hit.timestamp)

        # Trigger or quantize hit playback
        is_immediate = True
        if self.clock is not None:
            is_immediate = self.clock.schedule_hit(
                drum_name=drum_name,
                volume=volume,
                play_fn=self.audio.play_drum,
                timestamp=hit.timestamp,
            )
        else:
            self.audio.play_drum(drum_name, volume=volume)

        hit.quantized = not is_immediate

        # Looper recording hook
        if self.looper is not None and self.clock is not None:
            current_beat = self.clock.get_continuous_beat()
            self.looper.record_hit(drum_name, volume, current_beat)

        # Jam session recorder hook
        if self.jam_recorder is not None:
            self.jam_recorder.log_hit(drum_name, volume, hit.timestamp)

        # Rhythm Hero game scoring hook
        if self.rhythm_game is not None and getattr(self.rhythm_game, "is_active", False):
            self.rhythm_game.check_hit(drum_name, hit.timestamp)

        # Console logging
        q_tag = ""
        if not is_immediate:
            q_tag = f" [{self.quantize_mode} Quantized]"
        print(
            f"HIT {hit.finger} velocity={hit.velocity:.2f} "
            f"[{drum_name.upper()} vol={int(volume * 100)}%]{q_tag}"
        )

        self._register_flash(hit)

        # Holographic AR drum pad excitation and particle explosion
        if self.ar_pads_enabled and drum_name in self.ar_pads:
            pad = self.ar_pads[drum_name]
            pad["hit_glow"] = 1.0
            pad["compression"] = 1.0
            px, py = hit.pixel_pos
            pad_px = (int(pad["norm_pos"][0] * 1280), int(pad["norm_pos"][1] * 720))
            eff_px = (px, py) if (px > 0 and py > 0) else pad_px
            self._emit_shockwave(eff_px, pad["color"])
            self._emit_neon_particles(eff_px, pad["color"], count=15)

    def _register_flash(self, hit: HitEvent) -> None:
        """Register a visual screen flash event."""
        self._active_flashes.append(
            {
                "finger": hit.finger,
                "drum": hit.drum,
                "velocity": hit.velocity,
                "volume": hit.volume,
                "quantized": hit.quantized,
                "timestamp": hit.timestamp,
                "pixel_pos": hit.pixel_pos,
            }
        )

    def _emit_shockwave(self, center_px: Tuple[int, int], color: Tuple[int, int, int]) -> None:
        """Spawn an expanding shockwave ripple effect."""
        self.shockwaves.append(
            {
                "center": center_px,
                "color": color,
                "start_time": time.time(),
                "duration": 0.45,
                "max_radius": 95,
            }
        )

    def _emit_neon_particles(
        self, center_px: Tuple[int, int], color: Tuple[int, int, int], count: int = 15
    ) -> None:
        """Spawn radiant spark particles exploding outward on drum strike."""
        now = time.time()
        cx, cy = center_px
        for _ in range(count):
            angle = random.uniform(0, 2.0 * math.pi)
            speed = random.uniform(70.0, 250.0)
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)
            life = random.uniform(0.35, 0.70)
            size = random.randint(2, 4)
            self.neon_particles.append(
                {
                    "x": float(cx),
                    "y": float(cy),
                    "vx": vx,
                    "vy": vy,
                    "color": color,
                    "size": size,
                    "start_time": now,
                    "duration": life,
                }
            )

    def _render_shockwaves(self, frame: np.ndarray, current_time: float) -> None:
        """Draw expanding circular shockwaves."""
        active = []
        for sw in self.shockwaves:
            elapsed = current_time - sw["start_time"]
            if elapsed < sw["duration"]:
                active.append(sw)
                progress = elapsed / sw["duration"]
                alpha = max(0.0, 1.0 - progress)
                radius = int(14 + progress * sw["max_radius"])
                c = sw["color"]
                draw_color = (int(c[0] * alpha), int(c[1] * alpha), int(c[2] * alpha))
                cv2.circle(frame, sw["center"], radius, draw_color, max(1, int(3 * alpha)), cv2.LINE_AA)
        self.shockwaves = active

    def _render_neon_particles(self, frame: np.ndarray, current_time: float) -> None:
        """Draw radiating neon spark particles with gentle gravity."""
        active = []
        h, w = frame.shape[:2]
        for p in self.neon_particles:
            elapsed = current_time - p["start_time"]
            if elapsed < p["duration"]:
                active.append(p)
                progress = elapsed / p["duration"]
                alpha = max(0.0, 1.0 - progress)
                cur_x = int(p["x"] + p["vx"] * elapsed)
                cur_y = int(p["y"] + p["vy"] * elapsed + 60.0 * elapsed * elapsed)
                if 0 <= cur_x < w and 0 <= cur_y < h:
                    c = p["color"]
                    draw_c = (int(c[0] * alpha), int(c[1] * alpha), int(c[2] * alpha))
                    cv2.circle(frame, (cur_x, cur_y), p["size"], draw_c, -1, cv2.LINE_AA)
        self.neon_particles = active

    def _render_ar_drum_pads(self, frame: np.ndarray, current_time: float) -> None:
        """Render floating holographic sci-fi drum pads in camera space."""
        h, w = frame.shape[:2]

        for drum_name, pad in self.ar_pads.items():
            # Update decay
            if pad["hit_glow"] > 0:
                pad["hit_glow"] = max(0.0, pad["hit_glow"] - 0.05)
            if pad["compression"] > 0:
                pad["compression"] = max(0.0, pad["compression"] - 0.06)

            nx, ny = pad["norm_pos"]
            cx = int(nx * w)
            cy = int(ny * h)
            base_r = pad["radius"]
            # Dynamic bounce compression on impact
            r = int(base_r * (1.0 - 0.18 * pad["compression"]))

            base_color = pad["color"]
            glow = pad["hit_glow"]

            # Holographic glassmorphic fill
            overlay = frame.copy()
            fill_alpha = 0.20 + 0.45 * glow
            fill_color = tuple(min(255, int(c * (0.6 + 0.4 * glow))) for c in base_color)
            cv2.circle(overlay, (cx, cy), r, fill_color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0, frame)

            # Outer glow and border rings
            border_thick = 2 if glow < 0.3 else 3
            border_color = (255, 255, 255) if glow > 0.6 else base_color
            cv2.circle(frame, (cx, cy), r, border_color, border_thick, cv2.LINE_AA)
            cv2.circle(
                frame,
                (cx, cy),
                max(2, r - 6),
                (border_color[0] // 2, border_color[1] // 2, border_color[2] // 2),
                1,
                cv2.LINE_AA,
            )

            # Center target crosshairs
            cv2.circle(frame, (cx, cy), 4, border_color, -1, cv2.LINE_AA)
            tick_len = 8
            cv2.line(frame, (cx - r - 4, cy), (cx - r + tick_len, cy), border_color, 1, cv2.LINE_AA)
            cv2.line(frame, (cx + r - tick_len, cy), (cx + r + 4, cy), border_color, 1, cv2.LINE_AA)
            cv2.line(frame, (cx, cy - r - 4), (cx, cy - r + tick_len), border_color, 1, cv2.LINE_AA)
            cv2.line(frame, (cx, cy + r - tick_len), (cx, cy + r + 4), border_color, 1, cv2.LINE_AA)

            # Instrument label: e.g. "KICK" and finger tag "[THB]"
            name_text = pad["display"]
            (tw, th), _ = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
            cv2.putText(
                frame,
                name_text,
                (cx - tw // 2, cy + 4),
                cv2.FONT_HERSHEY_DUPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            finger_text = f"[{pad['finger']}]"
            (fw, fh), _ = cv2.getTextSize(finger_text, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)
            cv2.putText(
                frame,
                finger_text,
                (cx - fw // 2, cy + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.33,
                (200, 220, 255),
                1,
                cv2.LINE_AA,
            )

    def draw_debug_overlay(
        self,
        frame: np.ndarray,
        right_hand_data: Optional[Any],
        current_time: Optional[float] = None,
    ) -> np.ndarray:
        """Render live velocity meters, metronome beat HUD, zones, AR pads, and screen flash effects."""
        if current_time is None:
            current_time = time.time()

        h, w = frame.shape[:2]

        # 1. Render Horizontal Hand Zones if in "zones" mapping mode
        if self.mapping_mode == "zones":
            self._render_horizontal_zones(frame, w, h, right_hand_data)

        # 2. Render Holographic AR Drum Pads & Neon Particles/Shockwaves
        if self.ar_pads_enabled:
            self._render_ar_drum_pads(frame, current_time)
        self._render_shockwaves(frame, current_time)
        self._render_neon_particles(frame, current_time)

        # 3. Render Height Gauge if in "height" BPM mode
        if self.bpm_mode == "height":
            self._render_height_bpm_gauge(frame, w, h)

        # 4. Render Screen Flash Effect if any hits are active
        frame = self._render_screen_flashes(frame, current_time)

        # 5. Render Live Fingertip Velocity Indicators on Hand
        if right_hand_data is not None:
            self._render_fingertip_indicators(frame)

        # 6. Render Master Clock & Drum Tuning HUD Panel
        self._render_tuning_hud(frame, w, h, current_time)

        return frame

    def _render_height_bpm_gauge(self, frame: np.ndarray, w: int, h: int) -> None:
        """Draw vertical tempo gauge bar on the right side for height-based BPM control."""
        bar_x = w - 18
        y_top = int(BPM_HEIGHT_MIN_Y * h)
        y_bot = int(BPM_HEIGHT_MAX_Y * h)
        bar_h = y_bot - y_top

        # Background track
        cv2.rectangle(frame, (bar_x, y_top), (bar_x + 8, y_bot), (35, 35, 45), -1)
        cv2.rectangle(frame, (bar_x, y_top), (bar_x + 8, y_bot), (90, 90, 110), 1)

        # Active height indicator
        curr_y = int(y_bot - self.current_hand_height_frac * bar_h)
        cv2.line(frame, (bar_x - 6, curr_y), (bar_x + 14, curr_y), (0, 255, 200), 2)

        # Labels
        cv2.putText(
            frame,
            "180",
            (bar_x - 32, y_top + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "60",
            (bar_x - 26, y_bot),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (160, 160, 180),
            1,
            cv2.LINE_AA,
        )

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

        # Zone labels at bottom
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
                    cv2.rectangle(
                        overlay, (0, 0), (w - 1, h - 1), COLOR_HIT_FLASH, border_thick
                    )
                    cv2.addWeighted(
                        overlay, alpha * 0.7, frame, 1.0 - (alpha * 0.7), 0, frame
                    )

                # Expanding ripple circle at fingertip
                px, py = fl["pixel_pos"]
                radius = int(20 + 70 * progress)
                ripple_alpha = alpha
                ripple_color = (
                    int(COLOR_HIT_FLASH[0] * ripple_alpha),
                    int(COLOR_HIT_FLASH[1] * ripple_alpha),
                    int(COLOR_HIT_FLASH[2] * ripple_alpha),
                )
                cv2.circle(
                    frame,
                    (px, py),
                    radius,
                    ripple_color,
                    max(1, int(3 * alpha)),
                    cv2.LINE_AA,
                )

                # Hit tag text
                vol_pct = int(fl.get("volume", 1.0) * 100)
                q_note = " (Q)" if fl.get("quantized") else ""
                tag_text = (
                    f"HIT {fl['drum'].upper()}! (v={fl['velocity']:.2f}, vol={vol_pct}%{q_note})"
                )
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
        """Render floating velocity badges next to all 5 fingertips of the right hand."""
        trackers = self.finger_trackers

        for tr in trackers:
            px, py = tr.current_pixel_pos
            if px == 0 and py == 0:
                continue

            v = tr.velocity
            sign = "+" if v >= 0 else ""
            target_drum = self.audio.resolve_drum(
                tr.name, tr.current_norm_x, self.mapping_mode
            )
            label = f"{tr.name[:3].upper()}->{target_drum[:4].upper()}: {sign}{v:.2f}"

            if tr.state == "ARMED":
                badge_bg = (0, 165, 255)
                badge_fg = (255, 255, 255)
            elif tr.state == "COOLDOWN":
                badge_bg = (80, 80, 95)
                badge_fg = (200, 200, 200)
            elif v >= tr.velocity_threshold:
                badge_bg = (40, 200, 100)
                badge_fg = (0, 0, 0)
            else:
                badge_bg = (35, 35, 45)
                badge_fg = (220, 220, 220)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            box_x = px + 10
            box_y = py - 6
            cv2.rectangle(
                frame,
                (box_x, box_y - th - 4),
                (box_x + tw + 6, box_y + 4),
                badge_bg,
                -1,
            )
            cv2.rectangle(
                frame,
                (box_x, box_y - th - 4),
                (box_x + tw + 6, box_y + 4),
                (180, 180, 180),
                1,
            )
            cv2.putText(
                frame,
                label,
                (box_x + 3, box_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                badge_fg,
                1,
                cv2.LINE_AA,
            )

            # Downward motion vector arrow on downward flick
            if v > 0.10:
                arrow_len = min(60, int(v * 45))
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
        """Render master clock metronome, BPM mode, and 5-finger velocity meter HUD."""
        panel_w = 340
        panel_h = 286
        panel_x = w - panel_w - 28
        panel_y = 60

        # Background panel
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

        # Clock Visual State
        clock_state = (
            self.clock.get_visual_state()
            if self.clock
            else {"bpm": self.bpm, "beat_index": 0, "is_flash": False, "click_enabled": True}
        )

        # Row 1: Header & BPM Display
        bpm_str = f"BPM: {clock_state['bpm']}"
        cv2.putText(
            frame,
            "MASTER CLOCK",
            (panel_x + 12, panel_y + 20),
            cv2.FONT_HERSHEY_DUPLEX,
            0.46,
            COLOR_SECONDARY,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            bpm_str,
            (panel_x + 140, panel_y + 21),
            cv2.FONT_HERSHEY_DUPLEX,
            0.55,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )

        # 4-Beat LED visual metronome: [ ● ○ ○ ○ ]
        beat_idx = clock_state["beat_index"]
        is_flash = clock_state["is_flash"]
        led_start_x = panel_x + 240
        for b in range(4):
            led_x = led_start_x + b * 22
            led_y = panel_y + 17
            if b == beat_idx:
                led_c = (0, 255, 255) if is_flash else (0, 200, 140)
                cv2.circle(frame, (led_x, led_y), 6, led_c, -1, cv2.LINE_AA)
                cv2.circle(frame, (led_x, led_y), 7, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (led_x, led_y), 4, (60, 60, 75), -1, cv2.LINE_AA)

        # Row 2: BPM Mode & Quantize Info
        if self.bpm_mode == "tap":
            n_taps = len(self._tap_intervals)
            bpm_mode_str = f"BPM Mode: TAP ({n_taps}/4 avg) [B]"
        else:
            bpm_mode_str = f"BPM Mode: HEIGHT (Tgt:{int(self.target_height_bpm)}) [B]"

        cv2.putText(
            frame,
            bpm_mode_str,
            (panel_x + 12, panel_y + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 200, 100),
            1,
            cv2.LINE_AA,
        )

        q_info = f"Quant: {self.quantize_mode.upper()} [G] | Click: {'ON' if clock_state['click_enabled'] else 'OFF'} [C]"
        cv2.putText(
            frame,
            q_info,
            (panel_x + 12, panel_y + 53),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (180, 180, 220),
            1,
            cv2.LINE_AA,
        )

        # Row 3: Mapping Mode
        mode_desc = (
            "5-FINGER KIT (THB=Kik, IDX=Snr, MID=Hat, RNG=Tom, PNK=Csh)"
            if self.mapping_mode == "finger"
            else "ZONES (L=Kick, M=Snare, R=Hat)"
        )
        cv2.putText(
            frame,
            f"Drum Map: {mode_desc} [M]",
            (panel_x + 12, panel_y + 67),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.33,
            (0, 255, 180),
            1,
            cv2.LINE_AA,
        )

        # Row 4: Kit, Sensitivity & 2-Hand info
        two_hand_str = "ON" if self.two_hand_drums_enabled else "OFF"
        kit_info = (
            f"Kit: {self.audio.current_kit} [V] | "
            f"Sens: {self.sensitivity_preset} [E] | 2-Hand: {two_hand_str} [Y]"
        )
        cv2.putText(
            frame,
            kit_info,
            (panel_x + 12, panel_y + 81),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (255, 180, 220),
            1,
            cv2.LINE_AA,
        )

        # Row 5: All 5 Finger Velocity Meters (Thumb, Index, Middle, Ring, Pinky)
        finger_displays = [
            ("THUMB  #4", self.thumb_tracker),
            ("INDEX  #8", self.index_tracker),
            ("MIDDLE #12", self.middle_tracker),
            ("RING   #16", self.ring_tracker),
            ("PINKY  #20", self.pinky_tracker),
        ]

        meter_y_start = panel_y + 94
        for i, (label_name, tracker) in enumerate(finger_displays):
            self._render_compact_finger_meter(
                frame,
                tracker=tracker,
                display_name=label_name,
                x=panel_x + 12,
                y=meter_y_start + i * 36,
                bar_w=panel_w - 24,
                current_time=current_time,
            )


    def _render_compact_finger_meter(
        self,
        frame: np.ndarray,
        tracker: FingerStrikeTracker,
        display_name: str,
        x: int,
        y: int,
        bar_w: int,
        current_time: float,
    ) -> None:
        """Render a compact live velocity bar, threshold mark, and state badge for a finger."""
        bar_h = 7
        max_meter_v = max(0.85, tracker.velocity_threshold * 2.5)

        target_drum = self.audio.resolve_drum(
            tracker.name, tracker.current_norm_x, self.mapping_mode
        )
        dyn_vol = int(self.audio.velocity_to_volume(tracker.velocity) * 100)

        sign = "+" if tracker.velocity >= 0 else ""
        vel_text = f"{display_name}->{target_drum.upper()}: {sign}{tracker.velocity:.2f} ({dyn_vol}%)"
        cv2.putText(
            frame,
            vel_text,
            (x, y + 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )

        # State pill badge
        state_label = tracker.state
        if tracker.state == "COOLDOWN":
            state_label = "COOL"
            badge_color = (90, 90, 100)
        elif tracker.state == "ARMED":
            state_label = "ARMED"
            badge_color = (0, 180, 255)
        elif tracker.velocity >= tracker.velocity_threshold:
            state_label = "FAST"
            badge_color = COLOR_SUCCESS
        else:
            state_label = "IDLE"
            badge_color = (45, 45, 55)

        (sw, _), _ = cv2.getTextSize(state_label, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)
        badge_x = x + bar_w - sw - 8
        cv2.rectangle(
            frame,
            (badge_x, y),
            (badge_x + sw + 6, y + 13),
            badge_color,
            -1,
        )
        cv2.putText(
            frame,
            state_label,
            (badge_x + 3, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Bar background
        bar_y = y + 16
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
            elif tracker.velocity >= tracker.velocity_threshold:
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

        # Threshold tick indicator
        thresh_frac = tracker.velocity_threshold / max_meter_v
        thresh_x = int(x + thresh_frac * bar_w)
        cv2.line(
            frame,
            (thresh_x, bar_y - 2),
            (thresh_x, bar_y + bar_h + 2),
            (0, 255, 255),
            1,
            cv2.LINE_AA,
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
        """Alias for backward compatibility."""
        self._render_compact_finger_meter(
            frame, tracker, display_name, x, y, bar_w, current_time
        )

    def cleanup(self) -> None:
        """Stop metronome clock thread and cleanup audio resources."""
        if self.clock:
            self.clock.stop()
        if self.audio:
            self.audio.cleanup()


def run_drum_engine_test(
    camera_index: int = 0,
    max_frames: Optional[int] = None,
) -> None:
    """Standalone live test runner for drum engine, metronome clock, and BPM modes."""
    from hand_tracker import HandTracker

    print("[AirBeat] Initializing Drum Engine with Metronome Clock...")
    tracker = HandTracker(mirrored=True, use_one_euro_filter=True)
    if not tracker.initialize():
        print("❌ [AirBeat] Failed to initialize HandTracker.")
        return

    audio_manager = AudioManager()
    if not audio_manager.initialize():
        print("❌ [AirBeat] Failed to initialize AudioManager.")
        return

    clock = MetronomeClock(bpm=DEFAULT_BPM, quantize_mode=QUANTIZE_MODE)
    clock.start()

    drum_engine = DrumEngine(audio_manager=audio_manager, clock=clock)

    cap = None
    if hasattr(cv2, "CAP_AVFOUNDATION"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"❌ [AirBeat] Could not open camera at index {camera_index}.")
        clock.stop()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "AirBeat - Drum Engine, Metronome & BPM Controls"
    window_supported = True
    frame_count = 0

    print("=" * 60)
    print("🥁 AirBeat Drum Engine & Master Clock Live")
    print(f"   BPM:                {drum_engine.bpm}")
    print(f"   BPM Control Mode:   {drum_engine.bpm_mode.upper()} ('b' to toggle)")
    print(f"   Mapping Mode:       {drum_engine.mapping_mode.upper()} ('m' to toggle)")
    print(f"   Quantize:           {drum_engine.quantize_mode.upper()} ('g' to toggle)")
    print("   Hotkeys:")
    print("   [B] Toggle BPM mode (Tap Tempo <-> Hand Height)")
    print("   [M] Toggle Drum Mapping (Finger <-> Horizontal Zones)")
    print("   [G] Toggle Quantization (None -> 1/8 -> 1/16)")
    print("   [C] Toggle Metronome Click track")
    print("   [Q/ESC] Quit")
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

            # Draw drum tuning overlay (meters, zones, metronome HUD, flash)
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
                        drum_engine.toggle_mapping_mode()
                    elif key in (ord("b"), ord("B")):
                        drum_engine.toggle_bpm_mode()
                    elif key in (ord("g"), ord("G")):
                        drum_engine.toggle_quantize_mode()
                    elif key in (ord("c"), ord("C")):
                        drum_engine.toggle_metronome_click()
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
        drum_engine.cleanup()
        print(f"\n✅ [AirBeat] Drum test completed ({frame_count} frames processed).")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AirBeat Drum Engine & Metronome Live")
    parser.add_argument(
        "--camera-index", type=int, default=0, help="Camera index (default 0)"
    )
    parser.add_argument(
        "--test-frames", type=int, default=None, help="Stop after N frames"
    )
    args = parser.parse_args()

    run_drum_engine_test(camera_index=args.camera_index, max_frames=args.test_frames)
