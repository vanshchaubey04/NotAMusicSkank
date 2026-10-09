"""Threaded Metronome and Master Clock for AirBeat.

Provides:
- Thread-safe background clock running on high-precision time.perf_counter().
- Procedurally synthesized woodblock click track (downbeat + offbeat clicks).
- Grid timing tracking (quarter, 1/8, and 1/16 notes).
- Optional hit quantization to 1/8 or 1/16 note beat grid.
- Visual beat pulsing state for GUI and HUD displays.
"""

from collections import deque
import io
import math
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import wave

import numpy as np
import pygame

from config import (
    DEFAULT_BPM,
    METRONOME_CLICK_ENABLED,
    METRONOME_CLICK_VOLUME,
    QUANTIZE_MODE,
)


class MetronomeClock:
    """Thread-safe metronome clock running in a separate background thread."""

    def __init__(
        self,
        bpm: int = DEFAULT_BPM,
        click_enabled: bool = METRONOME_CLICK_ENABLED,
        click_volume: float = METRONOME_CLICK_VOLUME,
        quantize_mode: str = QUANTIZE_MODE,
        audio_callback: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        self._bpm: float = float(bpm)
        self.click_enabled: bool = click_enabled
        self.click_volume: float = click_volume
        self.quantize_mode: str = quantize_mode
        self.audio_callback = audio_callback

        # Threading state
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Timing and beat state
        self._beat_index: int = 0  # 0, 1, 2, 3
        self._sixteenth_index: int = 0  # 0 to 15
        self._last_beat_time: float = time.perf_counter()
        self._last_sixteenth_time: float = time.perf_counter()

        # Quantized scheduled hits queue: (scheduled_time, drum_name, volume)
        self._scheduled_queue: deque = deque()

        # Click audio sounds
        self._click_high: Optional[pygame.mixer.Sound] = None
        self._click_low: Optional[pygame.mixer.Sound] = None
        self._init_click_sounds()

    def _init_click_sounds(self) -> None:
        """Procedurally synthesize metronome downbeat and offbeat clicks."""
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()

            sr = 44100
            dur = 0.025
            t = np.linspace(0, dur, int(sr * dur), endpoint=False)

            # High click for downbeat (beat 1)
            sig_high = np.sin(2 * np.pi * 1400.0 * t) * np.exp(-150.0 * t)
            i16_high = (np.clip(sig_high, -1.0, 1.0) * 32767).astype(np.int16)
            stereo_high = np.column_stack((i16_high, i16_high))

            # Low click for beats 2, 3, 4
            sig_low = np.sin(2 * np.pi * 920.0 * t) * np.exp(-150.0 * t)
            i16_low = (np.clip(sig_low, -1.0, 1.0) * 32767).astype(np.int16)
            stereo_low = np.column_stack((i16_low, i16_low))

            self._click_high = self._samples_to_sound(stereo_high, sr)
            self._click_low = self._samples_to_sound(stereo_low, sr)
        except Exception as e:
            print(f"⚠️ [Metronome] Click sound synthesis error: {e}")

    @staticmethod
    def _samples_to_sound(stereo_samples: np.ndarray, sr: int) -> pygame.mixer.Sound:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(stereo_samples.tobytes())
        buf.seek(0)
        return pygame.mixer.Sound(buf)

    @property
    def bpm(self) -> int:
        with self._lock:
            return int(round(self._bpm))

    def set_bpm(self, new_bpm: float) -> None:
        with self._lock:
            self._bpm = max(40.0, min(240.0, float(new_bpm)))

    def set_quantize_mode(self, mode: str) -> None:
        if mode in ("none", "1/8", "1/16"):
            with self._lock:
                self.quantize_mode = mode

    def toggle_click(self) -> bool:
        with self._lock:
            self.click_enabled = not self.click_enabled
            return self.click_enabled

    def toggle_quantize(self) -> str:
        modes = ["none", "1/8", "1/16"]
        with self._lock:
            current_idx = modes.index(self.quantize_mode) if self.quantize_mode in modes else 0
            self.quantize_mode = modes[(current_idx + 1) % len(modes)]
            return self.quantize_mode

    def start(self) -> None:
        """Start the metronome thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="MetronomeClockThread", daemon=True)
        self._thread.start()
        print(f"[Metronome] Clock thread started at {self.bpm} BPM.")

    def stop(self) -> None:
        """Stop the metronome thread cleanly."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None
        print("[Metronome] Clock thread stopped.")

    def _run_loop(self) -> None:
        """High-precision timing loop driving sixteenth-note grid and beat clicks."""
        next_tick = time.perf_counter()

        while not self._stop_event.is_set():
            now = time.perf_counter()

            with self._lock:
                current_bpm = self._bpm
                click_on = self.click_enabled
                vol = self.click_volume

            # Calculate subdivision intervals (seconds)
            beat_interval = 60.0 / current_bpm
            sixteenth_interval = beat_interval / 4.0

            # 1. Fire due scheduled quantized drum hits
            with self._lock:
                while self._scheduled_queue and self._scheduled_queue[0][0] <= now:
                    _, drum_name, drum_vol, play_fn = self._scheduled_queue.popleft()
                    try:
                        play_fn(drum_name, drum_vol)
                    except Exception:
                        pass

            # 2. Process grid tick
            if now >= next_tick:
                self._sixteenth_index = (self._sixteenth_index + 1) % 16
                self._last_sixteenth_time = now

                # Every 4 sixteenths is a quarter-note beat
                if self._sixteenth_index % 4 == 0:
                    with self._lock:
                        self._beat_index = (self._beat_index + 1) % 4
                        current_beat = self._beat_index
                    self._last_beat_time = now

                    # Play audible click
                    if click_on:
                        if current_beat == 0 and self._click_high:
                            ch = self._click_high.play()
                            if ch:
                                ch.set_volume(vol)
                        elif self._click_low:
                            ch = self._click_low.play()
                            if ch:
                                ch.set_volume(vol * 0.7)

                next_tick = now + sixteenth_interval

            # Sleep briefly for low CPU usage while maintaining sub-millisecond precision
            time_to_next = max(0.0005, min(0.002, next_tick - time.perf_counter()))
            time.sleep(time_to_next)

    def schedule_hit(
        self,
        drum_name: str,
        volume: float,
        play_fn: Callable[[str, float], None],
        timestamp: Optional[float] = None,
    ) -> bool:
        """Trigger or quantize a hit based on current quantize_mode.

        Returns:
            bool: True if played immediately, False if scheduled for upcoming grid tick.
        """
        now = time.perf_counter() if timestamp is None else timestamp

        with self._lock:
            mode = self.quantize_mode
            current_bpm = self._bpm

        if mode == "none":
            play_fn(drum_name, volume)
            return True

        beat_interval = 60.0 / current_bpm
        subdiv_interval = beat_interval / (2.0 if mode == "1/8" else 4.0)

        # Calculate time since previous grid tick and time to next grid tick
        with self._lock:
            last_grid = self._last_sixteenth_time

        time_since_grid = (now - last_grid) % subdiv_interval
        time_to_next_grid = subdiv_interval - time_since_grid

        # Tolerance window: if within 40ms of grid point, play immediately
        if time_since_grid <= 0.040:
            play_fn(drum_name, volume)
            return True
        elif time_to_next_grid <= 0.040:
            # Very close to next grid tick, play immediately
            play_fn(drum_name, volume)
            return True
        else:
            # Schedule for next upcoming grid tick
            target_time = now + time_to_next_grid
            with self._lock:
                self._scheduled_queue.append((target_time, drum_name, volume, play_fn))
            return False

    def get_visual_state(self) -> Dict[str, Any]:
        """Return visual state snapshot for GUI and HUD rendering."""
        now = time.perf_counter()
        with self._lock:
            current_bpm = int(round(self._bpm))
            beat = self._beat_index
            click_on = self.click_enabled
            q_mode = self.quantize_mode

        beat_interval = 60.0 / max(1.0, current_bpm)
        elapsed = now - self._last_beat_time
        progress = max(0.0, min(1.0, elapsed / beat_interval))
        is_flash = elapsed < 0.08  # Flash for 80ms on each beat

        return {
            "bpm": current_bpm,
            "beat_index": beat,
            "progress": progress,
            "is_flash": is_flash,
            "click_enabled": click_on,
            "quantize_mode": q_mode,
        }
