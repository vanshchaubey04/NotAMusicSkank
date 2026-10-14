"""Live Beat Looper and Audio Jam Session Recorder for AirBeat.

Provides:
- LiveLooper: 4-bar (16-beat) loop recorder that records real-time air drum hits
  and plays them back in synchronized loops with the metronome clock.
- JamRecorder: High-fidelity performance recorder that records sessions
  and exports finished stereo WAV audio files directly to the disk.
"""

from dataclasses import dataclass
from datetime import datetime
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import wave
import numpy as np

from config import AUDIO_SAMPLE_RATE, LOOPER_DEFAULT_BARS, RECORDINGS_DIR


@dataclass
class LoopHit:
    """Represents a scheduled hit within a 16-beat loop."""

    beat_offset: float  # Normalized beat offset in loop [0.0, total_beats)
    drum_name: str
    volume: float


class LiveLooper:
    """4-bar (16-beat) real-time air drum beat looper."""

    def __init__(self, bars: int = LOOPER_DEFAULT_BARS) -> None:
        self.bars = bars
        self.total_beats: float = float(bars * 4)  # 16 beats for 4 bars
        self.state: str = "OFF"  # "OFF", "REC", "PLAY"
        self.loop_hits: List[LoopHit] = []
        self.start_beat: Optional[float] = None
        self._last_checked_beat: float = -1.0
        self.on_trigger_callback: Optional[Callable[[str, float], None]] = None

    def toggle(self) -> str:
        """Cycle looper mode: OFF -> REC -> PLAY -> OFF."""
        if self.state == "OFF":
            self.state = "REC"
            self.loop_hits.clear()
            self.start_beat = None
            print("[Looper] State: REC (Arm recording loop on next hit)")
        elif self.state == "REC":
            self.state = "PLAY"
            print(f"[Looper] State: PLAY ({len(self.loop_hits)} hits recorded)")
        else:
            self.state = "OFF"
            print("[Looper] State: OFF")
        return self.state

    def record_hit(self, drum_name: str, volume: float, current_beat: float) -> None:
        """Record a live drum hit into the loop if currently in REC mode."""
        if self.state != "REC":
            return

        if self.start_beat is None:
            # Anchor loop start to integer beat boundary
            self.start_beat = float(int(current_beat))
            self._last_checked_beat = self.start_beat

        offset = (current_beat - self.start_beat) % self.total_beats
        self.loop_hits.append(LoopHit(beat_offset=offset, drum_name=drum_name, volume=volume))
        print(f"[Looper] Recorded {drum_name.upper()} at beat {offset:.2f} (Total: {len(self.loop_hits)})")

    def update_clock(self, current_beat: float) -> List[Tuple[str, float]]:
        """Check for hits that fall between last beat and current beat in loop."""
        if self.state != "PLAY" or not self.loop_hits or self.start_beat is None:
            self._last_checked_beat = current_beat
            return []

        triggered: List[Tuple[str, float]] = []

        if self._last_checked_beat < 0:
            self._last_checked_beat = current_beat

        prev_loop_pos = (self._last_checked_beat - self.start_beat) % self.total_beats
        curr_loop_pos = (current_beat - self.start_beat) % self.total_beats

        for hit in self.loop_hits:
            hit_pos = hit.beat_offset
            # Handle normal advancement or loop wrap-around
            should_play = False
            if prev_loop_pos <= curr_loop_pos:
                if prev_loop_pos < hit_pos <= curr_loop_pos:
                    should_play = True
            else:
                # Wrap around beat 16 -> beat 0
                if hit_pos > prev_loop_pos or hit_pos <= curr_loop_pos:
                    should_play = True

            if should_play:
                triggered.append((hit.drum_name, hit.volume))
                if self.on_trigger_callback:
                    try:
                        self.on_trigger_callback(hit.drum_name, hit.volume)
                    except Exception as e:
                        print(f"⚠️ [Looper] Callback error: {e}")

        self._last_checked_beat = current_beat
        return triggered

    def clear(self) -> None:
        """Clear recorded loop."""
        self.loop_hits.clear()
        self.start_beat = None
        self.state = "OFF"
        print("[Looper] Loop cleared and stopped.")


class JamRecorder:
    """Records performance jam sessions and exports to standard 16-bit stereo WAV."""

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        output_dir: str = RECORDINGS_DIR,
    ) -> None:
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        self.is_recording: bool = False
        self.start_time: float = 0.0
        self.events: List[Dict[str, Any]] = []
        os.makedirs(self.output_dir, exist_ok=True)

    def toggle(self) -> bool:
        """Toggle jam recording on / off."""
        if not self.is_recording:
            return self.start()
        else:
            self.stop()
            return False

    def start(self) -> bool:
        """Start recording jam performance."""
        self.is_recording = True
        self.start_time = time.time()
        self.events.clear()
        print(f"[JamRecorder] ⏺ Recording STARTED at {datetime.now().strftime('%H:%M:%S')}")
        return True

    def log_hit(self, drum_name: str, volume: float, timestamp: Optional[float] = None) -> None:
        """Log a drum strike event during recording."""
        if not self.is_recording:
            return
        t = timestamp if timestamp is not None else time.time()
        elapsed = max(0.0, t - self.start_time)
        self.events.append(
            {"type": "drum", "drum": drum_name, "volume": volume, "offset_sec": elapsed}
        )

    def log_backing(self, style: str, key: str, bpm: int, timestamp: Optional[float] = None) -> None:
        """Log backing track state change."""
        if not self.is_recording:
            return
        t = timestamp if timestamp is not None else time.time()
        elapsed = max(0.0, t - self.start_time)
        self.events.append(
            {"type": "backing", "style": style, "key": key, "bpm": bpm, "offset_sec": elapsed}
        )

    def stop(self) -> Optional[str]:
        """Stop recording and export session to WAV file.

        Returns:
            Saved absolute filepath or None if no recording.
        """
        if not self.is_recording:
            return None

        self.is_recording = False
        total_duration = max(1.0, time.time() - self.start_time)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"AirBeat_Jam_{timestamp_str}.wav"
        filepath = os.path.join(self.output_dir, filename)

        # Synthesize audio session mix into WAV file
        sr = self.sample_rate
        total_samples = int(total_duration * sr)
        mix_left = np.zeros(total_samples, dtype=np.float32)
        mix_right = np.zeros(total_samples, dtype=np.float32)

        # Synthesize logged drum hits into timeline
        for ev in self.events:
            if ev.get("type") != "drum":
                continue
            drum = ev.get("drum", "kick")
            vol = ev.get("volume", 0.8)
            offset_sec = ev.get("offset_sec", 0.0)
            start_idx = int(offset_sec * sr)
            if start_idx >= total_samples:
                continue

            hit_sound = self._synthesize_offline_hit(drum, sr)
            sound_len = len(hit_sound)
            end_idx = min(total_samples, start_idx + sound_len)
            chunk_len = end_idx - start_idx
            mix_left[start_idx:end_idx] += hit_sound[:chunk_len] * vol
            mix_right[start_idx:end_idx] += hit_sound[:chunk_len] * vol

        # Peak normalize to -0.5 dB to prevent clipping
        peak = max(np.max(np.abs(mix_left)), np.max(np.abs(mix_right)), 0.01)
        norm_factor = 0.94 / peak if peak > 0.94 else 1.0
        mix_left = np.clip(mix_left * norm_factor, -1.0, 1.0)
        mix_right = np.clip(mix_right * norm_factor, -1.0, 1.0)

        # Write stereo 16-bit PCM WAV
        i16_left = (mix_left * 32767).astype(np.int16)
        i16_right = (mix_right * 32767).astype(np.int16)
        stereo = np.column_stack((i16_left, i16_right))

        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(stereo.tobytes())

        print(
            f"✅ [JamRecorder] Jam session saved successfully ({total_duration:.1f}s, "
            f"{len(self.events)} events):\n  -> {filepath}"
        )
        return filepath

    def _synthesize_offline_hit(self, drum: str, sr: int) -> np.ndarray:
        """Generate high-quality render buffer for offline export."""
        if drum == "kick":
            dur = 0.35
            t = np.linspace(0, dur, int(sr * dur), endpoint=False)
            f = 145.0 * np.exp(-18.0 * t) + 38.0
            phase = 2 * np.pi * np.cumsum(f) / sr
            body = np.sin(phase) * np.exp(-8.5 * t)
            click = np.sin(2 * np.pi * 1100.0 * t) * np.exp(-140.0 * t)
            return np.tanh(1.5 * (0.85 * body + 0.15 * click))
        elif drum == "snare":
            dur = 0.25
            t = np.linspace(0, dur, int(sr * dur), endpoint=False)
            body = np.sin(2 * np.pi * 200.0 * t) * np.exp(-22.0 * t)
            noise = np.random.uniform(-1.0, 1.0, len(t))
            noise = np.diff(noise, prepend=0.0)
            return np.tanh(1.3 * (0.38 * body + 0.62 * noise * np.exp(-13.0 * t)))
        elif drum == "hihat":
            dur = 0.08
            t = np.linspace(0, dur, int(sr * dur), endpoint=False)
            noise = np.random.uniform(-1.0, 1.0, len(t))
            noise = noise - 0.85 * np.roll(noise, 1)
            return np.clip(noise * np.exp(-48.0 * t) * 1.5, -1.0, 1.0)
        elif drum == "tom":
            dur = 0.30
            t = np.linspace(0, dur, int(sr * dur), endpoint=False)
            f = 145.0 * np.exp(-14.0 * t) + 65.0
            phase = 2 * np.pi * np.cumsum(f) / sr
            return np.tanh(1.3 * np.sin(phase) * np.exp(-10.0 * t))
        else:  # crash
            dur = 0.65
            t = np.linspace(0, dur, int(sr * dur), endpoint=False)
            metal = (
                0.25 * np.sin(2 * np.pi * 330.0 * t)
                + 0.25 * np.sin(2 * np.pi * 485.0 * t)
                + 0.25 * np.sin(2 * np.pi * 675.0 * t)
                + 0.25 * np.sin(2 * np.pi * 890.0 * t)
            )
            noise = np.random.uniform(-1.0, 1.0, len(t))
            noise = noise - 0.75 * np.roll(noise, 1)
            return np.clip((0.35 * metal + 0.65 * noise) * np.exp(-7.0 * t) * 1.3, -1.0, 1.0)
