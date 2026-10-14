"""Low-latency audio engine for AirBeat using Pygame mixer.

Features:
- Low-latency buffer configuration (256 samples, ~5.8ms latency).
- High-fidelity procedural drum synthesis with NumPy (Kick, Snare, Hi-Hat)
  if no external WAV files are present, working 100% out of the box.
- Velocity-to-volume dynamic scaling.
- Flexible mappings:
  1. "finger": Index finger -> Kick, Middle finger -> Snare.
  2. "zones":  Horizontal hand zones (Left -> Kick, Middle -> Snare, Right -> Hi-Hat).
"""

import io
import os
import wave
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pygame

from config import (
    AUDIO_BUFFER_SIZE,
    AUDIO_NUM_CHANNELS,
    AUDIO_SAMPLE_RATE,
    DEFAULT_DRUM_KIT,
    DRUM_KITS,
    DRUM_MAPPING_MODE,
    DRUM_MAX_VELOCITY,
    DRUM_MAX_VOLUME,
    DRUM_MIN_VOLUME,
    DRUM_VELOCITY_THRESHOLD,
    DRUM_ZONE_BOUNDARIES,
    FINGER_DRUM_MAP,
    SOUNDS_DIR,
)


class AudioManager:
    """Manages low-latency audio playback, procedural synthesis, and drum triggers."""

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        buffer_size: int = AUDIO_BUFFER_SIZE,
        num_channels: int = AUDIO_NUM_CHANNELS,
        mapping_mode: str = DRUM_MAPPING_MODE,
        zone_boundaries: Tuple[float, float] = DRUM_ZONE_BOUNDARIES,
    ) -> None:
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.num_channels = num_channels
        self.mapping_mode = mapping_mode
        self.zone_boundaries = zone_boundaries

        self.is_initialized = False
        self.current_kit: str = DEFAULT_DRUM_KIT
        self._kits: Dict[str, Dict[str, pygame.mixer.Sound]] = {}
        self._sounds: Dict[str, pygame.mixer.Sound] = {}
        self._current_backing_track: Optional[str] = None

    def initialize(self) -> bool:
        """Initialize pygame mixer subsystem with low-latency settings.

        Returns:
            bool: True if initialization was successful.
        """
        if self.is_initialized:
            return True

        try:
            # Low latency pre-init: 44.1kHz, 16-bit signed, stereo, 256 sample buffer
            pygame.mixer.pre_init(
                frequency=self.sample_rate,
                size=-16,
                channels=2,
                buffer=self.buffer_size,
            )
            pygame.mixer.init()
            pygame.mixer.set_num_channels(self.num_channels)

            # Load or procedurally synthesize drum samples across all kits
            self._load_or_synthesize_samples()

            self.is_initialized = True
            print(
                f"[AudioManager] Initialized successfully ({self.sample_rate}Hz, "
                f"buffer={self.buffer_size}, {self.num_channels} channels, kit={self.current_kit})."
            )
            return True
        except Exception as e:
            print(f"⚠️ [AudioManager] Initialization error: {e}")
            self.is_initialized = False
            return False

    def _load_or_synthesize_samples(self) -> None:
        """Load external WAV files if present, or synthesize distinct kits with NumPy."""
        drum_types = ["kick", "snare", "hihat", "tom", "crash"]

        for kit in DRUM_KITS:
            self._kits[kit] = {}
            kit_slug = kit.lower().replace(" ", "_").replace("-", "")
            for drum in drum_types:
                kit_wav = os.path.join(SOUNDS_DIR, kit_slug, f"{drum}.wav")
                root_wav = os.path.join(SOUNDS_DIR, f"{drum}.wav")
                sound = None
                for path in (kit_wav, root_wav):
                    if os.path.isfile(path):
                        try:
                            sound = pygame.mixer.Sound(path)
                            break
                        except Exception:
                            pass
                if sound is None:
                    sound = self._synthesize_sample(drum, kit=kit)
                self._kits[kit][drum] = sound

        # Set default active kit sounds for instant access and backward compatibility
        self._sounds = self._kits.get(self.current_kit, self._kits[DRUM_KITS[0]])

    def set_kit(self, kit_name: str) -> str:
        """Switch active drum kit."""
        if kit_name in self._kits:
            self.current_kit = kit_name
            self._sounds = self._kits[kit_name]
            print(f"[AudioManager] Drum kit switched to: {self.current_kit}")
        return self.current_kit

    def cycle_kit(self) -> str:
        """Cycle to next drum kit."""
        if not DRUM_KITS:
            return self.current_kit
        curr_idx = DRUM_KITS.index(self.current_kit) if self.current_kit in DRUM_KITS else 0
        next_idx = (curr_idx + 1) % len(DRUM_KITS)
        return self.set_kit(DRUM_KITS[next_idx])

    def get_sound(self, drum_type: str) -> Optional[pygame.mixer.Sound]:
        """Return active Sound object for a drum instrument."""
        return self._sounds.get(drum_type)

    def _synthesize_sample(self, drum_type: str, kit: str = "808 Trap") -> pygame.mixer.Sound:
        """Generate procedural drum sample using NumPy and convert to pygame Sound."""
        sr = self.sample_rate

        if drum_type == "kick":
            if kit == "Acoustic Rock":
                duration = 0.30
                t = np.linspace(0, duration, int(sr * duration), endpoint=False)
                f_start, f_end, decay_f = 165.0, 52.0, 28.0
                phase = 2 * np.pi * (f_end * t + (f_start - f_end) / decay_f * (1.0 - np.exp(-decay_f * t)))
                body = np.sin(phase) * np.exp(-11.0 * t)
                click = np.sin(2 * np.pi * 950.0 * t) * np.exp(-120.0 * t)
                signal = np.tanh(1.4 * (0.85 * body + 0.15 * click))
            elif kit == "Cyberpunk":
                duration = 0.32
                t = np.linspace(0, duration, int(sr * duration), endpoint=False)
                f_start, f_end, decay_f = 190.0, 38.0, 35.0
                phase = 2 * np.pi * (f_end * t + (f_start - f_end) / decay_f * (1.0 - np.exp(-decay_f * t)))
                sub = np.sin(phase) * np.exp(-8.0 * t)
                square = np.sign(np.sin(phase * 0.5)) * np.exp(-12.0 * t)
                signal = np.clip(1.3 * sub + 0.3 * square, -1.0, 1.0)
            elif kit == "Lo-Fi Vinyl":
                duration = 0.36
                t = np.linspace(0, duration, int(sr * duration), endpoint=False)
                f_start, f_end, decay_f = 95.0, 42.0, 20.0
                phase = 2 * np.pi * (f_end * t + (f_start - f_end) / decay_f * (1.0 - np.exp(-decay_f * t)))
                body = np.sin(phase) * np.exp(-7.5 * t)
                crackle = np.random.uniform(-0.04, 0.04, len(t))
                signal = np.tanh(1.2 * body) + crackle
            else:  # 808 Trap default
                duration = 0.38
                t = np.linspace(0, duration, int(sr * duration), endpoint=False)
                f_start, f_end, decay_f = 140.0, 36.0, 30.0
                phase = 2 * np.pi * (f_end * t + (f_start - f_end) / decay_f * (1.0 - np.exp(-decay_f * t)))
                body = np.sin(phase) * np.exp(-8.5 * t)
                click = np.sin(2 * np.pi * 1200.0 * t) * np.exp(-160.0 * t)
                signal = np.tanh(1.5 * (0.82 * body + 0.18 * click))

        elif drum_type == "snare":
            if kit == "Acoustic Rock":
                duration = 0.30
                t = np.linspace(0, duration, int(sr * duration), endpoint=False)
                body = np.sin(2 * np.pi * 195.0 * t) * np.exp(-18.0 * t)
                noise = np.random.uniform(-1.0, 1.0, len(t))
                noise = np.diff(noise, prepend=0.0)
                signal = np.tanh(1.25 * (0.42 * body + 0.58 * noise * np.exp(-11.0 * t)))
            elif kit == "Cyberpunk":
                duration = 0.22
                t = np.linspace(0, duration, int(sr * duration), endpoint=False)
                body = np.sin(2 * np.pi * 220.0 * t) * np.exp(-25.0 * t)
                noise = np.random.uniform(-1.0, 1.0, len(t))
                comb = noise + 0.5 * np.roll(noise, 32)
                signal = np.clip(1.4 * (0.35 * body + 0.65 * comb * np.exp(-15.0 * t)), -1.0, 1.0)
            elif kit == "Lo-Fi Vinyl":
                duration = 0.28
                t = np.linspace(0, duration, int(sr * duration), endpoint=False)
                body = np.sin(2 * np.pi * 175.0 * t) * np.exp(-16.0 * t)
                noise = np.random.uniform(-0.8, 0.8, len(t))
                noise = 0.6 * noise + 0.4 * np.roll(noise, 2)
                signal = np.tanh(1.1 * (0.45 * body + 0.55 * noise * np.exp(-12.0 * t)))
            else:  # 808 Trap
                duration = 0.24
                t = np.linspace(0, duration, int(sr * duration), endpoint=False)
                body = np.sin(2 * np.pi * 210.0 * t) * np.exp(-24.0 * t)
                noise = np.random.uniform(-1.0, 1.0, len(t))
                noise = np.diff(noise, prepend=0.0)
                signal = np.tanh(1.35 * (0.35 * body + 0.65 * noise * np.exp(-14.0 * t)))

        elif drum_type == "hihat":
            duration = 0.06 if kit == "808 Trap" else (0.12 if kit == "Acoustic Rock" else 0.08)
            t = np.linspace(0, duration, int(sr * duration), endpoint=False)
            noise = np.random.uniform(-1.0, 1.0, len(t))
            noise = noise - 0.85 * np.roll(noise, 1)
            decay = 55.0 if kit == "808 Trap" else (35.0 if kit == "Acoustic Rock" else 42.0)
            signal = np.clip(noise * np.exp(-decay * t) * 1.5, -1.0, 1.0)

        elif drum_type == "tom":
            duration = 0.32
            t = np.linspace(0, duration, int(sr * duration), endpoint=False)
            f_start, f_end, decay_f = (170.0, 70.0, 24.0) if kit == "Cyberpunk" else (145.0, 62.0, 20.0)
            phase = 2 * np.pi * (f_end * t + (f_start - f_end) / decay_f * (1.0 - np.exp(-decay_f * t)))
            body = np.sin(phase) * np.exp(-9.5 * t)
            click = np.sin(2 * np.pi * 360.0 * t) * np.exp(-70.0 * t)
            signal = np.tanh(1.3 * (0.85 * body + 0.15 * click))

        elif drum_type == "crash":
            duration = 0.75 if kit == "Acoustic Rock" else 0.55
            t = np.linspace(0, duration, int(sr * duration), endpoint=False)
            metal = (
                0.25 * np.sin(2 * np.pi * 330.0 * t)
                + 0.25 * np.sin(2 * np.pi * 485.0 * t)
                + 0.25 * np.sin(2 * np.pi * 675.0 * t)
                + 0.25 * np.sin(2 * np.pi * 890.0 * t)
            )
            noise = np.random.uniform(-1.0, 1.0, len(t))
            noise = noise - 0.75 * np.roll(noise, 1)
            decay = 5.5 if kit == "Acoustic Rock" else 7.5
            signal = np.clip((0.35 * metal + 0.65 * noise) * np.exp(-decay * t) * 1.35, -1.0, 1.0)

        else:
            duration = 0.1
            t = np.linspace(0, duration, int(sr * duration), endpoint=False)
            signal = np.sin(2 * np.pi * 440.0 * t) * np.exp(-20.0 * t)

        # Convert float [-1.0, 1.0] to signed 16-bit integer PCM stereo
        i16_samples = (np.clip(signal, -1.0, 1.0) * 32767).astype(np.int16)
        stereo_samples = np.column_stack((i16_samples, i16_samples))

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(stereo_samples.tobytes())
        buf.seek(0)

        return pygame.mixer.Sound(buf)

    def velocity_to_volume(self, velocity: float) -> float:
        """Scale hit velocity dynamically to playback volume.

        Args:
            velocity: Downward strike velocity in norm units/sec.

        Returns:
            Volume float in range [DRUM_MIN_VOLUME, DRUM_MAX_VOLUME].
        """
        v_min = DRUM_VELOCITY_THRESHOLD
        v_max = DRUM_MAX_VELOCITY
        norm_v = np.clip((velocity - v_min) / max(0.001, v_max - v_min), 0.0, 1.0)

        # Gentle power curve for natural acoustic dynamics
        curved_v = float(norm_v ** 0.85)
        volume = DRUM_MIN_VOLUME + (DRUM_MAX_VOLUME - DRUM_MIN_VOLUME) * curved_v
        return float(np.clip(volume, 0.0, 1.0))

    def resolve_drum(
        self,
        finger_name: str,
        norm_x: float,
        mapping_mode: Optional[str] = None,
    ) -> str:
        """Map finger strike or horizontal hand position to target drum sound.

        Args:
            finger_name: "thumb", "index", "middle", "ring", or "pinky".
            norm_x: Normalized horizontal coordinate of the striking fingertip (0=left, 1=right).
            mapping_mode: "finger" or "zones" (defaults to self.mapping_mode).

        Returns:
            Drum sound name ("kick", "snare", "hihat", "tom", or "crash").
        """
        mode = mapping_mode or self.mapping_mode

        if mode == "zones":
            b_left, b_right = self.zone_boundaries
            if norm_x < b_left:
                return "kick"
            elif norm_x < b_right:
                return "snare"
            else:
                return "hihat"

        # 5-finger drum kit mapping (right hand / green hand)
        name_lower = finger_name.lower()
        return FINGER_DRUM_MAP.get(name_lower, "kick")

    def play_drum(
        self,
        drum_name: str,
        volume: float = 1.0,
    ) -> Optional[pygame.mixer.Channel]:
        """Trigger drum sound playback on an available polyphonic channel.

        Args:
            drum_name: "kick", "snare", or "hihat".
            volume: Playback volume between 0.0 and 1.0.

        Returns:
            Channel playing the sound, or None if unavailable.
        """
        if not self.is_initialized:
            return None

        sound = self._sounds.get(drum_name.lower())
        if sound is None:
            return None

        # Play on next available free mixer channel
        channel = sound.play()
        if channel is not None:
            channel.set_volume(max(0.0, min(1.0, volume)))
        return channel

    def trigger_hit(
        self,
        finger_name: str,
        velocity: float,
        norm_x: float,
        mapping_mode: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Process hit event: resolve drum, scale volume, and trigger sound.

        Args:
            finger_name: "index" or "middle".
            velocity: Strike velocity.
            norm_x: Normalized x coordinate of fingertip.
            mapping_mode: Optional override for mapping mode.

        Returns:
            Tuple of (drum_name, volume).
        """
        drum = self.resolve_drum(finger_name, norm_x, mapping_mode)
        volume = self.velocity_to_volume(velocity)
        self.play_drum(drum, volume=volume)
        return drum, volume

    def set_mapping_mode(self, mode: str) -> None:
        """Switch between 'finger' and 'zones' mapping modes."""
        if mode in ("finger", "zones"):
            self.mapping_mode = mode

    def cleanup(self) -> None:
        """Quit audio mixer and release resources."""
        if self.is_initialized:
            try:
                pygame.mixer.quit()
            except Exception:
                pass
            self.is_initialized = False
