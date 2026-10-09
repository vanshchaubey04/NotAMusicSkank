"""Modular Music Engine and Procedural Backing Track Player for AirBeat.

Features:
- State management for Key (C through B) and Scale (Major, Minor, Pentatonic, Dorian).
- Programmatic 4-bar backing track generation (Chord Progression + Bass + Pad)
  synthesized with NumPy in ~20ms, dynamically adapting to Key, Scale, and BPM.
- 3 distinct musical styles:
  1. "Lo-Fi Chill": Mellow jazzy 7th chords, sub-bass, lush stereo detuned pad, electric piano.
  2. "Synthwave 80s": Anthemic progression, driving 8th-note synth bass, soaring analog pad.
  3. "Funk Groove": Syncopated funk bassline, snappy modal chords, warm Rhodes pad.
- Fully modular architecture:
  - Extensible BackingTrackSource base class.
  - Generates MidiEvent objects ready for future MIDI hardware / DAW streaming.
  - Ready for future audio loop files (.wav / .ogg).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import io
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import wave

import numpy as np
import pygame

from config import (
    AUDIO_SAMPLE_RATE,
    BACKING_STYLES,
    BACKING_VOLUME,
    DEFAULT_BACKING_STYLE,
    DEFAULT_BPM,
    DEFAULT_KEY,
    DEFAULT_SCALE,
    MUSICAL_KEYS,
    SCALES,
)


NOTE_SEMITONES: Dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11,
}


def midi_to_freq(midi_pitch: float) -> float:
    """Convert MIDI note number to frequency in Hertz."""
    return 440.0 * (2.0 ** ((midi_pitch - 69.0) / 12.0))


@dataclass
class MidiEvent:
    """Represents a MIDI event for future external DAW or synthesizer output."""

    bar: int
    beat: float
    channel: int  # 0=keys, 1=bass, 2=pad
    note: int  # MIDI pitch number (0-127)
    velocity: int  # MIDI velocity (0-127)
    duration: float  # Duration in beats


class BackingTrackSource(ABC):
    """Abstract base class for backing track audio and MIDI sources."""

    @abstractmethod
    def generate_audio(
        self,
        key: str,
        scale: str,
        style: str,
        bpm: int,
        sr: int = 44100,
    ) -> np.ndarray:
        """Generate stereo audio array (N, 2) in float32 [-1.0, 1.0]."""
        pass

    @abstractmethod
    def get_midi_events(
        self,
        key: str,
        scale: str,
        style: str,
        bpm: int,
    ) -> List[MidiEvent]:
        """Extract MIDI events for external synthesis or DAW recording."""
        pass


class ProceduralTrackSynthesizer(BackingTrackSource):
    """Generates procedural Pad, Bass, and Chords using NumPy synthesis."""

    def __init__(self) -> None:
        pass

    def _get_scale_pitches(self, key: str, scale: str, octave: int = 4) -> List[int]:
        """Return list of MIDI pitches for the given key and scale in the specified octave."""
        root_semi = NOTE_SEMITONES.get(key, 0)
        base_midi = 12 * (octave + 1) + root_semi
        intervals = SCALES.get(scale, SCALES["Major"])
        return [base_midi + i for i in intervals]

    def _get_progression_chords(
        self, key: str, scale: str, style: str
    ) -> List[Tuple[int, List[int], List[int]]]:
        """Return 4-bar progression: list of (bass_midi, pad_midis, chord_midis)."""
        scale_pitches = self._get_scale_pitches(key, scale, octave=4)
        bass_pitches = self._get_scale_pitches(key, scale, octave=2)

        n = len(scale_pitches)

        def make_chord(deg: int, num_notes: int = 4) -> List[int]:
            chord = []
            for k in range(num_notes):
                idx = (deg + k * 2) % n
                oct_shift = (deg + k * 2) // n
                chord.append(scale_pitches[idx] + oct_shift * 12)
            return chord

        # Progression degrees for 4 bars based on style and scale
        if style == "Lo-Fi Chill":
            # Smooth jazzy 7th chords
            if scale == "Major":
                degrees = [0, 5, 3, 4]  # Imaj7 - vi7 - IVmaj7 - V7
            elif scale == "Dorian":
                degrees = [0, 3, 0, 3]  # i7 - IV7 - i7 - IV7 (classic modal vamp)
            else:
                degrees = [0, 5, 2, 4]  # i7 - VI7 - III7 - v7

        elif style == "Synthwave 80s":
            # Driving anthemic chords
            if scale == "Major":
                degrees = [0, 4, 5, 3]  # I - V - vi - IV
            elif scale == "Dorian":
                degrees = [0, 6, 5, 6]  # i - VII - VI - VII
            else:
                degrees = [0, 5, 2, 6]  # i - VI - III - VII

        else:  # "Funk Groove"
            # Soul/Funk vamps
            if scale == "Dorian":
                degrees = [0, 3, 0, 3]  # i7 - IV9 - i7 - IV9
            elif scale == "Major":
                degrees = [0, 3, 0, 4]  # I7 - IV7 - I7 - V7
            else:
                degrees = [0, 3, 0, 4]  # i7 - iv7 - i7 - v7

        bars = []
        for d in degrees:
            bass_note = bass_pitches[d % len(bass_pitches)]
            pad_notes = make_chord(d, num_notes=4)
            keys_notes = make_chord(d, num_notes=3)
            bars.append((bass_note, pad_notes, keys_notes))

        return bars

    def generate_audio(
        self,
        key: str,
        scale: str,
        style: str,
        bpm: int,
        sr: int = AUDIO_SAMPLE_RATE,
    ) -> np.ndarray:
        """Synthesize 4-bar backing track audio with Pad, Bass, and Chords."""
        if style == "None":
            return np.zeros((sr, 2), dtype=np.float32)

        num_bars = 4
        beats_per_bar = 4
        total_beats = num_bars * beats_per_bar
        beat_dur = 60.0 / max(40.0, float(bpm))
        total_dur = total_beats * beat_dur
        total_samples = int(sr * total_dur)

        t = np.linspace(0, total_dur, total_samples, endpoint=False)
        bar_samples = total_samples // num_bars

        audio_left = np.zeros(total_samples, dtype=np.float32)
        audio_right = np.zeros(total_samples, dtype=np.float32)

        chords = self._get_progression_chords(key, scale, style)

        for bar_idx, (bass_midi, pad_midis, keys_midis) in enumerate(chords):
            idx_start = bar_idx * bar_samples
            idx_end = min(total_samples, (bar_idx + 1) * bar_samples)
            bar_len = idx_end - idx_start
            t_bar = t[idx_start:idx_end] - t[idx_start]
            bar_dur = beat_dur * beats_per_bar

            # -----------------------------------------------------------------
            # 1. Warm Stereo Pad (Detuned chorused oscillators)
            # -----------------------------------------------------------------
            pad_env = np.sin(np.pi * (t_bar / bar_dur)) ** 0.4
            for note in pad_midis:
                f = midi_to_freq(note)
                # Left oscillator + chorus detune
                s_l = np.sin(2 * np.pi * f * t_bar) + 0.45 * np.sin(
                    2 * np.pi * (f * 1.003) * t_bar
                )
                # Right oscillator + chorus detune
                s_r = np.sin(2 * np.pi * (f * 0.997) * t_bar) + 0.45 * np.sin(
                    2 * np.pi * f * t_bar
                )
                audio_left[idx_start:idx_end] += (s_l * pad_env * 0.12).astype(np.float32)
                audio_right[idx_start:idx_end] += (s_r * pad_env * 0.12).astype(np.float32)

            # -----------------------------------------------------------------
            # 2. Bassline (Sub-bass + 2nd harmonic)
            # -----------------------------------------------------------------
            f_bass = midi_to_freq(bass_midi)

            if style == "Synthwave 80s":
                # Driving 8th-note pulsing synth bass (8 pulses per bar)
                num_pulses = 8
                pulse_dur = bar_dur / num_pulses
                pulse_samples = int(sr * pulse_dur)
                for p in range(num_pulses):
                    p_start = p * pulse_samples
                    p_end = min(bar_len, (p + 1) * pulse_samples)
                    if p_start >= bar_len:
                        break
                    t_pulse = t_bar[p_start:p_end] - (p * pulse_dur)
                    env_pulse = np.exp(-7.0 * (t_pulse / pulse_dur))
                    # Punchy saw-like synth bass
                    b_sig = (
                        np.sin(2 * np.pi * f_bass * t_pulse)
                        + 0.5 * np.sin(4 * np.pi * f_bass * t_pulse)
                        + 0.25 * np.sin(6 * np.pi * f_bass * t_pulse)
                    ) * env_pulse * 0.32
                    audio_left[idx_start + p_start : idx_start + p_end] += b_sig
                    audio_right[idx_start + p_start : idx_start + p_end] += b_sig

            elif style == "Funk Groove":
                # Syncopated groovy bassline (root on beat 1, octave on beat 2.5, 5th on beat 3.5)
                f_oct = f_bass * 2.0
                f_fifth = f_bass * 1.498
                patterns = [
                    (0.0, beat_dur * 1.2, f_bass, 0.35),
                    (beat_dur * 1.5, beat_dur * 0.8, f_oct, 0.28),
                    (beat_dur * 2.5, beat_dur * 0.8, f_fifth, 0.30),
                    (beat_dur * 3.25, beat_dur * 0.6, f_oct, 0.26),
                ]
                for note_start, note_len, note_f, note_amp in patterns:
                    n_s_idx = int(note_start * sr)
                    n_e_idx = min(bar_len, int((note_start + note_len) * sr))
                    if n_s_idx >= bar_len:
                        continue
                    t_n = t_bar[n_s_idx:n_e_idx] - note_start
                    env_n = np.exp(-5.0 * (t_n / note_len))
                    b_sig = (
                        np.sin(2 * np.pi * note_f * t_n)
                        + 0.4 * np.sin(4 * np.pi * note_f * t_n)
                    ) * env_n * note_amp
                    audio_left[idx_start + n_s_idx : idx_start + n_e_idx] += b_sig
                    audio_right[idx_start + n_s_idx : idx_start + n_e_idx] += b_sig

            else:  # "Lo-Fi Chill"
                # Warm sustained sub-bass
                env_lofi = np.exp(-1.5 * (t_bar / bar_dur))
                b_sig = (
                    np.sin(2 * np.pi * f_bass * t_bar)
                    + 0.25 * np.sin(4 * np.pi * f_bass * t_bar)
                ) * env_lofi * 0.38
                audio_left[idx_start:idx_end] += b_sig
                audio_right[idx_start:idx_end] += b_sig

            # -----------------------------------------------------------------
            # 3. Chord Keys / Electric Piano
            # -----------------------------------------------------------------
            if style == "Lo-Fi Chill":
                # Rhodes-like chords on beat 1 and beat 3
                hit_beats = [0.0, 2.0]
                hit_decay = 3.5
            elif style == "Synthwave 80s":
                # Stabs on beats 1, 2, 3, 4
                hit_beats = [0.0, 1.0, 2.0, 3.0]
                hit_decay = 7.0
            else:  # "Funk Groove"
                # Funky syncopated comping on offbeats
                hit_beats = [0.0, 1.5, 2.5, 3.25]
                hit_decay = 9.0

            for hb in hit_beats:
                hit_start = hb * beat_dur
                hit_s_idx = int(hit_start * sr)
                hit_dur = beat_dur * 1.5
                hit_e_idx = min(bar_len, int((hit_start + hit_dur) * sr))
                if hit_s_idx >= bar_len:
                    continue
                t_chord = t_bar[hit_s_idx:hit_e_idx] - hit_start
                env_chord = np.exp(-hit_decay * (t_chord / hit_dur))

                for note in keys_midis:
                    f_k = midi_to_freq(note)
                    # Electric piano harmonic profile
                    chord_tone = (
                        np.sin(2 * np.pi * f_k * t_chord)
                        + 0.35 * np.sin(4 * np.pi * f_k * t_chord)
                        + 0.15 * np.sin(6 * np.pi * f_k * t_chord)
                    ) * env_chord * 0.10
                    audio_left[idx_start + hit_s_idx : idx_start + hit_e_idx] += chord_tone
                    audio_right[idx_start + hit_s_idx : idx_start + hit_e_idx] += chord_tone

        # Mix and analog soft saturation
        stereo_mix = np.column_stack((audio_left, audio_right))
        stereo_mix = np.tanh(1.25 * stereo_mix)
        return stereo_mix.astype(np.float32)

    def get_midi_events(
        self,
        key: str,
        scale: str,
        style: str,
        bpm: int,
    ) -> List[MidiEvent]:
        """Extract discrete MIDI events for external synthesis or DAW exporting."""
        events: List[MidiEvent] = []
        chords = self._get_progression_chords(key, scale, style)

        for bar_idx, (bass_midi, pad_midis, keys_midis) in enumerate(chords):
            # Bass event
            events.append(
                MidiEvent(bar=bar_idx, beat=0.0, channel=1, note=bass_midi, velocity=100, duration=4.0)
            )
            # Pad events
            for p_note in pad_midis:
                events.append(
                    MidiEvent(bar=bar_idx, beat=0.0, channel=2, note=p_note, velocity=80, duration=4.0)
                )
            # Keys events
            if style == "Lo-Fi Chill":
                beats = [0.0, 2.0]
            elif style == "Synthwave 80s":
                beats = [0.0, 1.0, 2.0, 3.0]
            else:
                beats = [0.0, 1.5, 2.5, 3.25]

            for b in beats:
                for k_note in keys_midis:
                    events.append(
                        MidiEvent(bar=bar_idx, beat=b, channel=0, note=k_note, velocity=95, duration=1.0)
                    )

        return events


class MusicEngine:
    """Central musical state manager and backing track playback controller."""

    def __init__(
        self,
        key: str = DEFAULT_KEY,
        scale: str = DEFAULT_SCALE,
        style: str = DEFAULT_BACKING_STYLE,
        bpm: int = DEFAULT_BPM,
        volume: float = BACKING_VOLUME,
        synthesizer: Optional[BackingTrackSource] = None,
    ) -> None:
        self.key: str = key if key in MUSICAL_KEYS else DEFAULT_KEY
        self.scale: str = scale if scale in SCALES else DEFAULT_SCALE
        self.style: str = style if style in BACKING_STYLES else DEFAULT_BACKING_STYLE
        self.bpm: int = bpm
        self.volume: float = volume
        self.is_playing: bool = False

        self.synth = synthesizer or ProceduralTrackSynthesizer()

        # Dedicated playback channel
        self._music_channel: Optional[pygame.mixer.Channel] = None
        self._current_sound: Optional[pygame.mixer.Sound] = None
        self._init_playback_channel()

    def _init_playback_channel(self) -> None:
        """Reserve a dedicated channel in pygame.mixer for backing track looping."""
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            # Channel 0 is designated for continuous backing track playback
            self._music_channel = pygame.mixer.Channel(0)
            self._music_channel.set_volume(self.volume)
        except Exception as e:
            print(f"⚠️ [MusicEngine] Playback channel initialization error: {e}")

    @staticmethod
    def _array_to_sound(samples_stereo: np.ndarray, sr: int = AUDIO_SAMPLE_RATE) -> pygame.mixer.Sound:
        """Convert stereo float32 array to pygame.mixer.Sound via in-memory WAV."""
        i16 = (np.clip(samples_stereo, -1.0, 1.0) * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(i16.tobytes())
        buf.seek(0)
        return pygame.mixer.Sound(buf)

    def set_key(self, new_key: str) -> None:
        """Update musical root key (C through B)."""
        if new_key in MUSICAL_KEYS and new_key != self.key:
            self.key = new_key
            print(f"[MusicEngine] Key set to: {self.key}")
            self._on_parameters_changed()

    def set_scale(self, new_scale: str) -> None:
        """Update musical scale (Major, Minor, Pentatonic, Dorian)."""
        if new_scale in SCALES and new_scale != self.scale:
            self.scale = new_scale
            print(f"[MusicEngine] Scale set to: {self.scale}")
            self._on_parameters_changed()

    def set_style(self, new_style: str) -> None:
        """Update backing track style (Lo-Fi Chill, Synthwave 80s, Funk Groove, None)."""
        if new_style in BACKING_STYLES and new_style != self.style:
            self.style = new_style
            print(f"[MusicEngine] Style set to: {self.style}")
            self._on_parameters_changed()

    def set_bpm(self, new_bpm: int) -> None:
        """Update BPM and re-render if playing."""
        clamped = max(40, min(240, int(new_bpm)))
        if abs(clamped - self.bpm) >= 3:
            self.bpm = clamped
            if self.is_playing:
                self._on_parameters_changed()

    def cycle_key(self, direction: int = 1) -> str:
        """Cycle to the next/previous musical key."""
        idx = MUSICAL_KEYS.index(self.key) if self.key in MUSICAL_KEYS else 0
        new_key = MUSICAL_KEYS[(idx + direction) % len(MUSICAL_KEYS)]
        self.set_key(new_key)
        return self.key

    def cycle_scale(self, direction: int = 1) -> str:
        """Cycle to the next/previous musical scale."""
        scale_names = list(SCALES.keys())
        idx = scale_names.index(self.scale) if self.scale in scale_names else 0
        new_scale = scale_names[(idx + direction) % len(scale_names)]
        self.set_scale(new_scale)
        return self.scale

    def cycle_style(self, direction: int = 1) -> str:
        """Cycle to the next/previous backing track style."""
        idx = BACKING_STYLES.index(self.style) if self.style in BACKING_STYLES else 0
        new_style = BACKING_STYLES[(idx + direction) % len(BACKING_STYLES)]
        self.set_style(new_style)
        return self.style

    def set_volume(self, vol: float) -> None:
        """Update backing track volume [0.0, 1.0]."""
        self.volume = max(0.0, min(1.0, float(vol)))
        if self._music_channel:
            self._music_channel.set_volume(self.volume)

    def play(self) -> None:
        """Start looping the current backing track."""
        if self.style == "None":
            self.stop()
            return

        t0 = time.perf_counter()
        audio_array = self.synth.generate_audio(
            key=self.key, scale=self.scale, style=self.style, bpm=self.bpm
        )
        self._current_sound = self._array_to_sound(audio_array)

        if self._music_channel:
            self._music_channel.set_volume(self.volume)
            self._music_channel.play(self._current_sound, loops=-1)
            self.is_playing = True
            dur_ms = (time.perf_counter() - t0) * 1000.0
            print(
                f"[MusicEngine] Playing '{self.style}' in {self.key} {self.scale} @ {self.bpm} BPM "
                f"(generated in {dur_ms:.1f}ms)."
            )

    def stop(self) -> None:
        """Stop backing track playback."""
        if pygame.mixer.get_init() and self._music_channel:
            try:
                self._music_channel.stop()
            except Exception:
                pass
        self.is_playing = False
        print("[MusicEngine] Backing track stopped.")

    def toggle_playback(self) -> bool:
        """Toggle backing track play/pause."""
        if self.is_playing:
            self.stop()
        else:
            self.play()
        return self.is_playing

    def _on_parameters_changed(self) -> None:
        """Re-render and restart backing loop if currently playing."""
        if self.is_playing:
            self.play()

    def get_state(self) -> Dict[str, Any]:
        """Return full musical state dictionary for GUI and HUD displays."""
        return {
            "key": self.key,
            "scale": self.scale,
            "style": self.style,
            "bpm": self.bpm,
            "is_playing": self.is_playing,
            "volume": self.volume,
        }

    def cleanup(self) -> None:
        """Stop playback and clean up mixer resources."""
        self.stop()
        self._current_sound = None
