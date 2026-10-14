"""Rhythm Hero: Interactive rhythm-action game mode for AirBeat.

Spawns falling neon rhythm gems synchronized with the active backing track and BPM.
Detects fingertip air drum strikes at the target bar, evaluates timing accuracy
(PERFECT / GOOD / MISS), and manages scoring, streak multipliers, and combo animations.
"""

from dataclasses import dataclass
import random
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from config import (
    COLOR_ACCENT,
    COLOR_HIT_FLASH,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SUCCESS,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    RHYTHM_GOOD_WINDOW_SEC,
    RHYTHM_PERFECT_WINDOW_SEC,
)

LANE_DRUMS = ["kick", "snare", "hihat", "tom", "crash"]
LANE_COLORS = [
    (255, 128, 0),    # Cyan/Blue - Kick (Thumb)
    (80, 240, 120),   # Emerald - Snare (Index)
    (0, 240, 255),    # Yellow - Hi-Hat (Middle)
    (0, 160, 255),    # Amber/Orange - Tom (Ring)
    (220, 40, 255),   # Magenta - Crash (Pinky)
]


@dataclass
class RhythmNote:
    """Represents a falling rhythm target gem."""

    note_id: int
    drum: str
    lane_idx: int
    target_time: float  # Absolute timestamp when gem hits target line
    hit: bool = False
    missed: bool = False


@dataclass
class FloatingScore:
    """Animated floating text feedback for hits."""

    text: str
    x: int
    y: int
    color: Tuple[int, int, int]
    start_time: float
    duration: float = 0.65


class RhythmHeroGame:
    """Rhythm action game engine with falling lanes, combo scoring, and beat synchronization."""

    def __init__(self) -> None:
        self.is_active: bool = False
        self.score: int = 0
        self.combo: int = 0
        self.max_combo: int = 0
        self.multiplier: int = 1
        self.perfect_count: int = 0
        self.good_count: int = 0
        self.miss_count: int = 0

        self.notes: List[RhythmNote] = []
        self._note_counter: int = 0
        self._last_spawn_beat: float = -1.0
        self.floating_scores: List[FloatingScore] = []

        # Target highway layout parameters (centered horizontally)
        self.target_y: int = 620
        self.spawn_y: int = 120
        self.note_travel_sec: float = 1.6  # Time from top to hit line

    def toggle(self) -> bool:
        """Toggle game mode on / off."""
        self.is_active = not self.is_active
        if self.is_active:
            self.reset()
            print("[RhythmHero] 🎮 GAME MODE ACTIVATED! Get ready to rock!")
        else:
            print(f"[RhythmHero] Game over! Final Score: {self.score} (Max Combo: {self.max_combo})")
        return self.is_active

    def reset(self) -> None:
        """Reset game statistics and clear tracks."""
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.multiplier = 1
        self.perfect_count = 0
        self.good_count = 0
        self.miss_count = 0
        self.notes.clear()
        self.floating_scores.clear()
        self._last_spawn_beat = -1.0

    def spawn_note(self, drum: str, target_time: float) -> Optional[RhythmNote]:
        """Manually spawn a rhythm note gem for a specific drum."""
        if drum not in LANE_DRUMS:
            return None
        lane_idx = LANE_DRUMS.index(drum)
        self._note_counter += 1
        note = RhythmNote(
            note_id=self._note_counter,
            drum=drum,
            lane_idx=lane_idx,
            target_time=target_time,
        )
        self.notes.append(note)
        return note

    def update_clock(
        self,
        current_beat: float = 0.0,
        bpm: int = 120,
        current_time: Optional[float] = None,
    ) -> None:
        """Spawn new notes in sync with beat and check for missed notes."""
        if not self.is_active:
            return

        if current_time is None:
            current_time = time.time()

        # Multiplier logic based on combo
        if self.combo >= 30:
            self.multiplier = 4
        elif self.combo >= 15:
            self.multiplier = 3
        elif self.combo >= 5:
            self.multiplier = 2
        else:
            self.multiplier = 1

        # Spawn notes on beat subdivisions
        sec_per_beat = 60.0 / max(60, bpm)
        beat_int = int(current_beat * 2)  # Half-beat resolution (8th notes)

        if beat_int > self._last_spawn_beat:
            self._last_spawn_beat = beat_int
            self._spawn_procedural_notes(current_beat, current_time, sec_per_beat)

        # Check for missed notes
        for note in self.notes:
            if not note.hit and not note.missed:
                if current_time - note.target_time > RHYTHM_GOOD_WINDOW_SEC:
                    note.missed = True
                    self.combo = 0
                    self.miss_count += 1
                    lane_x = self._get_lane_x(note.lane_idx, 1280)
                    self._add_floating_score("MISS", lane_x, self.target_y, (60, 60, 240), current_time)

        # Cleanup old notes past target line
        self.notes = [n for n in self.notes if current_time - n.target_time < 0.6]

        # Cleanup floating scores
        self.floating_scores = [
            fs for fs in self.floating_scores if current_time - fs.start_time < fs.duration
        ]

    def _spawn_procedural_notes(self, beat: float, current_time: float, sec_per_beat: float) -> None:
        """Generate musical rhythm patterns (e.g. Kick on 1 & 3, Snare on 2 & 4, Hi-Hat 8ths)."""
        beat_mod4 = (beat % 4.0)
        target_time = current_time + self.note_travel_sec

        lanes_to_spawn: List[int] = []

        # Standard rock/groove foundation
        if abs(beat_mod4 - 0.0) < 0.3 or abs(beat_mod4 - 2.5) < 0.3:
            lanes_to_spawn.append(0)  # Kick
        if abs(beat_mod4 - 1.0) < 0.3 or abs(beat_mod4 - 3.0) < 0.3:
            lanes_to_spawn.append(1)  # Snare
        if random.random() < 0.70:
            lanes_to_spawn.append(2)  # Hi-Hat
        if abs(beat_mod4 - 3.5) < 0.2 and random.random() < 0.50:
            lanes_to_spawn.append(3)  # Tom fill
        if abs(beat % 16.0) < 0.2:
            lanes_to_spawn.append(4)  # Crash on bar 1

        for lane_idx in lanes_to_spawn:
            self._note_counter += 1
            note = RhythmNote(
                note_id=self._note_counter,
                drum=LANE_DRUMS[lane_idx],
                lane_idx=lane_idx,
                target_time=target_time,
            )
            self.notes.append(note)

    def check_hit(
        self,
        drum_name: str,
        hit_time: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> Optional[str]:
        """Check if incoming drum hit scores on active falling notes."""
        if not self.is_active:
            return None

        if hit_time is None:
            hit_time = timestamp if timestamp is not None else time.time()

        # Find closest unhit note matching drum
        best_note: Optional[RhythmNote] = None
        min_diff = 999.0

        for note in self.notes:
            if note.hit or note.missed or note.drum != drum_name:
                continue
            diff = abs(hit_time - note.target_time)
            if diff < min_diff:
                min_diff = diff
                best_note = note

        if best_note is None or min_diff > RHYTHM_GOOD_WINDOW_SEC:
            return None

        # Evaluate timing score
        lane_x = self._get_lane_x(best_note.lane_idx, 1280)
        if min_diff <= RHYTHM_PERFECT_WINDOW_SEC:
            best_note.hit = True
            pts = 100 * self.multiplier
            self.score += pts
            self.combo += 1
            self.max_combo = max(self.max_combo, self.combo)
            self.perfect_count += 1
            self._add_floating_score("PERFECT! +100", lane_x, self.target_y - 20, (0, 255, 200), hit_time)
            return "PERFECT"
        else:
            best_note.hit = True
            pts = 50 * self.multiplier
            self.score += pts
            self.combo += 1
            self.max_combo = max(self.max_combo, self.combo)
            self.good_count += 1
            self._add_floating_score("GOOD! +50", lane_x, self.target_y - 20, (0, 215, 255), hit_time)
            return "GOOD"

    def _get_lane_x(self, lane_idx: int, frame_w: int) -> int:
        """Calculate horizontal screen position for lane."""
        start_x = (frame_w // 2) - 200
        lane_gap = 100
        return start_x + lane_idx * lane_gap

    def _add_floating_score(
        self, text: str, x: int, y: int, color: Tuple[int, int, int], timestamp: float
    ) -> None:
        self.floating_scores.append(FloatingScore(text=text, x=x, y=y, color=color, start_time=timestamp))

    def render(self, frame: np.ndarray, current_time: float) -> np.ndarray:
        """Render the neon rhythm highway, falling target gems, hit line, and score HUD."""
        if not self.is_active:
            return frame

        h, w = frame.shape[:2]
        self.target_y = h - 100
        self.spawn_y = 110

        # Draw translucent highway backdrop
        start_x = (w // 2) - 240
        highway_w = 480
        highway_sub = frame[self.spawn_y : self.target_y + 40, start_x : start_x + highway_w]
        overlay = highway_sub.copy()
        overlay[:] = (15, 15, 22)
        cv2.addWeighted(overlay, 0.70, highway_sub, 0.30, 0, highway_sub)

        # Draw vertical lanes & headers
        for i, drum in enumerate(LANE_DRUMS):
            lx = self._get_lane_x(i, w)
            color = LANE_COLORS[i]
            # Lane guideline
            cv2.line(frame, (lx, self.spawn_y), (lx, self.target_y + 25), (45, 45, 60), 1, cv2.LINE_AA)
            # Lane drum tag at top
            tag = drum[:3].upper()
            cv2.putText(
                frame,
                tag,
                (lx - 12, self.spawn_y - 8),
                cv2.FONT_HERSHEY_DUPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )

        # Draw target hit line with glowing bar
        cv2.line(frame, (start_x, self.target_y), (start_x + highway_w, self.target_y), (0, 255, 255), 3, cv2.LINE_AA)
        for i in range(len(LANE_DRUMS)):
            lx = self._get_lane_x(i, w)
            cv2.circle(frame, (lx, self.target_y), 14, LANE_COLORS[i], 2, cv2.LINE_AA)
            cv2.circle(frame, (lx, self.target_y), 4, (255, 255, 255), -1, cv2.LINE_AA)

        # Draw falling notes
        for note in self.notes:
            if note.hit:
                continue
            time_to_target = note.target_time - current_time
            # Progress 0.0 at spawn_y, 1.0 at target_y
            progress = 1.0 - (time_to_target / self.note_travel_sec)
            if progress < 0.0 or progress > 1.2:
                continue

            ny = int(self.spawn_y + progress * (self.target_y - self.spawn_y))
            nx = self._get_lane_x(note.lane_idx, w)
            color = LANE_COLORS[note.lane_idx]

            # Draw glowing gem orb
            cv2.circle(frame, (nx, ny), 12, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (nx, ny), 13, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, (nx, ny), 4, (255, 255, 255), -1, cv2.LINE_AA)

        # Draw floating score animations
        for fs in self.floating_scores:
            elapsed = current_time - fs.start_time
            prog = min(1.0, elapsed / fs.duration)
            alpha = 1.0 - prog
            float_y = int(fs.y - 45 * prog)
            c = (int(fs.color[0] * alpha), int(fs.color[1] * alpha), int(fs.color[2] * alpha))
            cv2.putText(
                frame,
                fs.text,
                (fs.x - 30, float_y),
                cv2.FONT_HERSHEY_DUPLEX,
                0.55,
                c,
                1,
                cv2.LINE_AA,
            )

        # Draw Game Mode Header Banner (Score, Combo, Multiplier)
        badge_y = self.spawn_y - 35
        score_text = f"SCORE: {self.score:,}  |  COMBO: {self.combo}x  |  MULT: {self.multiplier}x"
        (tw, _), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        bx = (w - tw) // 2
        cv2.rectangle(frame, (bx - 12, badge_y - 20), (bx + tw + 12, badge_y + 8), (25, 25, 35), -1)
        cv2.rectangle(frame, (bx - 12, badge_y - 20), (bx + tw + 12, badge_y + 8), COLOR_SECONDARY, 1)
        cv2.putText(
            frame,
            score_text,
            (bx, badge_y),
            cv2.FONT_HERSHEY_DUPLEX,
            0.55,
            COLOR_HIT_FLASH if self.combo > 10 else COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )

        return frame
