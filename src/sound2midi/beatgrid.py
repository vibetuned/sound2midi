"""Beat-grid and tempo-map time mapping shared by the exporter and the MPE engine.

``BeatGrid`` maps absolute seconds to felt-beat positions using the beat grid
tracked from the source audio (``meter.json``, Beat This!). The tempo-map
helpers convert between a MIDI file's ticks and absolute seconds honoring its
``set_tempo`` events.
"""

from __future__ import annotations

import bisect
import statistics
from collections.abc import Callable
from typing import Any

import mido


def tempo_changes(mid: Any) -> list[tuple[int, int]]:
    """The file's ``(absolute_tick, tempo)`` changes, sorted, with a 120 BPM default."""
    changes: list[tuple[int, int]] = []
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.is_meta and msg.type == "set_tempo":
                changes.append((tick, msg.tempo))
    changes.sort(key=lambda c: c[0])
    if not changes or changes[0][0] != 0:
        changes.insert(0, (0, 500000))  # default 120 BPM
    return changes


def tick_to_sec_fn(mid: Any) -> Callable[[int], float]:
    """A tick -> absolute-seconds function honoring the file's tempo map."""
    changes = tempo_changes(mid)

    segments: list[tuple[int, float, int]] = []  # (start_tick, start_sec, tempo)
    for i, (tick, tempo) in enumerate(changes):
        if i == 0:
            segments.append((tick, 0.0, tempo))
        else:
            ptick, psec, ptempo = segments[-1]
            sec = psec + mido.tick2second(tick - ptick, mid.ticks_per_beat, ptempo)
            segments.append((tick, sec, tempo))

    starts = [s[0] for s in segments]

    def tick_to_sec(tick: int) -> float:
        i = max(0, bisect.bisect_right(starts, tick) - 1)
        stick, ssec, tempo = segments[i]
        return ssec + mido.tick2second(tick - stick, mid.ticks_per_beat, tempo)

    return tick_to_sec


def sec_to_tick_fn(mid: Any) -> Callable[[float], int]:
    """The inverse of :func:`tick_to_sec_fn` (the tempo map is monotonic)."""
    tick_to_sec = tick_to_sec_fn(mid)
    changes = tempo_changes(mid)
    segments = [(tick, tick_to_sec(tick), tempo) for tick, tempo in changes]
    start_secs = [s[1] for s in segments]

    def sec_to_tick(sec: float) -> int:
        i = max(0, bisect.bisect_right(start_secs, sec) - 1)
        stick, ssec, tempo = segments[i]
        return stick + round(mido.second2tick(max(0.0, sec - ssec), mid.ticks_per_beat, tempo))

    return sec_to_tick


class BeatGrid:
    """Maps absolute seconds to felt-beat positions, anchored at the first downbeat."""

    def __init__(self, meter: dict) -> None:
        self.beats: list[float] = [float(b) for b in meter["beats"]]
        if len(self.beats) < 2:
            raise ValueError("Beat grid needs at least 2 beats.")
        self.period = statistics.median(
            b - a for a, b in zip(self.beats[:-1], self.beats[1:], strict=False)
        )
        self.compound = bool(meter.get("compound", False))
        self.beat_ql = 1.5 if self.compound else 1.0
        self.felt_per_bar = int(meter.get("felt_beats_per_bar") or meter.get("numerator", 4))
        self.bpm = float(meter.get("bpm") or 60.0 / self.period)

        first_downbeat = meter.get("first_downbeat")
        if first_downbeat is None:
            self.anchor = 0
        else:  # index of the beat closest to the first downbeat
            self.anchor = min(
                range(len(self.beats)), key=lambda i: abs(self.beats[i] - float(first_downbeat))
            )

    def pos(self, t: float) -> float:
        """Felt-beat position of time ``t`` relative to the first downbeat."""
        beats = self.beats
        i = bisect.bisect_right(beats, t) - 1
        if i < 0:
            raw = (t - beats[0]) / self.period
        elif i >= len(beats) - 1:
            raw = (len(beats) - 1) + (t - beats[-1]) / self.period
        else:
            raw = i + (t - beats[i]) / (beats[i + 1] - beats[i])
        return raw - self.anchor
