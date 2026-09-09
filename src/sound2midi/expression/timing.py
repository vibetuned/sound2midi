"""Module G — rubato / expressive timing.

Applied before curve generation so vibrato onsets, swells and phrase arcs
track the displaced timings. Two layers: a global monotonic time-warp shared
by every track (agogics), then small per-voice asynchrony. Event times are
displaced; the tempo map is never rewritten, so DAW import and alignment with
``meter.json`` stay predictable.
"""

from __future__ import annotations

import random
from bisect import bisect_right

from sound2midi.beatgrid import BeatGrid
from sound2midi.expression.context import Artifacts, Note, TrackData

DRIFT_CAP_S = 0.080  # |w(t) - t| ≤ 80 ms · strength
QUANTIZED_MEDIAN_S = 0.010
_WARP_STEP_S = 0.125


def is_quantized(tracks: list[TrackData], grid: BeatGrid | None, ppq: int) -> bool:
    """Median onset-to-grid deviation < 10 ms → the input is score-quantized."""
    deviations: list[float] = []
    for track in tracks:
        for note in track.notes:
            if grid is not None:
                pos = grid.pos(note.start)
                beat_dev = abs(pos - round(pos * 2) / 2)  # nearest 8th on the felt grid
                deviations.append(beat_dev * 60.0 / grid.bpm)
            else:
                tick_pos = note.start_tick / (ppq / 4)  # 16th grid
                # tick deviation in beats -> seconds at 120 BPM default
                deviations.append(abs(tick_pos - round(tick_pos)) * (ppq / 4) / ppq * 0.5)
    if not deviations:
        return True
    deviations.sort()
    return deviations[len(deviations) // 2] < QUANTIZED_MEDIAN_S


def _reference_track(tracks: list[TrackData]) -> TrackData | None:
    """The phrase reference for the global warp: the busiest melodic track."""
    melodic = [t for t in tracks if not t.is_drum and t.notes]
    return max(melodic, key=lambda t: len(t.notes)) if melodic else None


def _melody_track(tracks: list[TrackData]) -> TrackData | None:
    """The track carrying the top voice: highest mean pitch with enough notes."""
    candidates = [t for t in tracks if not t.is_drum and len(t.notes) >= 8]
    if not candidates:
        return None
    return max(candidates, key=lambda t: sum(n.pitch for n in t.notes) / len(t.notes))


def _build_warp(
    tracks: list[TrackData],
    artifacts: Artifacts,
    grid: BeatGrid | None,
    strength: float,
) -> tuple[list[float], list[float]]:
    """The shared displacement curve d(t): ``w(t) = t + d(t)``, |d| capped.

    Built by integrating local tempo deviations (accelerando into phrase peaks,
    ritenuto at phrase ends, agogic downbeat accents, section broadening) on a
    fixed sample grid, then clamping to the drift cap. Displacements change by
    at most a few ms per step, so w stays monotonic by construction.
    """
    end = max((n.perf_end for t in tracks for n in t.notes), default=0.0) + 1.0
    times = [i * _WARP_STEP_S for i in range(int(end / _WARP_STEP_S) + 2)]
    dev = [0.0] * len(times)  # local tempo deviation: +x = faster, -x = slower

    beat_len = 60.0 / grid.bpm if grid else 0.5
    reference = _reference_track(tracks)
    if reference is not None and reference.notes:
        phrases: dict[int, list[Note]] = {}
        for note in reference.notes:
            phrases.setdefault(note.phrase, []).append(note)
        for members in phrases.values():
            start = members[0].start
            last = members[-1].start
            peak = next((n.start for n in members if n.phrase_peak), (start + last) / 2)
            for i, t in enumerate(times):
                if start <= t < peak:  # slight acceleration into the phrase peak
                    dev[i] += 0.02 * strength
                if last - 1.5 * beat_len <= t <= last + 0.5 * beat_len:  # ritenuto
                    dev[i] -= 0.04 * strength

    if grid is not None:  # bar-level agogic accent: downbeats lengthened ~1%
        for i, t in enumerate(times):
            pos = grid.pos(t)
            if pos >= 0 and (pos % grid.felt_per_bar) < 1.0:
                dev[i] -= 0.01 * strength

    displacement = [0.0]
    for i in range(1, len(times)):
        d = displacement[-1] - dev[i - 1] * _WARP_STEP_S
        cap = DRIFT_CAP_S * strength
        displacement.append(max(-cap, min(cap, d)))

    if artifacts.sections:  # broadening into a section change (up to 60 ms)
        cap = DRIFT_CAP_S * strength
        for _, _, seg_end in artifacts.sections[:-1]:
            for i, t in enumerate(times):
                if seg_end - beat_len <= t <= seg_end:
                    ramp = (t - (seg_end - beat_len)) / beat_len
                    displacement[i] = max(-cap, min(cap, displacement[i] + 0.060 * strength * ramp))
    return times, displacement


def apply_rubato(
    tracks: list[TrackData],
    artifacts: Artifacts,
    rng: random.Random,
    *,
    strength: float = 0.5,
) -> None:
    """Displace every note's performance times (Module G). ``strength`` 0 is a no-op."""
    if strength <= 0.0:
        return
    grid = artifacts.grid()
    times, displacement = _build_warp(tracks, artifacts, grid, strength)

    def warp(t: float) -> float:
        i = min(len(times) - 2, max(0, bisect_right(times, t) - 1))
        frac = (t - times[i]) / _WARP_STEP_S
        return t + displacement[i] + frac * (displacement[i + 1] - displacement[i])

    for track in tracks:  # layer 1: shared monotonic warp
        for note in track.notes:
            note.perf_start = warp(note.start)
            note.perf_end = warp(note.end)

    melody = _melody_track(tracks)
    for track in tracks:  # layer 2: per-voice asynchrony
        for note in track.notes:
            if track is melody and note.metric_weight >= 0.75:
                note.perf_start -= (0.010 + 0.015 * rng.random()) * strength
            note.perf_start += rng.gauss(0.0, 0.006 * strength)
            duration = note.perf_end - note.perf_start
            if note.phrase_final:
                duration *= 1.05
            else:
                duration *= 0.97
            note.perf_end = note.perf_start + max(0.02, duration)
