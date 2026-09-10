"""``--airwave``: a synthesized Airwave gesture performance for midi-sink.

Maps a MIDI file onto the Airwave controller's CC layout (midi-sink's
``sumi_ctl_t`` table). The vortex is visually powerful even at low values,
so it carries the ATTACKS: brief melody-driven flares hard-capped at 47,
its centre tracking the tune. The swirl is mild on screen, so it is the
WIND: a slow energy-following layer free to use the full range, drifting in
multi-bar sways. The tracks passed in come from the ``--airwave`` source
MIDI — its attacks flare the vortex, its energy drives the swirl and the
ripples, its accents the pinches — while the beat grid and section boundaries
come from the song's artifacts, so the gestures stay in sync with the
performance they are merged into. Everything rides channel 1 (index 0, the
MPE master), away from the member note channels.
"""

from __future__ import annotations

import math
import random

import mido

from sound2midi.beatgrid import BeatGrid
from sound2midi.expression.context import Artifacts, TrackData
from sound2midi.expression.mpe_encoder import TimedMessage
from sound2midi.expression.timing import _melody_track

# The Airwave CC map (midi-sink `sumi_ctl_t`).
CC_PINCH_SADDLE = 20  # Grasp, left: folds at the vortex centre
CC_PINCH_CROSS = 21  # Grasp, right: crossed tines at the swirl centre
CC_VORTEX_Y = 22  # Slide, left (reversed)
CC_SWIRL_Y = 23  # Slide, right (reversed)
CC_VORTEX_X = 24  # Glide, left
CC_SWIRL_X = 25  # Glide, right
CC_VORTEX_STRENGTH = 26  # Raise, left: wind over the water
CC_SWIRL_STRENGTH = 27  # Raise, right: the Lamb-Oseen stir
CC_RIPPLE_FREQ = 28  # Tilt, left: the waves' wavelength
CC_RIPPLE_AMP = 29  # Tilt, right: their amount

AIRWAVE_CCS = (
    CC_PINCH_SADDLE,
    CC_PINCH_CROSS,
    CC_VORTEX_Y,
    CC_SWIRL_Y,
    CC_VORTEX_X,
    CC_SWIRL_X,
    CC_VORTEX_STRENGTH,
    CC_SWIRL_STRENGTH,
    CC_RIPPLE_FREQ,
    CC_RIPPLE_AMP,
)

CHANNEL = 0  # the MPE master channel — never a member note channel
SAMPLE_HZ = 25.0
MIN_DT = 0.05  # per-CC emission rate cap (20 Hz)
_PRIORITY = 3  # same slot as in-note controller streams


def _clamp7(value: float) -> int:
    return max(0, min(127, round(value)))


class _Smooth:
    """One-pole smoother: gestures are arm movements, never steps."""

    def __init__(self, tau: float, initial: float = 0.0) -> None:
        self.tau = tau
        self.value = initial

    def step(self, target: float, dt: float) -> float:
        alpha = 1.0 - math.exp(-dt / self.tau)
        self.value += alpha * (target - self.value)
        return self.value


def _energy_curve(tracks: list[TrackData], times: list[float], tau: float) -> list[float]:
    """Normalized song energy over ``times``: sounding-note velocity mass,
    smoothed by ``tau`` seconds (the arm reacts slower than the ear)."""
    raw = []
    notes = [n for t in tracks for n in t.notes]
    for t in times:
        raw.append(sum(n.velocity for n in notes if n.perf_start <= t < n.perf_end))
    peak = max(raw) or 1.0
    smooth = _Smooth(tau)
    dt = times[1] - times[0] if len(times) > 1 else 0.04
    return [smooth.step(value / peak, dt) for value in raw]


def _pulses(times: list[float], hits: list[tuple[float, float]], decay: float) -> list[float]:
    """A pulse train: each ``(time, amplitude)`` hit decays exponentially."""
    curve = []
    hits = sorted(hits)
    for t in times:
        level = 0.0
        for hit_t, amp in hits:
            if hit_t > t:
                break
            level = max(level, amp * math.exp(-(t - hit_t) / decay))
        curve.append(level)
    return curve


def synthesize(
    tracks: list[TrackData],
    artifacts: Artifacts,
    rng: random.Random,
) -> list[TimedMessage]:
    """An Airwave gesture stream synced to the performance (channel 1 CCs)."""
    notes = [n for t in tracks for n in t.notes]
    if not notes:
        return []
    end = max(n.perf_end for n in notes) + 1.0
    dt = 1.0 / SAMPLE_HZ
    times = [i * dt for i in range(int(end / dt) + 1)]

    grid: BeatGrid | None = artifacts.grid()
    beat_len = 60.0 / grid.bpm if grid else 0.5
    bar_len = beat_len * (grid.felt_per_bar if grid else 4)

    def bar_phase(t: float) -> float:
        if grid is not None:
            pos = grid.pos(t)
            return (pos % grid.felt_per_bar) / grid.felt_per_bar
        return (t % bar_len) / bar_len

    energy = _energy_curve(tracks, times, tau=0.6)
    energy_slow = _energy_curve(tracks, times, tau=3.0)
    # The heavy smoothing undershoots on short material; renormalize so the
    # wind spans the song's own dynamic range.
    slow_peak = max(energy_slow) or 1.0
    energy_slow = [v / slow_peak for v in energy_slow]

    melody = _melody_track(tracks)
    melody_notes = melody.notes if melody is not None else []
    pitches = [n.pitch for n in melody_notes] or [60]
    pitch_lo, pitch_hi = min(pitches), max(pitches)
    pitch_span = max(1, pitch_hi - pitch_lo)

    # Pinches: saddle folds on downbeats (scaled by the moment's energy),
    # crossed tines at section boundaries and melody phrase peaks.
    downbeats = (
        [b for b in (grid.beats[grid.anchor :: grid.felt_per_bar]) if b < end]
        if grid is not None
        else [i * bar_len for i in range(int(end / bar_len))]
    )
    energy_at = dict(zip(times, energy, strict=True))

    def nearest_energy(t: float) -> float:
        return energy_at.get(round(t / dt) * dt, 0.5)

    saddle_hits = [(t, 25.0 + 95.0 * nearest_energy(t)) for t in downbeats]
    cross_hits = [(start, 127.0) for _, start, _ in artifacts.sections or []]
    cross_hits += [(n.perf_start, 50.0 + 0.6 * n.velocity) for n in melody_notes if n.phrase_peak]
    saddle = _pulses(times, saddle_hits, decay=0.25)
    cross = _pulses(times, cross_hits, decay=0.6)

    # Vortex strength — the ATTACK hand. The vortex is visually powerful even
    # at low values, so it must be silent between hits: no floor, a snappy
    # 0.2 s decay, and it fires only on ACCENTED notes (strong beats and
    # phrase peaks) so dense lines don't keep it permanently lit. Capped 47.
    accents = [n for n in melody_notes if n.metric_weight >= 0.75 or n.phrase_peak]
    if not accents and melody_notes:  # no beat grid: accent by velocity instead
        mean_velocity = sum(n.velocity for n in melody_notes) / len(melody_notes)
        accents = [n for n in melody_notes if n.velocity >= mean_velocity]
    vortex_hits = [(n.perf_start, 26.0 + 0.17 * n.velocity) for n in accents]
    vortex_pulses = _pulses(times, vortex_hits, decay=0.2)
    vortex_strength = [min(47.0, pulse) for pulse in vortex_pulses]

    # Swirl strength — the WIND over the water: a gentle continuous layer
    # following the song's slow energy; mild on screen, so it may use the
    # full range.
    swirl_strength = [min(127.0, 14.0 + 110.0 * energy_slow[i]) for i in range(len(times))]

    # The attack hand (vortex) tracks the melodic contour sideways.
    contour = _Smooth(0.15, 64.0)
    melody_sorted = sorted(melody_notes, key=lambda n: n.perf_start)
    vortex_x_contour: list[float] = []
    cursor = 0
    current_pitch = pitches[0]
    for t in times:
        while cursor < len(melody_sorted) and melody_sorted[cursor].perf_start <= t:
            current_pitch = melody_sorted[cursor].pitch
            cursor += 1
        target = 18.0 + 91.0 * (current_pitch - pitch_lo) / pitch_span
        vortex_x_contour.append(contour.step(target, dt))

    # The wind hand (swirl) drifts in slow multi-bar sways.
    sway_period = max(4.0, 8.0 * bar_len)
    phase_x = rng.uniform(0.0, 2.0 * math.pi)
    phase_y = rng.uniform(0.0, 2.0 * math.pi)
    wander = _Smooth(2.5)

    # Ripple wavelength: swells and decays WITH the amount — the same fast
    # energy curve drives both, so the waves grow and die with the music.
    # Never below 32: a zero wavelength renders nothing.
    ripple_rest = 45.0

    def ripple_wavelength(i: int, t: float) -> float:
        wave = 32.0 + 78.0 * energy[i] + 5.0 * math.sin(2.0 * math.pi * t / (2.0 * bar_len))
        return max(32.0, min(110.0, wave))

    events: list[TimedMessage] = []
    seq = 0
    last: dict[int, int] = {}
    last_t: dict[int, float] = {}

    def emit(t: float, cc: int, value: float) -> None:
        nonlocal seq
        clamped = _clamp7(value)
        if last.get(cc) == clamped:
            return
        if t - last_t.get(cc, -1.0) < MIN_DT:
            return
        message = mido.Message("control_change", channel=CHANNEL, control=cc, value=clamped)
        events.append((t, _PRIORITY, seq, message))
        last[cc] = clamped
        last_t[cc] = t
        seq += 1

    for i, t in enumerate(times):
        phase = bar_phase(t)
        drift = wander.step(rng.gauss(0.0, 10.0), dt)

        # Vortex: the attack — flares with the melody, lands where it plays.
        emit(t, CC_VORTEX_STRENGTH, vortex_strength[i])
        emit(t, CC_VORTEX_X, vortex_x_contour[i] + 10.0 * math.sin(2.0 * math.pi * phase))
        vortex_y = 64.0 - 40.0 * math.cos(2.0 * math.pi * phase)  # circles with the bar
        emit(t, CC_VORTEX_Y, 127.0 - vortex_y)  # Slide axes are reversed

        # Swirl: the wind — a broad slow layer drifting over the water.
        emit(t, CC_SWIRL_STRENGTH, swirl_strength[i])
        sway = 42.0 * math.sin(2.0 * math.pi * t / sway_period + phase_x)
        emit(t, CC_SWIRL_X, 64.0 + sway + drift)
        swirl_y = 64.0 + 30.0 * math.sin(2.0 * math.pi * t / (1.7 * sway_period) + phase_y)
        emit(t, CC_SWIRL_Y, 127.0 - swirl_y)

        emit(t, CC_PINCH_SADDLE, saddle[i])
        emit(t, CC_PINCH_CROSS, cross[i])

        emit(t, CC_RIPPLE_FREQ, ripple_wavelength(i, t))
        emit(t, CC_RIPPLE_AMP, 10.0 + 105.0 * energy[i])

    # Rest the hands at the end: strengths/positions/amount back to zero, but
    # the wavelength stays at its base — a zero wavelength renders nothing.
    for cc in AIRWAVE_CCS:
        value = _clamp7(ripple_rest) if cc == CC_RIPPLE_FREQ else 0
        message = mido.Message("control_change", channel=CHANNEL, control=cc, value=value)
        events.append((end, _PRIORITY, seq, message))
        seq += 1
    return events
