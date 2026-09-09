"""Modules C & H — expression curve generation (pressure, CC74, vibrato, legato).

Per note, three streams are sampled at 60 Hz internally (emission is
delta-thresholded by the encoder). All per-note jitters come from the seeded
RNG, so a fixed ``--seed`` reproduces the output byte-for-byte while identical
consecutive notes still never yield identical curves.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, fields
from importlib import resources
from itertools import pairwise
from typing import TYPE_CHECKING

import yaml

from sound2midi.expression.context import Note, TrackData
from sound2midi.expression.mpe_encoder import PerfNote

if TYPE_CHECKING:
    from sound2midi.expression.loudness import StemCurve

SAMPLE_HZ = 60.0
PRESSURE_BASE = 18.0
CC74_FLOOR = 24.0
LEGATO_MODES = ("auto", "glide", "overlap", "off")
LEGATO_GAP_S = 0.060


@dataclass(frozen=True)
class Profile:
    """Stream enables + weight overrides for one instrument family."""

    name: str = "strings"
    pressure: bool = True
    vibrato: bool = True
    cc74: bool = True
    swell: bool = True  # False: attack transient + decay only (pluck/keys)
    attack: bool = True  # False: slow swell, no attack transient (pads)
    attack_time: float = 0.03
    beta: float = 0.85  # the u exponent in the swell shape sin^1.2(π·u^β)
    vib_depth: float = 1.0
    vib_onset: float = 1.0  # >1 = later vibrato onset
    vib_rate: float = 1.0
    cc74_tracks_decay: bool = False  # CC74 follows the pressure decay only
    legato: str = "off"  # mode used when --legato auto
    legato_interval: float = 4.0  # max |Δp| (st) for a legato transition


_PROFILE_CACHE: dict[str, Profile] = {}
_PROFILE_FIELDS = {f.name for f in fields(Profile)}


def load_profile(name: str) -> Profile:
    """Load ``profiles/<name>.yaml`` shipped with the package."""
    if name in _PROFILE_CACHE:
        return _PROFILE_CACHE[name]
    path = resources.files("sound2midi.expression") / "profiles" / f"{name}.yaml"
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        raise ValueError(f"Unknown profile {name!r}.") from None
    data = {k: v for k, v in (raw or {}).items() if k in _PROFILE_FIELDS}
    profile = Profile(**data)
    _PROFILE_CACHE[name] = profile
    return profile


def available_profiles() -> list[str]:
    root = resources.files("sound2midi.expression") / "profiles"
    return sorted(p.name[:-5] for p in root.iterdir() if p.name.endswith(".yaml"))


# Stem/track-name keyword -> profile (spec §4.5 auto-selection).
_AUTO_KEYWORDS = (
    ("drum", "none"),
    ("perc", "none"),
    ("vocal", "sung"),
    ("voice", "sung"),
    ("sing", "sung"),
    ("guitar", "pluck"),
    ("bass", "pluck"),
    ("piano", "keys"),
    ("keys", "keys"),
    ("organ", "pads"),
    ("pad", "pads"),
    ("flute", "winds"),
    ("sax", "winds"),
    ("wind", "winds"),
)


def auto_profile(track_name: str, *, is_drum: bool = False) -> str:
    """Pick a profile from a (stem) track name; unmatched melodic tracks → strings."""
    if is_drum:
        return "none"
    lowered = track_name.lower()
    for keyword, profile in _AUTO_KEYWORDS:
        if keyword in lowered:
            return profile
    return "strings"


def _smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


@dataclass
class _NoteJitter:
    """Humanization values drawn once per note from the seeded RNG."""

    swell: float
    vib_onset: float
    vib_depth: float
    vib_rate: float  # r_note, clamped so the rate stays in the plausible band
    wobble_phase: float

    @classmethod
    def draw(cls, rng: random.Random) -> _NoteJitter:
        return cls(
            swell=1.0 + rng.uniform(-0.1, 0.1),
            vib_onset=rng.uniform(-0.03, 0.03),
            vib_depth=1.0 + rng.uniform(-0.1, 0.1),
            vib_rate=max(-0.4, min(0.4, rng.gauss(0.0, 0.3))),
            wobble_phase=rng.uniform(0.0, 2.0 * math.pi),
        )


@dataclass
class _Drift:
    """Low-pass-filtered gaussian noise (~1.5 Hz cutoff, ±4) so held notes
    never sit on a perfect curve. Two cascaded poles keep the sample-to-sample
    movement smooth, so the delta-thresholded encoder stays quiet."""

    rng: random.Random
    stage1: float = 0.0
    stage2: float = 0.0

    def step(self, dt: float) -> float:
        alpha = 1.0 - math.exp(-2.0 * math.pi * 1.5 * dt)
        self.stage1 += alpha * (self.rng.gauss(0.0, 8.0) - self.stage1)
        self.stage2 += alpha * (self.stage1 - self.stage2)
        return max(-4.0, min(4.0, self.stage2))


@dataclass
class _GlideSegment:
    start: float  # transition start (absolute seconds)
    duration: float  # S-curve travel time
    from_st: float
    to_st: float


def _glide_offset(segments: list[_GlideSegment], t: float) -> float:
    """Cumulative pitch offset (semitones) of a glide chain at time ``t``."""
    offset = segments[0].from_st if segments else 0.0
    for seg in segments:
        if t < seg.start:
            break
        u = _smoothstep((t - seg.start) / seg.duration) if seg.duration > 0 else 1.0
        offset = seg.from_st + (seg.to_st - seg.from_st) * u
    return offset


def detect_legato(
    track: TrackData, profile: Profile, mode: str, bend_range: int
) -> list[list[Note]]:
    """Group the track's notes into legato chains (singletons included).

    Detection (spec §9): within a phrase, consecutive notes qualify when the
    gap between them is ≤ 60 ms (measured after rubato) and |Δp| is within the
    profile threshold. In ``glide`` mode a chain whose cumulative offset from
    its root would leave ±(bend_range - 2) st splits with a clean retrigger.
    Inner notes are marked ``legato_prev`` for the velocity stage.
    """
    resolved = profile.legato if mode == "auto" else mode
    notes = sorted(track.notes, key=lambda n: (n.perf_start, -n.pitch))
    if resolved == "off" or track.is_drum or not notes:
        return [[n] for n in notes]

    budget = (bend_range - 2.0) if resolved == "glide" else math.inf
    groups: list[list[Note]] = [[notes[0]]]
    for prev, note in pairwise(notes):
        gap = note.perf_start - prev.perf_end
        chord = note.perf_start - prev.perf_start <= 0.03
        joins = (
            not chord
            and note.phrase == prev.phrase
            and -0.03 <= gap <= LEGATO_GAP_S
            and abs(note.pitch - prev.pitch) <= profile.legato_interval
            and abs(note.pitch - groups[-1][0].pitch) <= budget
        )
        if joins:
            note.legato_prev = True
            groups[-1].append(note)
        else:
            note.legato_prev = False
            groups.append([note])
    return groups


def _sample_streams(
    *,
    t_start: float,
    duration: float,
    velocity: int,
    note: Note,
    profile: Profile,
    rng: random.Random,
    att_scale: float = 1.0,
    glide: list[_GlideSegment] | None = None,
    bumps: list[tuple[float, float]] | None = None,  # (time offset, amplitude)
    loudness: list[float] | None = None,
    breath: bool = False,
) -> list[tuple[float, float, float, float]]:
    """Sample (t, pressure, cc74, cents) at 60 Hz over ``duration`` seconds.

    ``breath`` shapes the pressure like a wind player's air column: no
    percussive attack (breath builds over ~60 ms instead), support dropping
    slightly as the phrase spends its air, and a taper on the phrase-final
    note; legato-joined notes keep the column going without re-attacking.
    """
    jitter = _NoteJitter.draw(rng)
    drift = _Drift(rng)
    dt = 1.0 / SAMPLE_HZ
    steps = max(1, int(duration / dt))

    arc = note.arc
    swing = min(1.0, abs(note.interval) / 12.0)
    a_swell = (
        arc
        * note.section_intensity
        * 45.0
        * (0.4 * note.metric_weight + 0.35 * swing + 0.25 * note.tension)
        * jitter.swell
        if profile.swell
        else 0.0
    )
    a_att = 25.0 * (velocity / 96.0) * att_scale if profile.attack and not breath else 0.0
    decay_tau = max(0.25, 0.6 * duration)  # pluck/keys pressure decay

    vibrato_on = profile.vibrato and duration >= 0.35
    t_onset = 0.0
    a_max = 0.0
    if vibrato_on:
        t_onset = max(0.0, min(0.3, 0.35 * duration) * profile.vib_onset + jitter.vib_onset)
        a_max = arc * 35.0 * (0.6 + 0.4 * note.tension) * profile.vib_depth * jitter.vib_depth
    phase = 0.0

    samples: list[tuple[float, float, float, float]] = []
    for k in range(steps + 1):
        t = min(k * dt, duration)
        u = t / duration if duration > 0 else 0.0

        pressure = 0.0
        if profile.pressure:
            pressure = PRESSURE_BASE + a_att * math.exp(-t / profile.attack_time)
            if profile.swell:
                pressure += a_swell * math.sin(math.pi * u**profile.beta) ** 1.2
            else:
                pressure *= math.exp(-t / decay_tau)
            if bumps:
                for bump_t, bump_amp in bumps:
                    if t >= bump_t:
                        pressure += bump_amp * math.exp(-(t - bump_t) / 0.08)
            if breath:
                envelope = 1.0 - 0.15 * min(1.0, max(0.0, note.phrase_pos))  # air spent
                if not note.legato_prev:
                    envelope *= _smoothstep(t / 0.06)  # breath builds
                if note.phrase_final and u > 0.7:
                    envelope *= 1.0 - 0.5 * _smoothstep((u - 0.7) / 0.3)  # release taper
                pressure *= envelope
            pressure += drift.step(dt)
            if loudness is not None and k < len(loudness):
                pressure = 0.5 * pressure + 0.5 * (127.0 * loudness[k])
            pressure = min(127.0, max(0.0, pressure))

        if profile.cc74:
            if profile.cc74_tracks_decay:
                cc74 = CC74_FLOOR + 0.55 * pressure
            else:
                cc74 = (
                    CC74_FLOOR
                    + 0.55 * pressure
                    + 25.0 * note.tension
                    + 2.0 * max(0.0, note.interval) * math.exp(-t / 0.08)
                )
                if bumps:
                    for bump_t, _ in bumps:
                        if t >= bump_t:
                            cc74 += 12.0 * math.exp(-(t - bump_t) / 0.08)
            cc74 = min(127.0, max(0.0, cc74))
        else:
            cc74 = CC74_FLOOR

        cents = 0.0
        if vibrato_on:
            f_v = (4.8 + 1.4 * u + jitter.vib_rate) * profile.vib_rate
            phase += 2.0 * math.pi * f_v * dt  # accumulate phase — never sin(2π·f(t)·t)
            depth = a_max * _smoothstep((t - t_onset) / 0.25)
            wobble = 1.0 + 0.08 * math.sin(2.0 * math.pi * 0.7 * t + jitter.wobble_phase)
            cents = depth * wobble * math.sin(phase)
        if glide is not None:
            cents += 100.0 * _glide_offset(glide, t_start + t)

        samples.append((t_start + t, pressure, cc74, cents))
    return samples


@dataclass
class RenderOptions:
    bend_range: int = 48
    legato_mode: str = "auto"
    loudness_pressure: bool = False
    loudness_curve: StemCurve | None = None  # the track's stem loudness
    dry: bool = False  # velocity only, expression muted (A/B)
    breath: bool = False  # wind-instrument pressure shaping (--wind)


_DRY_PROFILE = Profile(name="none", pressure=False, vibrato=False, cc74=False)


def render_track(
    track: TrackData,
    groups: list[list[Note]],
    profile: Profile,
    rng: random.Random,
    options: RenderOptions,
) -> list[PerfNote]:
    """Render the track's legato groups into performance notes with streams."""
    if options.dry:
        profile = _DRY_PROFILE
    silent = (
        track.is_drum
        or profile.name == "none"
        or not (profile.pressure or profile.vibrato or profile.cc74)
    )
    resolved = profile.legato if options.legato_mode == "auto" else options.legato_mode

    phrase_mean: dict[int, float] = {}
    for note in track.notes:
        phrase_mean.setdefault(note.phrase, 0.0)
    for phrase in phrase_mean:
        members = [n.velocity for n in track.notes if n.phrase == phrase]
        phrase_mean[phrase] = sum(members) / len(members)

    perf: list[PerfNote] = []
    for group in groups:
        if silent:
            for note in group:
                perf.append(
                    PerfNote(
                        track=track.index,
                        start=note.perf_start,
                        end=note.perf_end,
                        pitch=note.pitch,
                        velocity=note.velocity,
                        samples=[],
                    )
                )
        elif resolved == "glide" and len(group) > 1:
            perf.append(_render_glide(track, group, profile, rng, options))
        else:
            perf.extend(_render_overlap(track, group, profile, rng, options, phrase_mean))
    perf.sort(key=lambda n: n.start)
    return perf


def _loudness_samples(options: RenderOptions, start: float, duration: float) -> list[float] | None:
    curve = options.loudness_curve
    if not options.loudness_pressure or curve is None:
        return None
    n = max(2, int(duration * SAMPLE_HZ) + 1)
    return curve.resample(start, start + duration, n)


def _render_glide(
    track: TrackData,
    group: list[Note],
    profile: Profile,
    rng: random.Random,
    options: RenderOptions,
) -> PerfNote:
    """A legato group as one sustained MPE note whose bend travels (spec §9)."""
    root = group[0]
    duration = group[-1].perf_end - root.perf_start
    segments: list[_GlideSegment] = []
    bumps: list[tuple[float, float]] = []
    offset = 0.0
    for prev, note in pairwise(group):
        delta = note.pitch - prev.pitch
        travel = min(0.08, max(0.03, 0.012 * abs(delta)))
        segments.append(
            _GlideSegment(
                start=note.perf_start,
                duration=travel,
                from_st=offset,
                to_st=offset + delta,
            )
        )
        offset += delta
        bumps.append((note.perf_start - root.perf_start, rng.uniform(10.0, 15.0)))

    samples = _sample_streams(
        t_start=root.perf_start,
        duration=duration,
        velocity=root.velocity,
        note=root,
        profile=profile,
        rng=rng,
        glide=segments,
        bumps=bumps,
        loudness=_loudness_samples(options, root.perf_start, duration),
        breath=options.breath,
    )
    return PerfNote(
        track=track.index,
        start=root.perf_start,
        end=root.perf_start + duration,
        pitch=root.pitch,
        velocity=root.velocity,
        samples=samples,
    )


def _render_overlap(
    track: TrackData,
    group: list[Note],
    profile: Profile,
    rng: random.Random,
    options: RenderOptions,
    phrase_mean: dict[int, float],
) -> list[PerfNote]:
    """Per-note channels; chained notes overlap and soften their attack (spec §9)."""
    resolved = profile.legato if options.legato_mode == "auto" else options.legato_mode
    overlap_on = resolved == "overlap" and len(group) > 1
    perf: list[PerfNote] = []
    for i, note in enumerate(group):
        end = note.perf_end
        att_scale = 1.0
        velocity = note.velocity
        if overlap_on:
            if i < len(group) - 1:  # extend into the successor by 20-40 ms
                end = min(group[i + 1].perf_start + rng.uniform(0.020, 0.040), note.perf_end + 0.08)
                end = max(end, note.perf_end)
            if i > 0:  # suppressed attack, velocity toward the phrase mean
                att_scale = 0.3
                velocity = round((velocity + phrase_mean.get(note.phrase, velocity)) / 2)
        duration = max(0.02, end - note.perf_start)
        samples = _sample_streams(
            t_start=note.perf_start,
            duration=duration,
            velocity=velocity,
            note=note,
            profile=profile,
            rng=rng,
            att_scale=att_scale,
            loudness=_loudness_samples(options, note.perf_start, duration),
            breath=options.breath,
        )
        perf.append(
            PerfNote(
                track=track.index,
                start=note.perf_start,
                end=note.perf_start + duration,
                pitch=note.pitch,
                velocity=velocity,
                samples=samples,
            )
        )
    return perf
