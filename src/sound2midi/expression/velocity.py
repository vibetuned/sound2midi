"""Module B — velocity synthesis from the musical context.

Transcribed MIDI may carry flat/fixed velocity, and score-derived MIDI has
none, so velocities are synthesized from the Module A features. When stem
loudness is available (Module F) the synthetic value is blended with the
measured one.
"""

from __future__ import annotations

import random

from sound2midi.expression.context import TrackData

V_BASE = 72.0
SHORT_NOTE_S = 0.120


def clamp_velocity(value: float) -> int:
    return max(20, min(127, round(value)))


def synthesize(
    track: TrackData,
    rng: random.Random,
    *,
    base: float = V_BASE,
    loudness_x: list[float | None] | None = None,
    vel_blend: float = 0.3,
) -> None:
    """Set every note's velocity from its features (in place, source order).

    ``loudness_x`` — optional per-note normalized stem loudness in [0, 1]
    (Module F); when a note has one, ``vel = w·vel_model + (1-w)·vel_loudness``
    with ``w = vel_blend``.
    """
    prev_pitch: int | None = None
    alternate = 1
    for i, note in enumerate(track.notes):
        vel = (
            base * note.section_intensity
            + 30.0 * note.metric_weight
            + 18.0 * min(1.0, abs(note.interval) / 12.0) * (1.25 if note.interval > 0 else 1.0)
            + 10.0 * note.tension
        )
        vel *= note.arc * (1.0 + rng.gauss(0.0, 0.04))

        if note.duration < SHORT_NOTE_S:
            vel -= 8.0
        if note.phrase_peak:
            vel += 6.0
        if prev_pitch == note.pitch:
            alternate = -alternate
            vel += 4.0 * alternate
        else:
            alternate = 1
        prev_pitch = note.pitch

        if note.legato_prev:  # inner legato notes get soft attacks (Module H)
            vel -= 10.0

        x = loudness_x[i] if loudness_x is not None else None
        if x is not None:
            vel_loud = 20.0 + 100.0 * max(0.0, min(1.0, x)) ** 0.8
            vel = vel_blend * vel + (1.0 - vel_blend) * vel_loud

        note.velocity = clamp_velocity(vel)
