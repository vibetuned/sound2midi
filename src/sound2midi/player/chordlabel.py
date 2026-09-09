"""Parse Harte-style chord labels (lv-chordia output). Pure — no Qt/music21.

Labels look like ``C:maj``, ``A:min7``, ``E:maj/3`` (bass as a chord degree,
so ``/3`` = first inversion) or ``N`` (no chord). The parsing core lives in
``sound2midi.harte`` (shared with the MPE expression engine) and is re-exported
here; this module keeps the chord *realization* (voicings, display) the chord
strip and the exporter use.
"""

from __future__ import annotations

from sound2midi.harte import (
    DEGREE_SEMITONES as DEGREE_SEMITONES,
)
from sound2midi.harte import (
    FLAT_NAMES as _FLAT_NAMES,
)
from sound2midi.harte import (
    KIND_DISPLAY as KIND_DISPLAY,
)
from sound2midi.harte import (
    KIND_M21 as KIND_M21,
)
from sound2midi.harte import (
    KIND_TONES as KIND_TONES,
)
from sound2midi.harte import (
    ROOT_PC as ROOT_PC,
)
from sound2midi.harte import (
    SHARP_NAMES as _SHARP_NAMES,
)
from sound2midi.harte import (
    ParsedChord as ParsedChord,
)
from sound2midi.harte import (
    base_kind as base_kind,
)
from sound2midi.harte import (
    bass_pc as bass_pc,
)
from sound2midi.harte import (
    parse as parse,
)


def voicing(parsed: ParsedChord) -> tuple[int, ...]:
    """MIDI notes realizing the chord: bass note first (C2 octave), then up to
    four chord tones in root position around C3 — a simple piano comp voicing."""
    root = ROOT_PC[parsed.root]
    tones = KIND_TONES.get(parsed.kind) or KIND_TONES[base_kind(parsed.kind)]
    bass = bass_pc(parsed)
    if bass is None:
        bass = root
    return (36 + bass, *(48 + root + t for t in tones[:4]))


# Chord realization styles: how a labeled progression becomes actual notes.
REALIZE_STYLES = ("block", "smooth", "arpeggio", "bass")


def _bass_note(parsed: ParsedChord) -> int:
    pc = bass_pc(parsed)
    if pc is None:
        pc = ROOT_PC[parsed.root]
    return 36 + pc  # C2 octave


def _tone_pcs(parsed: ParsedChord) -> list[int]:
    tones = KIND_TONES.get(parsed.kind) or KIND_TONES[base_kind(parsed.kind)]
    root = ROOT_PC[parsed.root]
    seen: list[int] = []
    for t in tones[:4]:
        pc = (root + t) % 12
        if pc not in seen:
            seen.append(pc)
    return seen


def _stack(pcs: list[int], *, floor: int = 52) -> list[int]:
    """Place pitch classes as an ascending stack starting at/above ``floor``."""
    notes: list[int] = []
    for pc in pcs:
        if not notes:
            notes.append(floor + ((pc - floor) % 12))
        else:
            step = (pc - notes[-1]) % 12
            notes.append(notes[-1] + (step or 12))
    return notes


def _smooth_stack(pcs: list[int], previous: list[int] | None) -> list[int]:
    """The inversion (rotation) of the chord tones closest to the previous voicing."""
    candidates = [_stack(pcs[i:] + pcs[:i]) for i in range(len(pcs))]
    if previous is None:
        return candidates[0]  # root position to start
    prev = previous

    def distance(stack: list[int]) -> int:
        return sum(min(abs(n - p) for p in prev) for n in stack)

    return min(candidates, key=distance)


def realize_chords(
    segments: list[tuple[str, float, float]],
    style: str = "block",
    *,
    beats: list[float] | None = None,
    gap: float = 0.02,
) -> list[tuple[float, float, tuple[int, ...]]]:
    """Turn labeled ``(label, start, end)`` segments into ``(start, end, notes)``
    events (times in seconds, MIDI note numbers; simultaneous notes share an event).

    Styles: ``block`` — bass + root-position chord tones, one hit per chord;
    ``smooth`` — like block, but each chord takes the inversion closest to the
    previous one (voice-led); ``arpeggio`` — bass then chord tones cycled upward,
    one note per beat (``beats``; falls back to 0.5 s steps); ``bass`` — bass
    note only.
    """
    if style not in REALIZE_STYLES:
        raise ValueError(f"Unknown chord style {style!r}; choose from {REALIZE_STYLES}.")
    events: list[tuple[float, float, tuple[int, ...]]] = []
    previous: list[int] | None = None
    for label, start, end in segments:
        parsed = parse(label)
        if parsed is None or end - start < 0.05:
            continue
        off = max(start + 0.05, end - gap)
        if style == "bass":
            events.append((start, off, (_bass_note(parsed),)))
        elif style == "block":
            events.append((start, off, voicing(parsed)))
        elif style == "smooth":
            stack = _smooth_stack(_tone_pcs(parsed), previous)
            events.append((start, off, (_bass_note(parsed), *stack)))
            previous = stack
        else:  # arpeggio
            sequence = [_bass_note(parsed), *_stack(_tone_pcs(parsed))]
            times = [b for b in beats if start - 0.01 <= b < end - 0.05] if beats else []
            if not times or times[0] > start + 0.35:
                times = [start, *times]
            for i, t in enumerate(times):
                nxt = times[i + 1] if i + 1 < len(times) else end
                events.append((t, max(t + 0.05, nxt - gap), (sequence[i % len(sequence)],)))
    return events


def display(label: str) -> str:
    """Compact human label: 'A:min7' -> 'Am7', 'E:maj/3' -> 'E/G#', 'N' -> ''."""
    parsed = parse(label)
    if parsed is None:
        return ""
    suffix = KIND_DISPLAY.get(parsed.kind)
    if suffix is None:
        suffix = KIND_DISPLAY.get(base_kind(parsed.kind), parsed.kind)
    text = parsed.root + suffix
    pc = bass_pc(parsed)
    if pc is not None and pc != ROOT_PC[parsed.root]:
        names = _FLAT_NAMES if "b" in parsed.root else _SHARP_NAMES
        text += "/" + names[pc]
    return text
