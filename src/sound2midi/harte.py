"""Parse Harte-style chord labels (lv-chordia output). Pure — no Qt/music21.

Labels look like ``C:maj``, ``A:min7``, ``E:maj/3`` (bass as a chord degree,
so ``/3`` = first inversion) or ``N`` (no chord). This module holds the parsing
core shared by the player (chord strip, exporter) and the MPE expression
engine; chord *realization* (voicings, display) stays in
``sound2midi.player.chordlabel``.
"""

from __future__ import annotations

from dataclasses import dataclass

ROOT_PC = {
    "C": 0, "B#": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "E#": 5, "F": 5, "F#": 6, "Gb": 6, "G": 7,
    "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11,
}  # fmt: skip

# Harte chord degree -> semitones above the root.
DEGREE_SEMITONES = {
    "1": 0, "b2": 1, "2": 2, "#2": 3, "b3": 3, "3": 4, "4": 5, "#4": 6,
    "b5": 6, "5": 7, "#5": 8, "b6": 8, "6": 9, "#6": 10, "bb7": 9,
    "b7": 10, "7": 11, "b9": 1, "9": 2, "#9": 3, "11": 5, "#11": 6,
    "b13": 8, "13": 9,
}  # fmt: skip

# Chord kind -> compact display suffix ("" = plain major triad).
KIND_DISPLAY = {
    "maj": "", "min": "m", "7": "7", "maj7": "maj7", "min7": "m7",
    "dim": "dim", "dim7": "dim7", "hdim7": "m7b5", "aug": "+",
    "minmaj7": "mM7", "maj6": "6", "min6": "m6", "9": "9", "maj9": "maj9",
    "min9": "m9", "11": "11", "min11": "m11", "13": "13", "maj13": "maj13",
    "min13": "m13", "sus2": "sus2", "sus4": "sus4", "5": "5", "1": "",
}  # fmt: skip

# Chord kind -> chord tones as semitones above the root (root-position voicing).
KIND_TONES = {
    "maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6), "aug": (0, 4, 8),
    "maj7": (0, 4, 7, 11), "min7": (0, 3, 7, 10), "7": (0, 4, 7, 10),
    "dim7": (0, 3, 6, 9), "hdim7": (0, 3, 6, 10), "minmaj7": (0, 3, 7, 11),
    "maj6": (0, 4, 7, 9), "min6": (0, 3, 7, 9),
    "9": (0, 4, 7, 10), "maj9": (0, 4, 7, 11), "min9": (0, 3, 7, 10),
    "11": (0, 4, 7, 10), "min11": (0, 3, 7, 10), "13": (0, 4, 7, 10),
    "maj13": (0, 4, 7, 11), "min13": (0, 3, 7, 10),
    "sus2": (0, 2, 7), "sus4": (0, 5, 7), "5": (0, 7), "1": (0,),
}  # fmt: skip

# Chord kind -> music21 harmony.ChordSymbol kind string.
KIND_M21 = {
    "maj": "major", "min": "minor", "dim": "diminished", "aug": "augmented",
    "maj7": "major-seventh", "min7": "minor-seventh", "7": "dominant-seventh",
    "dim7": "diminished-seventh", "hdim7": "half-diminished-seventh",
    "minmaj7": "minor-major-seventh", "maj6": "major-sixth", "min6": "minor-sixth",
    "9": "dominant-ninth", "maj9": "major-ninth", "min9": "minor-ninth",
    "11": "dominant-11th", "min11": "minor-11th", "13": "dominant-13th",
    "maj13": "major-13th", "min13": "minor-13th",
    "sus2": "suspended-second", "sus4": "suspended-fourth",
    "5": "power", "1": "pedal",
}  # fmt: skip

SHARP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FLAT_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")


@dataclass
class ParsedChord:
    root: str  # e.g. "C#"
    kind: str  # e.g. "min7" (extensions like "(9)" stripped into base_kind)
    bass: str | None  # Harte degree, e.g. "3", "b7", or None


def parse(label: str) -> ParsedChord | None:
    """Parse a Harte-style label; None for no-chord (N/X) or unparseable input."""
    label = label.strip()
    if not label or label in ("N", "X"):
        return None
    root, _, rest = label.partition(":")
    if root not in ROOT_PC:
        return None
    if not rest:
        rest = "maj"
    kind, _, bass = rest.partition("/")
    kind = kind.split("(")[0] or "maj"  # strip parenthesized extensions
    return ParsedChord(root=root, kind=kind, bass=bass or None)


def base_kind(kind: str) -> str:
    """Reduce an unknown kind to a family the maps know (minor-ish else major)."""
    if kind in KIND_M21:
        return kind
    return "min" if kind.startswith("min") else "maj"


def bass_pc(parsed: ParsedChord) -> int | None:
    """Pitch class of the bass note, if the label carries an inversion degree."""
    if parsed.bass is None:
        return None
    semis = DEGREE_SEMITONES.get(parsed.bass)
    if semis is None:
        return None
    return (ROOT_PC[parsed.root] + semis) % 12


def chord_pcs(parsed: ParsedChord) -> frozenset[int]:
    """Pitch classes of the chord tones (root position, bass included)."""
    root = ROOT_PC[parsed.root]
    tones = KIND_TONES.get(parsed.kind) or KIND_TONES[base_kind(parsed.kind)]
    pcs = {(root + t) % 12 for t in tones}
    bass = bass_pc(parsed)
    if bass is not None:
        pcs.add(bass)
    return frozenset(pcs)
