"""Module A — per-note musical features from the MIDI file and the song artifacts.

Every artifact is optional: missing ones degrade to the documented fallbacks
(PPQ grid for meter, key-diatonic distance for tension, constant section
intensity), and :func:`load_artifacts` records what was missing so the CLI can
log it.
"""

from __future__ import annotations

import json
import math
import statistics
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path

import mido

from sound2midi.beatgrid import BeatGrid, tick_to_sec_fn
from sound2midi.harte import ROOT_PC, chord_pcs
from sound2midi.harte import parse as parse_chord

# Section label -> intensity scalar (spec defaults).
SECTION_INTENSITY = {
    "intro": 0.7,
    "verse": 0.85,
    "pre-chorus": 0.95,
    "chorus": 1.0,
    "bridge": 0.9,
    "inst": 0.95,
    "outro": 0.7,
    "silence": 0.5,
}
DEFAULT_SECTION_INTENSITY = 0.9
DEFAULT_TENSION = 0.3

_MAJOR_PCS = frozenset({0, 2, 4, 5, 7, 9, 11})
_MINOR_PCS = frozenset({0, 2, 3, 5, 7, 8, 10})

# Interval above the chord root (semitones) -> harmonic tension (spec table).
_CHORD_TENSION = {
    0: 0.0, 7: 0.0,          # root / 5th
    3: 0.2, 4: 0.2,          # 3rd
    9: 0.4, 11: 0.4,         # 6th / maj7
    10: 0.7, 1: 0.7, 2: 0.7, 5: 0.7,  # 7th / 9ths / sus
    6: 1.0, 8: 1.0,          # tritone-vs-root / chromatic
}  # fmt: skip

# Onsets closer together than this are treated as one chord attack.
CHORD_CLUSTER_S = 0.03

# Stem names produced by the separation step, longest keyword first. Per-stem
# files are named ``<song>_<stem>`` (MIDIs under ``stems/midi/``, WAVs under
# ``stems/<song>/``).
STEM_KEYWORDS = ("vocals", "vocal", "guitar", "drums", "drum", "piano", "bass", "other")


def stem_hint(midi_path: Path) -> str | None:
    """The stem a per-stem MIDI belongs to, from its ``<song>_<stem>`` filename."""
    name = midi_path.stem.lower()
    for suffix in (".mpe", ".stems"):
        name = name.removesuffix(suffix)
    for keyword in STEM_KEYWORDS:
        if name == keyword or name.endswith(f"_{keyword}"):
            return keyword
    return None


def phrase_arc(pos: float) -> float:
    """The phrase dynamic arc Φ(φ) = 0.6 + 0.4·sin(π·φ^0.9)."""
    pos = min(1.0, max(0.0, pos))
    return 0.6 + 0.4 * math.sin(math.pi * pos**0.9)


@dataclass
class Note:
    """One note with its source timing, performance timing, and features."""

    track: int
    pitch: int
    velocity: int
    channel: int
    start: float  # source seconds (tempo-map true)
    end: float
    start_tick: int
    # Performance timing (rubato-displaced); initialized to the source timing.
    perf_start: float = 0.0
    perf_end: float = 0.0
    # Features (Module A).
    interval: float = 0.0  # Δp vs previous top-voice note, semitones
    metric_weight: float = 0.1
    tension: float = DEFAULT_TENSION
    phrase: int = 0
    phrase_pos: float = 0.5
    phrase_peak: bool = False
    phrase_final: bool = False
    section: str = ""
    section_intensity: float = DEFAULT_SECTION_INTENSITY
    legato_prev: bool = False  # joined to the previous note in a legato group

    @property
    def duration(self) -> float:
        return self.perf_end - self.perf_start

    @property
    def arc(self) -> float:
        return phrase_arc(self.phrase_pos)


@dataclass
class TrackData:
    index: int
    name: str
    program: int
    is_drum: bool
    notes: list[Note] = field(default_factory=list)


@dataclass
class Artifacts:
    meter: dict | None = None
    key: str | None = None
    chords: list[tuple[str, float, float]] | None = None
    sections: list[tuple[str, float, float]] | None = None
    missing: tuple[str, ...] = ()

    def grid(self) -> BeatGrid | None:
        if not self.meter or len(self.meter.get("beats") or ()) < 2:
            return None
        return BeatGrid(self.meter)


def song_name(midi_path: Path) -> str:
    """The song id a MIDI belongs to: the stem minus render/stem suffixes.

    ``<id>.stems.mid``, ``<id>.stems.mpe.mid`` and the per-stem
    ``<id>_vocals.mid`` all map to ``<id>``.
    """
    name = midi_path.stem
    while True:
        for suffix in (".mpe", ".stems"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        else:
            for keyword in STEM_KEYWORDS:
                if name.lower().endswith(f"_{keyword}"):
                    name = name[: -len(keyword) - 1]
                    break
            else:
                return name


def _find_artifacts_dir(midi_path: Path, song: str) -> Path:
    """The nearest ``artifacts/`` dir (walking up) that holds this song's files.

    Handles both the merged layout (``output/<id>/<id>.stems.mid`` with
    ``artifacts/`` as a sibling) and per-stem MIDIs two levels down
    (``output/<id>/stems/midi/<id>_vocals.mid``).
    """
    for base in list(midi_path.resolve().parents)[:3]:
        candidate = base / "artifacts"
        if candidate.is_dir() and any(candidate.glob(f"{song}.*.json")):
            return candidate
    return midi_path.parent / "artifacts"


def load_artifacts(midi_path: Path, artifacts_dir: Path | None = None) -> Artifacts:
    """Discover the song's artifacts from the song-folder layout.

    Default: the nearest ``artifacts/<song>.<kind>.json`` walking up from the
    MIDI, so per-stem MIDIs under ``stems/midi/`` find the song's artifacts too.
    """
    song = song_name(midi_path)
    base = artifacts_dir if artifacts_dir is not None else _find_artifacts_dir(midi_path, song)

    def read(kind: str) -> dict | None:
        path = base / f"{song}.{kind}.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    missing: list[str] = []
    meter = read("meter")
    if not meter or len(meter.get("beats") or ()) < 2:
        meter = None
        missing.append("meter")

    key_data = read("key")
    key = str(key_data["key"]) if key_data and key_data.get("key") else None
    if key is None:
        missing.append("key")

    chords_data = read("chords")
    chords = None
    if chords_data and chords_data.get("chords"):
        chords = [
            (str(c["label"]), float(c["start"]), float(c["end"])) for c in chords_data["chords"]
        ]
    if not chords:
        chords = None
        missing.append("chords")

    sections_data = read("sections")
    sections = None
    if sections_data and sections_data.get("segments"):
        sections = [
            (str(s["label"]), float(s["start"]), float(s["end"])) for s in sections_data["segments"]
        ]
    if not sections:
        sections = None
        missing.append("sections")

    return Artifacts(meter=meter, key=key, chords=chords, sections=sections, missing=tuple(missing))


def extract_tracks(mid: mido.MidiFile) -> list[TrackData]:
    """Group the file's notes per track (a track = a voice/instrument lane)."""
    tick_to_sec = tick_to_sec_fn(mid)
    tracks: list[TrackData] = []
    for index, track in enumerate(mid.tracks):
        name = ""
        program = 0
        notes: list[Note] = []
        open_notes: dict[tuple[int, int], list[tuple[int, int]]] = {}
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.is_meta and msg.type == "track_name" and not name:
                name = msg.name
            elif msg.type == "program_change":
                program = msg.program
            elif msg.type == "note_on" and msg.velocity > 0:
                open_notes.setdefault((msg.channel, msg.note), []).append((tick, msg.velocity))
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                stack = open_notes.get((msg.channel, msg.note))
                if stack:
                    start_tick, velocity = stack.pop(0)
                    start = tick_to_sec(start_tick)
                    end = max(start + 0.01, tick_to_sec(tick))
                    notes.append(
                        Note(
                            track=index,
                            pitch=msg.note,
                            velocity=velocity,
                            channel=msg.channel,
                            start=start,
                            end=end,
                            start_tick=start_tick,
                            perf_start=start,
                            perf_end=end,
                        )
                    )
        notes.sort(key=lambda n: (n.start, -n.pitch))
        is_drum = ("drum" in name.lower()) or any(n.channel == 9 for n in notes)
        tracks.append(
            TrackData(index=index, name=name, program=program, is_drum=is_drum, notes=notes)
        )
    return tracks


def _segment_at(segments: list[tuple[str, float, float]], t: float) -> str | None:
    starts = [s[1] for s in segments]
    i = bisect_right(starts, t) - 1
    if i < 0:
        return None
    label, start, end = segments[i]
    return label if start <= t < end + 1e-9 else None


def _metric_weight_grid(pos: float, felt_per_bar: int) -> float:
    """Spec weights from a felt-beat position (0 = first downbeat)."""
    tol = 0.10
    nearest = round(pos)
    if abs(pos - nearest) < tol:
        beat_in_bar = nearest % felt_per_bar
        if beat_in_bar == 0:
            return 1.0
        if felt_per_bar % 2 == 0 and beat_in_bar == felt_per_bar // 2:
            return 0.75
        return 0.5
    frac = pos - math.floor(pos)
    if abs(frac - 0.5) < tol:
        return 0.25
    return 0.1


def _key_tension(pitch: int, key: str | None) -> float:
    """Fallback tension from the key's diatonic set (spec: N spans / no chords)."""
    if not key:
        return DEFAULT_TENSION
    parts = key.split()
    tonic = ROOT_PC.get(parts[0])
    if tonic is None:
        return DEFAULT_TENSION
    minor = len(parts) > 1 and parts[1].lower().startswith("min")
    degree = (pitch - tonic) % 12
    if degree in (0, 7):
        return 0.0
    diatonic = _MINOR_PCS if minor else _MAJOR_PCS
    return 0.3 if degree in diatonic else 0.8


def _chord_tension(
    pitch: int, chords: list[tuple[str, float, float]], t: float, key: str | None
) -> float:
    label = _segment_at(chords, t)
    if label is None:
        return _key_tension(pitch, key)
    parsed = parse_chord(label)
    if parsed is None:  # "N" (no chord) or unparseable
        return _key_tension(pitch, key)
    degree = (pitch - ROOT_PC[parsed.root]) % 12
    if degree not in chord_pcs(parsed) and degree in (3, 4, 9, 10, 11):
        # a color tone the chord itself doesn't contain reads more tense
        return max(0.7, _CHORD_TENSION[degree])
    return _CHORD_TENSION[degree]


def _annotate_phrases(track: TrackData, beat_len: float, boundaries: list[float]) -> None:
    """Phrase segmentation (spec §2): rest ≥ 1 beat, long-note-then-gap, or a
    section boundary. Sets phrase index/position/peak/final on every note."""
    notes = track.notes
    if not notes:
        return
    durations = [n.end - n.start for n in notes]

    def local_median(i: int) -> float:
        lo, hi = max(0, i - 8), min(len(notes), i + 9)
        return statistics.median(durations[lo:hi])

    phrase_ids = [0] * len(notes)
    current = 0
    max_end = notes[0].end
    for i in range(1, len(notes)):
        gap = notes[i].start - max_end
        crossed = any(notes[i - 1].start < b <= notes[i].start for b in boundaries)
        long_note = durations[i - 1] >= 2.0 * local_median(i - 1) and gap > 1e-3
        if gap >= beat_len or long_note or crossed:
            current += 1
        phrase_ids[i] = current
        max_end = max(max_end, notes[i].end)

    for phrase in range(current + 1):
        members = [i for i, p in enumerate(phrase_ids) if p == phrase]
        first, last = notes[members[0]].start, notes[members[-1]].start
        span = last - first
        peak = max(members, key=lambda i: notes[i].pitch)
        for i in members:
            note = notes[i]
            note.phrase = phrase
            note.phrase_pos = 0.5 if span <= 1e-9 else (note.start - first) / span
            note.phrase_peak = i == peak
            note.phrase_final = notes[i].start >= last - 1e-9
            note.legato_prev = False


def _annotate_intervals(track: TrackData) -> None:
    """Δp vs the previous top-voice note (highest concurrent pitch, like `1v`)."""
    notes = track.notes
    clusters: list[list[Note]] = []
    for note in notes:
        if clusters and note.start - clusters[-1][0].start <= CHORD_CLUSTER_S:
            clusters[-1].append(note)
        else:
            clusters.append([note])
    prev_top: Note | None = None
    for cluster in clusters:
        top = max(cluster, key=lambda n: n.pitch)
        for note in cluster:
            if prev_top is None or note.phrase != prev_top.phrase:
                note.interval = 0.0  # first note of a phrase
            else:
                note.interval = float(note.pitch - prev_top.pitch)
        prev_top = top


def annotate(tracks: list[TrackData], artifacts: Artifacts, ppq: int) -> None:
    """Compute all per-note features in place (Module A)."""
    grid = artifacts.grid()
    beat_len = 60.0 / grid.bpm if grid else 0.5
    boundaries = [s[1] for s in artifacts.sections or []]

    for track in tracks:
        _annotate_phrases(track, beat_len, boundaries)
        _annotate_intervals(track)
        for note in track.notes:
            if grid is not None:
                note.metric_weight = _metric_weight_grid(grid.pos(note.start), grid.felt_per_bar)
            else:  # PPQ fallback, assume 4/4
                note.metric_weight = _metric_weight_grid(note.start_tick / ppq, 4)

            if artifacts.chords:
                note.tension = _chord_tension(
                    note.pitch, artifacts.chords, note.start, artifacts.key
                )
            elif artifacts.key:
                note.tension = _key_tension(note.pitch, artifacts.key)
            else:
                note.tension = DEFAULT_TENSION

            if artifacts.sections:
                label = _segment_at(artifacts.sections, note.start)
                if label is not None:
                    note.section = label
                    note.section_intensity = SECTION_INTENSITY.get(
                        label.lower(), DEFAULT_SECTION_INTENSITY
                    )
