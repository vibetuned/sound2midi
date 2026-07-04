"""Export selected MIDI tracks to a notation staff (MusicXML, and ABC via xml2abc).

You assign instruments to staves explicitly (no pitch guessing): pass one list of track
indices per staff. One staff -> a single system; two staves -> a braced piano grand staff
(staff 1 = treble, staff 2 = bass). Each staff is built from its own reduced MIDI, parsed
with music21, quantized, and chordified into one clean voice. ABC is produced by running
the vendored ``xml2abc.py`` on the MusicXML.
"""

from __future__ import annotations

import bisect
import itertools
import math
import os
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mido
from music21 import chord, clef, converter, harmony, instrument, layout, note, pitch, stream
from music21 import key as m21key

from sound2midi.player import chordlabel

_CONDUCTOR_META = frozenset({"set_tempo", "time_signature", "key_signature"})

# Chromatic pitch-class -> name, for non-diatonic tones, per key accidental direction.
_SHARP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_FLAT_NAMES = ("C", "D-", "D", "E-", "E", "F", "G-", "G", "A-", "A", "B-", "B")


def _respelled(old: Any, name: str) -> Any:
    """A Pitch named ``name`` with the same sounding pitch (MIDI) as ``old``."""
    target = pitch.Pitch(name)
    target.octave = 4
    for _ in range(12):  # same pitch class, so this converges within an octave step or two
        if target.midi == old.midi:
            break
        target.octave += 1 if target.midi < old.midi else -1
    return target


def _respell_to_key(part: Any, ksig: Any) -> None:
    """Respell note/chord pitches to match ``ksig`` (diatonic spelling; flats in a flat
    key, sharps in a sharp key for chromatic tones). Sounding pitches are preserved."""
    diatonic: dict[int, str] = {}
    for scale_pitch in ksig.pitches:
        diatonic.setdefault(int(scale_pitch.pitchClass), str(scale_pitch.name))
    chromatic = _FLAT_NAMES if ksig.sharps < 0 else _SHARP_NAMES

    for element in part.recurse().notes:
        respelled = []
        changed = False
        for pch in element.pitches:
            name = diatonic.get(pch.pitchClass, chromatic[pch.pitchClass])
            if pch.name != name:
                respelled.append(_respelled(pch, name))
                changed = True
            else:
                respelled.append(pch)
        if not changed:
            continue
        if isinstance(element, note.Note):
            element.pitch = respelled[0]
        elif isinstance(element, chord.Chord):
            element.pitches = tuple(respelled)


def _key_signature(label: str) -> Any | None:
    """Turn a skey label like 'G# Major' / 'Bb minor' into a music21 Key.

    skey uses sharp spellings for every chromatic tonic, so some are theoretical keys
    with >7 sharps (e.g. G# major); those are re-spelled to their common enharmonic
    (G# major -> Ab major) so the key signature is writable.
    """
    parts = label.split()
    if len(parts) < 2:
        return None
    tonic_raw, mode = parts[0], parts[1].lower()
    if mode not in ("major", "minor"):
        return None
    tonic = tonic_raw[0].upper() + tonic_raw[1:].replace("b", "-")  # flats use '-' in music21
    try:
        ksig = m21key.Key(tonic, mode)
        if abs(ksig.sharps) > 7:
            ksig = m21key.Key(pitch.Pitch(tonic).getEnharmonic().name, mode)
    except Exception:
        return None
    return ksig


def _spelled_pc(pc: int, ksig: Any | None) -> str:
    """Spell a pitch class to the key (diatonic name, else the key's accidental
    direction), music21-style ('-' flats). Sharps when there is no key."""
    if ksig is not None:
        for scale_pitch in ksig.pitches:
            if int(scale_pitch.pitchClass) == pc:
                return str(scale_pitch.name)
        return (_FLAT_NAMES if ksig.sharps < 0 else _SHARP_NAMES)[pc]
    return _SHARP_NAMES[pc]


def _chord_symbol(label: str, ksig: Any | None) -> tuple[str, Any] | None:
    """A (figure-key, music21 ChordSymbol) for a Harte label; None for N/X."""
    parsed = chordlabel.parse(label)
    if parsed is None:
        return None
    root_name = _spelled_pc(chordlabel.ROOT_PC[parsed.root], ksig)
    kind = chordlabel.KIND_M21[chordlabel.base_kind(parsed.kind)]
    bass_name = None
    bpc = chordlabel.bass_pc(parsed)
    if bpc is not None and bpc != chordlabel.ROOT_PC[parsed.root]:
        bass_name = _spelled_pc(bpc, ksig)
    try:
        symbol = harmony.ChordSymbol(root=root_name, bass=bass_name, kind=kind)
    except Exception:
        return None
    symbol.writeAsChord = False  # annotation above the staff, not sounding notes
    return (f"{root_name}:{kind}/{bass_name}", symbol)


def _chord_symbol_events(
    chords: Sequence[dict],
    *,
    midi_path: Path,
    grid: _BeatGrid | None,
    beat_shift: float,
    windows: list[_Window] | None,
    key_label: str | None,
) -> list[tuple[float, Any]]:
    """Map chord segments to (quarterLength offset, ChordSymbol) score events.

    Offsets go through the same time mapping as the notes: snapped to the
    nearest felt beat on the beat grid (when there is one), then through the
    section windows. Same-offset collisions keep the later chord; consecutive
    repeats are merged.
    """
    ksig = _key_signature(key_label) if key_label else None
    sec_to_tick = None
    tpb = _EXPORT_TPB
    if grid is None:
        src = mido.MidiFile(str(midi_path))
        sec_to_tick = _sec_to_tick_fn(src)
        tpb = src.ticks_per_beat

    def map_tick(tick: int) -> int | None:
        if windows is None:
            return max(0, tick)
        for in0, in1, out0, _ in windows:
            if in0 <= tick < in1:
                return tick - in0 + out0
        return None

    by_offset: dict[float, tuple[str, Any]] = {}
    for seg in sorted(chords, key=lambda c: float(c["start"])):
        made = _chord_symbol(str(seg["label"]), ksig)
        if made is None:
            continue
        start = float(seg["start"])
        if grid is not None:
            beat = round(grid.pos(start))  # nearest felt beat
            tick = round((beat + beat_shift) * grid.beat_ql * _EXPORT_TPB)
        else:
            assert sec_to_tick is not None
            tick = sec_to_tick(start)
        mapped = map_tick(tick)
        if mapped is None:
            continue
        by_offset[mapped / tpb] = made  # later chord wins a same-beat collision

    events: list[tuple[float, Any]] = []
    previous = None
    for offset in sorted(by_offset):
        figure_key, symbol = by_offset[offset]
        if figure_key == previous:
            continue
        previous = figure_key
        events.append((offset, symbol))
    return events


def _insert_chord_symbols(part: Any, events: list[tuple[float, Any]]) -> None:
    """Insert ChordSymbols into the measures containing their offsets."""
    measures = list(part.getElementsByClass(stream.Measure))
    if not measures:
        for offset, symbol in events:
            part.insert(offset, symbol)
        return
    offsets = [m.offset for m in measures]
    span_end = measures[-1].offset + measures[-1].barDuration.quarterLength
    for offset, symbol in events:
        if offset >= span_end:
            continue
        i = max(0, bisect.bisect_right(offsets, offset + 1e-6) - 1)
        measure = measures[i]
        measure.insert(max(0.0, offset - measure.offset), symbol)


def find_xml2abc() -> Path | None:
    """Locate the vendored xml2abc.py (MusicXML -> ABC)."""
    env = os.environ.get("SOUND2MIDI_XML2ABC")
    candidates = [
        Path(env) if env else None,
        Path.cwd() / "vendor" / "xml2abc.py",
        Path(__file__).resolve().parents[3] / "vendor" / "xml2abc.py",
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate
    return None


def _extract_notes(track: Any) -> list[list]:
    """Note intervals ``[start_tick, end_tick, note, velocity, channel]`` of a track."""
    notes: list[list] = []
    active: dict[tuple[int, int], list] = {}
    tick = 0
    for msg in track:
        tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            record = [tick, None, msg.note, msg.velocity, msg.channel]
            active[(msg.channel, msg.note)] = record
            notes.append(record)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            record = active.pop((msg.channel, msg.note), None)
            if record is not None:
                record[1] = tick
    for record in active.values():  # close notes still sounding at track end
        record[1] = tick
    return [n for n in notes if n[1] is not None and n[1] > n[0]]


def _snap_ragged_onsets(notes: list[list], *, window_ticks: int) -> None:
    """Align chord attacks: a note starting shortly after a still-sounding note is
    snapped back to that note's onset (in place). Fixes the A / [A C] half-sliver where
    a chord-mate's transcribed attack lags the rest of the chord.

    ``window_ticks`` must stay below one quantize-grid slot, otherwise genuinely
    sequential fast notes (one slot apart) would be merged into fake block chords.
    """
    notes.sort(key=lambda n: n[0])
    anchor: int | None = None
    cluster_end = 0
    for record in notes:
        if anchor is None or record[0] - anchor > window_ticks or record[0] >= cluster_end:
            anchor, cluster_end = record[0], record[1]
        else:
            record[0] = anchor
            cluster_end = max(cluster_end, record[1])


def _snap_ragged_offsets(notes: list[list], *, window_ticks: int) -> None:
    """Align chord releases: co-sounding notes whose ends fall within the window are
    trimmed to the cluster's earliest end (in place) — but only if no new note starts
    inside the ragged span (then the ring is dangling into silence, not harmony).
    Fixes the [A C] / C half-sliver where one chord-mate rings a little longer."""
    onsets = sorted(n[0] for n in notes)
    ordered = sorted(notes, key=lambda n: n[1])
    cluster: list[list] = []

    def flush() -> None:
        if len(cluster) < 2:
            return
        lo = cluster[0][1]
        hi = cluster[-1][1]
        i = bisect.bisect_right(onsets, lo)
        if i < len(onsets) and onsets[i] < hi:
            return  # a note attacks inside the span; leave it to the tail-trim pass
        for n in cluster:
            n[1] = lo

    for record in ordered:
        if cluster and record[1] - cluster[0][1] <= window_ticks and record[0] < cluster[0][1]:
            cluster.append(record)
        else:
            flush()
            cluster = [record]
    flush()


def _trim_legato_tails(notes: list[list], *, sync_ticks: int, max_overlap_ticks: int) -> None:
    """Cut notes that ring slightly past the next onset, in place.

    Transcribed releases often overlap the next note by a few tens of ms; chordify
    slices a spurious chord out of every such overlap (A -> C legato becomes
    A / [A C] / C). A note overlapping the next onset by at most ``max_overlap_ticks``
    is trimmed to end exactly at that onset. Onsets within ``sync_ticks`` count as the
    same chord attack (never trim against each other), and overlaps longer than the
    threshold are kept — those are genuine suspensions/held harmony.
    """
    onsets = sorted(n[0] for n in notes)
    for record in notes:
        i = bisect.bisect_right(onsets, record[0] + sync_ticks)
        if i < len(onsets):
            next_onset = onsets[i]
            if next_onset < record[1] and (record[1] - next_onset) <= max_overlap_ticks:
                record[1] = next_onset


def _clean_overlaps(notes: list[list], *, sync_ticks: int, max_overlap_ticks: int) -> list[list]:
    """Full overlap cleanup: align ragged attacks, cut legato tails, align ragged
    releases. Together these remove all three chordify sliver shapes (A/[AC],
    A/[AC]/C, [AC]/C) while preserving genuine chords and long suspensions.

    The onset-snap window is half the tail threshold (0.75 of a grid slot) so genuinely
    sequential fast notes — one slot apart — are never merged into fake chords. The
    offset snap runs after tail-trimming (which resolves all small sequential overlaps),
    so remaining co-sounding notes are chord-mates and it can use the full threshold.
    """
    _snap_ragged_onsets(notes, window_ticks=max(sync_ticks, max_overlap_ticks // 2))
    _trim_legato_tails(notes, sync_ticks=sync_ticks, max_overlap_ticks=max_overlap_ticks)
    _snap_ragged_offsets(notes, window_ticks=max_overlap_ticks)
    # snapping can create exact duplicates; keep one per (start, end, pitch, channel)
    unique: dict[tuple[int, int, int, int], list] = {}
    for record in notes:
        key = (record[0], record[1], record[2], record[4])
        kept = unique.get(key)
        if kept is None or record[3] > kept[3]:
            unique[key] = record
    return [n for n in unique.values() if n[1] > n[0]]


# A synthesized (non-MIDI-file) track: (start_sec, end_sec, midi_notes) chord
# events, realized from a chords artifact. Injected into the staff builders under
# a track index beyond the file's own tracks.
SynthTrack = list[tuple[float, float, tuple[int, ...]]]


def _synth_notes(events: SynthTrack, sec_to_out_tick: Any) -> list[list]:
    """Flatten synth chord events into note records in the builder's tick space."""
    notes: list[list] = []
    for start, end, pitches in events:
        t0 = sec_to_out_tick(start)
        t1 = max(t0 + 1, sec_to_out_tick(end))
        notes.extend([t0, t1, note_num, 70, 0] for note_num in pitches)
    return notes


# A section window in some tick space: (in_start, in_end, out_start, allowed_tracks).
# Notes whose onset falls in [in_start, in_end) are shifted to out_start + (t - in_start)
# (ends clipped to the window); ``allowed_tracks=None`` admits every track.
_Window = tuple[int, int, int, "frozenset[int] | None"]


def _append_window(
    windows: list[_Window], in0: int, in1: int, offset: int, tracks: frozenset | None
) -> int:
    """Add a window (merging into the previous one when seamless); return new offset."""
    if windows and windows[-1][1] == in0 and windows[-1][3] == tracks:
        prev = windows[-1]
        windows[-1] = (prev[0], in1, prev[2], tracks)
    else:
        windows.append((in0, in1, offset, tracks))
    return offset + (in1 - in0)


def _slice_notes_to_windows(
    notes: list[list], windows: list[_Window], track_index: int
) -> list[list]:
    """Keep notes whose onset falls in a window admitting this track; concatenate."""
    out: list[list] = []
    for record in notes:
        start, end = record[0], record[1]
        for in0, in1, out0, allowed in windows:
            if not in0 <= start < in1:
                continue
            if allowed is None or track_index in allowed:
                new_start = start - in0 + out0
                new_end = min(end, in1) - in0 + out0
                if new_end > new_start:
                    out.append([new_start, new_end, record[2], record[3], record[4]])
            break  # windows don't overlap; the onset lives in exactly one
    return out


def _notes_to_track(notes: list[list]) -> Any:
    """Serialize note intervals back into a delta-time mido track."""
    events: list[tuple[int, int, Any]] = []
    for start, end, note_num, velocity, channel in notes:
        events.append(
            (start, 1, mido.Message("note_on", note=note_num, velocity=velocity, channel=channel))
        )
        events.append(
            (end, 0, mido.Message("note_off", note=note_num, velocity=0, channel=channel))
        )
    events.sort(key=lambda e: (e[0], e[1]))  # note_offs first at equal ticks

    track = mido.MidiTrack()
    last = 0
    for tick, _, msg in events:
        msg.time = tick - last
        last = tick
        track.append(msg)
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


def _enforce_monophony(notes: list[list], *, min_fragment_ticks: int) -> list[list]:
    """Rebuild a track as a single voice: at any instant only the most recent attack
    sounds (ties broken by higher pitch), and a held note *resumes* after an inner note
    ends. This reconstructs melodies from transcription artifacts like a C4 held right
    across a Bb3 (chordified as C4 / [Bb3 C4] / C4 -> becomes C4, Bb3, C4). Fragments
    shorter than ``min_fragment_ticks`` are dropped. Chords are collapsed to their top
    note, so this is only for tracks the user marks as monophonic."""
    if not notes:
        return notes
    points = sorted({n[0] for n in notes} | {n[1] for n in notes})
    segments: list[list] = []  # [start, end, source_note_index]
    for a, b in itertools.pairwise(points):
        sounding = [i for i, n in enumerate(notes) if n[0] <= a and n[1] >= b]
        if not sounding:
            continue
        winner = max(sounding, key=lambda i: (notes[i][0], notes[i][2]))
        if segments and segments[-1][2] == winner and segments[-1][1] == a:
            segments[-1][1] = b  # extend a resumed/continuing fragment
        else:
            segments.append([a, b, winner])
    out: list[list] = []
    for start, end, i in segments:
        if end - start < min_fragment_ticks:
            continue
        source = notes[i]
        out.append([start, end, source[2], source[3], source[4]])
    return out


def _snap_to_grid(notes: list[list], *, divisors: Sequence[int], tpb: int) -> list[list]:
    """Snap note onsets AND offsets to the quantize grid, in tick space.

    music21's quantizer rounds onset and *duration* independently, which can push a
    note's end past the next note's rounded start — recreating exactly the overlap
    chords the cleanup passes removed. Rounding both boundaries with the same monotonic
    function can never create an overlap. Notes shorter than half a grid slot collapse
    and are dropped (as any quantizer would).
    """

    def snap(tick: int) -> int:
        best = tick
        best_err = None
        for d in divisors:
            step = tpb / d
            candidate = round(round(tick / step) * step)
            err = abs(candidate - tick)
            if best_err is None or err < best_err:
                best, best_err = candidate, err
        return best

    out: list[list] = []
    for record in notes:
        start, end = snap(record[0]), snap(record[1])
        if end > start:
            out.append([start, end, record[2], record[3], record[4]])
    return out


def _overlap_threshold_ql(quantize_divisors: Sequence[int] | None) -> float:
    """Max legato overlap (in quarter lengths) to trim: 1.5x the finest grid unit.

    After quantization, overlaps between ~0.5 and ~1.5 grid units survive as one-slot
    sliver chords — exactly the artifact we want gone. Longer overlaps span 2+ slots
    and are treated as genuine harmony.
    """
    unit = 1.0 / max(quantize_divisors) if quantize_divisors else 0.25
    return 1.5 * unit


def _build_reduced_midi(
    src: Path,
    track_indices: Sequence[int],
    dest: Path,
    *,
    trim_overlap_ql: float | None = None,
    mono_tracks: frozenset[int] = frozenset(),
    mono_min_fragment_ql: float = 0.125,
    quantize_divisors: Sequence[int] | None = None,
    windows: list[_Window] | None = None,
    synth_tracks: dict[int, SynthTrack] | None = None,
) -> None:
    """Write a MIDI containing only the given tracks plus a tempo/meta conductor.

    ``windows`` (original tick space) restricts the output to those spans,
    concatenated; conductor state at the first window's start is carried to 0.
    ``synth_tracks`` maps out-of-file track indices to realized chord events.
    """
    orig = mido.MidiFile(str(src))
    reduced = mido.MidiFile(type=1, ticks_per_beat=orig.ticks_per_beat)

    conductor_events: list[tuple[int, Any]] = []
    for track in orig.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.is_meta and msg.type in _CONDUCTOR_META:
                conductor_events.append((tick, msg.copy()))
    conductor_events.sort(key=lambda e: e[0])

    if windows is not None:
        state: dict[str, Any] = {}
        sliced: list[tuple[int, Any]] = []
        for tick, msg in conductor_events:
            mapped = None
            for in0, in1, out0, _ in windows:
                if in0 <= tick < in1:
                    mapped = tick - in0 + out0
                    break
            if mapped is not None:
                sliced.append((mapped, msg))
            elif tick <= windows[0][0]:
                state[msg.type] = msg  # last state before the first window opens at 0
        conductor_events = sorted([(0, msg) for msg in state.values()] + sliced, key=lambda e: e[0])

    conductor = mido.MidiTrack()
    last = 0
    for abs_tick, msg in conductor_events:
        msg.time = abs_tick - last
        last = abs_tick
        conductor.append(msg)
    conductor.append(mido.MetaMessage("end_of_track", time=0))
    reduced.tracks.append(conductor)

    tpb = orig.ticks_per_beat
    sec_to_tick = None
    for index in track_indices:
        if synth_tracks is not None and index in synth_tracks:
            if sec_to_tick is None:
                sec_to_tick = _sec_to_tick_fn(orig)
            notes = _synth_notes(synth_tracks[index], sec_to_tick)
        elif 0 <= index < len(orig.tracks):
            notes = _extract_notes(orig.tracks[index])
        else:
            continue
        if windows is not None:
            notes = _slice_notes_to_windows(notes, windows, index)
        if trim_overlap_ql:
            notes = _clean_overlaps(
                notes,
                sync_ticks=max(1, tpb // 16),
                max_overlap_ticks=round(trim_overlap_ql * tpb),
            )
        if index in mono_tracks:
            notes = _enforce_monophony(
                notes, min_fragment_ticks=max(1, round(mono_min_fragment_ql * tpb))
            )
        if quantize_divisors:
            notes = _snap_to_grid(notes, divisors=quantize_divisors, tpb=tpb)
        reduced.tracks.append(_notes_to_track(notes))

    reduced.save(str(dest))


def _tick_to_sec_fn(mid: Any) -> Any:
    """A tick -> absolute-seconds function honoring the file's tempo map."""
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


def _sec_to_tick_fn(mid: Any) -> Any:
    """The inverse of :func:`_tick_to_sec_fn` (the tempo map is monotonic)."""
    tick_to_sec = _tick_to_sec_fn(mid)
    changes: list[tuple[int, int]] = []
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.is_meta and msg.type == "set_tempo":
                changes.append((tick, msg.tempo))
    changes.sort(key=lambda c: c[0])
    if not changes or changes[0][0] != 0:
        changes.insert(0, (0, 500000))
    segments = [(tick, tick_to_sec(tick), tempo) for tick, tempo in changes]
    start_secs = [s[1] for s in segments]

    def sec_to_tick(sec: float) -> int:
        i = max(0, bisect.bisect_right(start_secs, sec) - 1)
        stick, ssec, tempo = segments[i]
        return stick + round(mido.second2tick(max(0.0, sec - ssec), mid.ticks_per_beat, tempo))

    return sec_to_tick


class _BeatGrid:
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


def _min_beat_position(src: Any, track_indices: Sequence[int], grid: _BeatGrid) -> float:
    """Earliest felt-beat position of any selected note (for a shared pickup shift)."""
    tick_to_sec = _tick_to_sec_fn(src)
    minimum = math.inf
    for index in track_indices:
        if not (0 <= index < len(src.tracks)):
            continue
        tick = 0
        for msg in src.tracks[index]:
            tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                minimum = min(minimum, grid.pos(tick_to_sec(tick)))
                break  # tracks are time-ordered; first note is the earliest
    return 0.0 if minimum is math.inf else minimum


_EXPORT_TPB = 480


def _build_beat_aligned_midi(
    src: Any,
    track_indices: Sequence[int],
    dest: Path,
    *,
    meter: dict,
    grid: _BeatGrid,
    beat_shift: float,
    trim_overlap_ql: float | None = None,
    mono_tracks: frozenset[int] = frozenset(),
    mono_min_fragment_ql: float = 0.125,
    quantize_divisors: Sequence[int] | None = None,
    windows: list[_Window] | None = None,
    synth_tracks: dict[int, SynthTrack] | None = None,
) -> None:
    """Write a reduced MIDI retimed to the beat grid, with real tempo + time signature.

    Note times are mapped from seconds to felt-beat positions (piecewise-linear between
    tracked beats), so a quarter note in the output is an actual beat of the music and
    bar 1 starts on the first downbeat. ``beat_shift`` (whole bars, in felt beats) makes
    room for pickup notes and must be identical across staves to keep them aligned.

    ``windows`` (retimed-tick space, bar-aligned) restricts the output to those
    spans, concatenated; the retimed ticks are left unclamped so that windows
    starting before the first downbeat still slice correctly.
    """
    tick_to_sec = _tick_to_sec_fn(src)
    out = mido.MidiFile(type=1, ticks_per_beat=_EXPORT_TPB)

    conductor = mido.MidiTrack()
    conductor.append(
        mido.MetaMessage(
            "time_signature",
            numerator=int(meter.get("numerator", 4)),
            denominator=int(meter.get("denominator", 4)),
            time=0,
        )
    )
    conductor.append(
        mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(grid.bpm * grid.beat_ql), time=0)
    )
    conductor.append(mido.MetaMessage("end_of_track", time=0))
    out.tracks.append(conductor)

    def new_tick_sec(seconds: float) -> int:
        beat_pos = grid.pos(seconds) + beat_shift
        tick = round(beat_pos * grid.beat_ql * _EXPORT_TPB)
        return tick if windows is not None else max(0, tick)

    def new_tick(abs_tick: int) -> int:
        return new_tick_sec(tick_to_sec(abs_tick))

    for index in track_indices:
        if synth_tracks is not None and index in synth_tracks:
            notes = _synth_notes(synth_tracks[index], new_tick_sec)
        elif 0 <= index < len(src.tracks):
            notes = _extract_notes(src.tracks[index])
            for record in notes:
                record[0] = new_tick(record[0])
                record[1] = max(record[0] + 1, new_tick(record[1]))
        else:
            continue
        if windows is not None:
            notes = _slice_notes_to_windows(notes, windows, index)
        if trim_overlap_ql:
            notes = _clean_overlaps(
                notes,
                sync_ticks=max(1, _EXPORT_TPB // 16),
                max_overlap_ticks=round(trim_overlap_ql * _EXPORT_TPB),
            )
        if index in mono_tracks:
            notes = _enforce_monophony(
                notes, min_fragment_ticks=max(1, round(mono_min_fragment_ql * _EXPORT_TPB))
            )
        if quantize_divisors:
            notes = _snap_to_grid(notes, divisors=quantize_divisors, tpb=_EXPORT_TPB)
        out.tracks.append(_notes_to_track(notes))

    out.save(str(dest))


def _staff_part(
    midi_path: Path,
    track_indices: Sequence[int],
    *,
    quantize_divisors: Sequence[int] | None,
    treble: bool,
    key_label: str | None = None,
    meter: dict | None = None,
    grid: _BeatGrid | None = None,
    beat_shift: float = 0.0,
    trim_overlap_ql: float | None = None,
    mono_tracks: frozenset[int] = frozenset(),
    mono_min_fragment_ql: float = 0.125,
    windows: list[_Window] | None = None,
    synth_tracks: dict[int, SynthTrack] | None = None,
) -> Any:
    """Build one chordified staff (a music21 Part) from the given tracks."""
    # music21's MIDI parser quantizes on import, so the grid must be set at parse time
    # (a post-parse quantize() is a no-op against the parser's own snapping).
    with tempfile.TemporaryDirectory() as tmp:
        reduced_midi = Path(tmp) / "reduced.mid"
        if meter is not None and grid is not None:
            src = mido.MidiFile(str(midi_path))
            _build_beat_aligned_midi(
                src,
                track_indices,
                reduced_midi,
                meter=meter,
                grid=grid,
                beat_shift=beat_shift,
                trim_overlap_ql=trim_overlap_ql,
                mono_tracks=mono_tracks,
                mono_min_fragment_ql=mono_min_fragment_ql,
                quantize_divisors=quantize_divisors,
                windows=windows,
                synth_tracks=synth_tracks,
            )
        else:
            _build_reduced_midi(
                midi_path,
                track_indices,
                reduced_midi,
                trim_overlap_ql=trim_overlap_ql,
                mono_tracks=mono_tracks,
                mono_min_fragment_ql=mono_min_fragment_ql,
                quantize_divisors=quantize_divisors,
                windows=windows,
                synth_tracks=synth_tracks,
            )
        if quantize_divisors:
            # already snapped to the grid in tick space (see _snap_to_grid); music21's
            # own quantizer would round onsets/durations independently and could
            # recreate the overlap chords the cleanup passes removed.
            score = converter.parse(str(reduced_midi), quantizePost=False)
        else:
            score = converter.parse(str(reduced_midi))
    part = score.chordify()

    # Drop MIDI/instrument metadata (programs, channels) that music21 carries over from the
    # source tracks — otherwise their program changes show up as "instrument change" clutter
    # all over the MusicXML. We only want pitch + rhythm, on one clean instrument.
    for inst in list(part.recurse().getElementsByClass(instrument.Instrument)):
        part.remove(inst, recurse=True)
    part.insert(0, instrument.Piano())

    # Set the clef inside the first measure; a part-level clef is overridden by the
    # default (treble) clef that chordify writes into measure 1.
    desired = clef.TrebleClef() if treble else clef.BassClef()
    first = part.getElementsByClass(stream.Measure).first()
    target = first if first is not None else part
    for existing in list(target.getElementsByClass(clef.Clef)):
        target.remove(existing)
    target.insert(0, desired)

    if key_label:
        ksig = _key_signature(key_label)
        if ksig is not None:
            for existing in list(target.getElementsByClass(m21key.KeySignature)):
                target.remove(existing)
            target.insert(0, ksig)
            _respell_to_key(part, ksig)  # spell notes to the key, not just show the signature
    return part


def _pad_to_equal_length(parts: Sequence[Any]) -> None:
    """Append full-bar rests to the shorter staves so all staves span the same measures.

    Each staff is parsed from its own tracks, so a short instrument yields fewer measures;
    a grand staff must have both hands aligned. All parts share the tempo/time signature
    (same conductor), so equal measure counts means equal duration.
    """
    measures = [list(part.getElementsByClass(stream.Measure)) for part in parts]
    target = max((len(m) for m in measures), default=0)
    for part, part_measures in zip(parts, measures, strict=False):
        if len(part_measures) >= target or not part_measures:
            continue
        bar_ql = part_measures[-1].barDuration.quarterLength
        last_number = part_measures[-1].number or len(part_measures)
        for offset in range(target - len(part_measures)):
            measure = stream.Measure(number=last_number + 1 + offset)
            measure.append(note.Rest(quarterLength=bar_ql))
            part.append(measure)


def export_to_staff(
    midi_path: str | Path,
    staves: Sequence[Sequence[int]],
    out_dir: str | Path,
    *,
    basename: str | None = None,
    formats: Sequence[str] = ("musicxml", "abc"),
    quantize_divisors: Sequence[int] | None = (4,),
    key: str | None = None,
    meter: dict | None = None,
    trim_overlaps: bool = True,
    mono_tracks: frozenset[int] | set[int] = frozenset(),
    title: str | None = None,
    sections: Sequence[dict] | None = None,
    chords: Sequence[dict] | None = None,
    synth_tracks: dict[int, SynthTrack] | None = None,
) -> dict[str, Path]:
    """Export the chosen tracks of ``midi_path`` to notation files in ``out_dir``.

    ``staves`` is one list of track indices per staff. One non-empty staff -> a single
    system; two -> a braced grand staff (staff 1 treble, staff 2 bass).

    ``quantize_divisors`` sets the notation grid as music21 quarter-length divisors,
    e.g. ``(4,)`` = 16th-note grid (clean, no tuplets), ``(4, 3)`` = 16ths + triplets,
    ``None`` = keep raw timing.

    ``meter`` is a detected-meter artifact (from ``<song>.meter.json``). When it carries
    a beat grid, notes are retimed to it: the output gets the real tempo and time
    signature, and bar 1 is anchored on the first downbeat.

    ``sections`` cuts the export to song sections: a time-ordered list of
    ``{"start": seconds, "end": seconds, "tracks": iterable[int] | None}``. Each
    window is snapped to whole bars of the beat grid (when ``meter`` is applied)
    and the windows are concatenated; a window's ``tracks`` restricts which of the
    staff's tracks sound in it (``None`` = all of them).

    ``chords`` writes chord symbols above the top staff: a list of
    ``{"label": "C:maj", "start": seconds, "end": seconds}`` segments (from
    ``<song>.chords.json``), snapped to the beat grid and respelled to ``key``.

    ``synth_tracks`` injects synthesized tracks (realized chord voicings, times
    in seconds) under out-of-file track indices, so they can be assigned to
    staves like any instrument.

    Returns a mapping of produced format -> path (and "abc_error" if ABC could not
    be produced).
    """
    midi_path = Path(midi_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    staves = [list(s) for s in staves if s]  # drop empty staves
    if not staves:
        raise ValueError("No instruments assigned to any staff.")

    mode = "grand" if len(staves) >= 2 else "single"
    basename = basename or f"{midi_path.stem}.{mode}"

    grid: _BeatGrid | None = None
    beat_shift = 0.0
    if meter is not None and len(meter.get("beats") or []) >= 8:
        grid = _BeatGrid(meter)
        # one shift for ALL staves so they stay bar-aligned; whole bars only
        src = mido.MidiFile(str(midi_path))
        all_tracks = [t for staff in staves for t in staff]
        min_pos = _min_beat_position(src, all_tracks, grid)
        if min_pos < 0:
            bars = math.ceil(-min_pos / grid.felt_per_bar)
            beat_shift = bars * grid.felt_per_bar
    else:
        meter = None  # no usable beat grid -> plain export

    windows: list[_Window] | None = None
    if sections:
        ordered = sorted(
            (s for s in sections if float(s["end"]) > float(s["start"])),
            key=lambda s: float(s["start"]),
        )
        windows = []
        offset = 0
        if grid is not None:
            # Section boundaries are approximate; snap them to whole bars so the
            # cuts are clean and every window starts on a downbeat. Windows are
            # bar-anchored themselves, so the global pickup shift is not needed.
            beat_shift = 0.0
            bar = grid.felt_per_bar

            def beat_tick(beats: float) -> int:
                return round(beats * grid.beat_ql * _EXPORT_TPB)

            for s in ordered:
                b0 = round(grid.pos(float(s["start"])) / bar) * bar
                b1 = round(grid.pos(float(s["end"])) / bar) * bar
                if b1 <= b0:
                    continue
                tracks = frozenset(s["tracks"]) if s.get("tracks") is not None else None
                offset = _append_window(windows, beat_tick(b0), beat_tick(b1), offset, tracks)
        else:
            sec_to_tick = _sec_to_tick_fn(mido.MidiFile(str(midi_path)))
            for s in ordered:
                t0, t1 = sec_to_tick(float(s["start"])), sec_to_tick(float(s["end"]))
                if t1 <= t0:
                    continue
                tracks = frozenset(s["tracks"]) if s.get("tracks") is not None else None
                offset = _append_window(windows, t0, t1, offset, tracks)
        if not windows:
            raise ValueError("The selected sections are shorter than one bar; nothing to export.")

    trim_overlap_ql = _overlap_threshold_ql(quantize_divisors) if trim_overlaps else None
    grid_unit = 1.0 / max(quantize_divisors) if quantize_divisors else 0.25

    parts = [
        _staff_part(
            midi_path,
            tracks,
            quantize_divisors=quantize_divisors,
            treble=(i == 0),
            key_label=key,
            meter=meter,
            grid=grid,
            beat_shift=beat_shift,
            trim_overlap_ql=trim_overlap_ql,
            mono_tracks=frozenset(mono_tracks),
            mono_min_fragment_ql=grid_unit / 2,
            windows=windows,
            synth_tracks=synth_tracks,
        )
        for i, tracks in enumerate(staves)
    ]

    if len(parts) >= 2:
        _pad_to_equal_length(parts)

    if chords:
        events = _chord_symbol_events(
            chords,
            midi_path=midi_path,
            grid=grid,
            beat_shift=beat_shift,
            windows=windows,
            key_label=key,
        )
        _insert_chord_symbols(parts[0], events)

    score = stream.Score()
    for part in parts:
        score.insert(0, part)
    if len(parts) >= 2:
        group = layout.StaffGroup(parts, name="Piano", abbreviation="Pno.", symbol="brace")
        group.barTogether = True
        score.insert(0, group)
    if title:
        score.insert(0, _metadata(title))

    results: dict[str, Path] = {}
    musicxml_path = out_dir / f"{basename}.musicxml"
    score.write("musicxml", fp=str(musicxml_path))
    if "musicxml" in formats:
        results["musicxml"] = musicxml_path

    if "abc" in formats:
        abc_path = _musicxml_to_abc(musicxml_path, out_dir)
        if abc_path is not None:
            results["abc"] = abc_path
        else:
            results["abc_error"] = Path(  # type: ignore[assignment]
                "xml2abc.py not found (set $SOUND2MIDI_XML2ABC or keep vendor/xml2abc.py)"
            )

    if "musicxml" not in formats:
        musicxml_path.unlink(missing_ok=True)  # was only a stepping stone to ABC

    return results


def _metadata(title: str) -> Any:
    from music21 import metadata

    md = metadata.Metadata()
    md.title = title
    md.composer = "sound2midi"
    return md


def _musicxml_to_abc(musicxml_path: Path, out_dir: Path) -> Path | None:
    xml2abc = find_xml2abc()
    if xml2abc is None:
        return None
    subprocess.run(
        [sys.executable, str(xml2abc), "-o", str(out_dir), str(musicxml_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    abc_path = out_dir / f"{musicxml_path.stem}.abc"
    return abc_path if abc_path.exists() else None
