"""Export selected MIDI tracks to a notation staff (MusicXML, and ABC via xml2abc).

You assign instruments to staves explicitly (no pitch guessing): pass one list of track
indices per staff. One staff -> a single system; two staves -> a braced piano grand staff
(staff 1 = treble, staff 2 = bass). Each staff is built from its own reduced MIDI, parsed
with music21, quantized, and chordified into one clean voice. ABC is produced by running
the vendored ``xml2abc.py`` on the MusicXML.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mido
from music21 import chord, clef, converter, instrument, layout, note, pitch, stream
from music21 import key as m21key

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


def _build_reduced_midi(src: Path, track_indices: Sequence[int], dest: Path) -> None:
    """Write a MIDI containing only the given tracks plus a tempo/meta conductor."""
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

    conductor = mido.MidiTrack()
    last = 0
    for abs_tick, msg in conductor_events:
        msg.time = abs_tick - last
        last = abs_tick
        conductor.append(msg)
    conductor.append(mido.MetaMessage("end_of_track", time=0))
    reduced.tracks.append(conductor)

    for index in track_indices:
        if 0 <= index < len(orig.tracks):
            reduced.tracks.append(orig.tracks[index])

    reduced.save(str(dest))


def _staff_part(
    midi_path: Path,
    track_indices: Sequence[int],
    *,
    quantize_divisors: Sequence[int] | None,
    treble: bool,
    key_label: str | None = None,
) -> Any:
    """Build one chordified staff (a music21 Part) from the given tracks."""
    # music21's MIDI parser quantizes on import, so the grid must be set at parse time
    # (a post-parse quantize() is a no-op against the parser's own snapping).
    with tempfile.TemporaryDirectory() as tmp:
        reduced_midi = Path(tmp) / "reduced.mid"
        _build_reduced_midi(midi_path, track_indices, reduced_midi)
        if quantize_divisors:
            score = converter.parse(
                str(reduced_midi),
                quantizePost=True,
                quarterLengthDivisors=tuple(quantize_divisors),
            )
        else:
            score = converter.parse(str(reduced_midi), quantizePost=False)
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
    title: str | None = None,
) -> dict[str, Path]:
    """Export the chosen tracks of ``midi_path`` to notation files in ``out_dir``.

    ``staves`` is one list of track indices per staff. One non-empty staff -> a single
    system; two -> a braced grand staff (staff 1 treble, staff 2 bass).

    ``quantize_divisors`` sets the notation grid as music21 quarter-length divisors,
    e.g. ``(4,)`` = 16th-note grid (clean, no tuplets), ``(4, 3)`` = 16ths + triplets,
    ``None`` = keep raw timing. Returns a mapping of produced format -> path (and
    "abc_error" if ABC could not be produced).
    """
    midi_path = Path(midi_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    staves = [list(s) for s in staves if s]  # drop empty staves
    if not staves:
        raise ValueError("No instruments assigned to any staff.")

    mode = "grand" if len(staves) >= 2 else "single"
    basename = basename or f"{midi_path.stem}.{mode}"

    parts = [
        _staff_part(
            midi_path,
            tracks,
            quantize_divisors=quantize_divisors,
            treble=(i == 0),
            key_label=key,
        )
        for i, tracks in enumerate(staves)
    ]

    if len(parts) >= 2:
        _pad_to_equal_length(parts)

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
