"""Shared helpers: synthetic MIDI files, artifacts, notes and tracks."""

from __future__ import annotations

import json
from pathlib import Path

import mido
import pytest

from sound2midi.expression.context import Note, TrackData

PPQ = 480
TEMPO = 500_000  # 120 BPM -> 1 beat = 0.5 s


def sec_to_tick(sec: float) -> int:
    return round(sec / 0.5 * PPQ)


def make_midi(
    path: Path,
    tracks: list[list[tuple[int, float, float]]],  # (pitch, start_s, duration_s)
    *,
    names: list[str] | None = None,
    velocity: int = 100,
) -> Path:
    mid = mido.MidiFile(type=1, ticks_per_beat=PPQ)
    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=TEMPO, time=0))
    mid.tracks.append(tempo_track)
    for i, notes in enumerate(tracks):
        track = mido.MidiTrack()
        if names and names[i]:
            track.append(mido.MetaMessage("track_name", name=names[i], time=0))
        events = []
        for pitch, start, duration in notes:
            events.append((sec_to_tick(start), "note_on", pitch))
            events.append((sec_to_tick(start + duration), "note_off", pitch))
        events.sort(key=lambda e: (e[0], e[1] == "note_on"))
        tick = 0
        for abs_tick, kind, pitch in events:
            track.append(
                mido.Message(
                    kind,
                    note=pitch,
                    velocity=velocity if kind == "note_on" else 0,
                    time=abs_tick - tick,
                )
            )
            tick = abs_tick
        mid.tracks.append(track)
    mid.save(str(path))
    return path


def make_artifacts(directory: Path, song: str, *, length_s: float = 30.0) -> Path:
    """Write a full artifact set on a 120 BPM 4/4 grid starting at 0.5 s."""
    directory.mkdir(parents=True, exist_ok=True)
    beats = [round(0.5 + 0.5 * i, 4) for i in range(int(length_s / 0.5))]
    (directory / f"{song}.meter.json").write_text(
        json.dumps(
            {
                "time_signature": "4/4",
                "numerator": 4,
                "denominator": 4,
                "felt_beats_per_bar": 4,
                "compound": False,
                "bpm": 120.0,
                "first_downbeat": 0.5,
                "beats": beats,
                "downbeats": beats[::4],
            }
        )
    )
    (directory / f"{song}.key.json").write_text(json.dumps({"key": "C Major"}))
    (directory / f"{song}.chords.json").write_text(
        json.dumps(
            {
                "chords": [
                    {"label": "C:maj", "start": 0.0, "end": length_s / 2},
                    {"label": "A:min7", "start": length_s / 2, "end": length_s},
                ]
            }
        )
    )
    (directory / f"{song}.sections.json").write_text(
        json.dumps(
            {
                "segments": [
                    {"label": "verse", "start": 0.0, "end": length_s / 2},
                    {"label": "chorus", "start": length_s / 2, "end": length_s},
                ]
            }
        )
    )
    return directory


def make_note(
    pitch: int,
    start: float,
    duration: float,
    *,
    track: int = 0,
    phrase: int = 0,
    **features,
) -> Note:
    note = Note(
        track=track,
        pitch=pitch,
        velocity=100,
        channel=0,
        start=start,
        end=start + duration,
        start_tick=sec_to_tick(start),
        perf_start=start,
        perf_end=start + duration,
        phrase=phrase,
    )
    for key, value in features.items():
        setattr(note, key, value)
    return note


def make_track(notes: list[Note], *, index: int = 0, name: str = "", is_drum: bool = False):
    return TrackData(index=index, name=name, program=0, is_drum=is_drum, notes=notes)


@pytest.fixture
def song_dir(tmp_path: Path) -> Path:
    """A song folder with a two-track MIDI and a full artifact set."""
    melody = [(60 + (i % 12), 0.5 + i * 0.5, 0.45) for i in range(40)]
    accompaniment = [(48 + (i % 5), 0.5 + i * 1.0, 0.9) for i in range(20)]
    make_midi(tmp_path / "song.mid", [melody, accompaniment], names=["vocals", "piano"])
    make_artifacts(tmp_path / "artifacts", "song")
    return tmp_path
