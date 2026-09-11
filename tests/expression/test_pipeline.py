"""Integration tests: artifacts discovery, loudness, CLI render (criteria 1, 2, 5, 8)."""

from __future__ import annotations

from itertools import pairwise

import mido
import numpy as np
import soundfile as sf
from conftest import make_midi

from sound2midi.expression import loudness
from sound2midi.expression.cli import render_main
from sound2midi.expression.context import (
    annotate,
    extract_tracks,
    load_artifacts,
    song_name,
)


def test_song_name_strips_render_suffixes(tmp_path):
    assert song_name(tmp_path / "abc.mid") == "abc"
    assert song_name(tmp_path / "abc.stems.mid") == "abc"
    assert song_name(tmp_path / "abc.stems.mpe.mid") == "abc"


def test_render_with_artifacts(song_dir, capsys):
    assert render_main([str(song_dir / "song.mid"), "--seed", "7"]) == 0
    dest = song_dir / "song.mpe.mid"
    assert dest.is_file()
    mid = mido.MidiFile(str(dest))
    assert mid.type == 1 and mid.ticks_per_beat == 480
    messages = [m for track in mid.tracks for m in track]
    types = {m.type for m in messages}
    assert {"note_on", "note_off", "aftertouch", "pitchwheel", "control_change"} <= types
    # tempo map preserved
    assert any(m.is_meta and m.type == "set_tempo" and m.tempo == 500_000 for m in messages)
    # member channels only (no notes on the master channel 0)
    assert all(m.channel >= 1 for m in messages if m.type == "note_on")


def test_render_without_artifacts(tmp_path, capsys):
    make_midi(tmp_path / "bare.mid", [[(60 + i % 5, i * 0.5, 0.45) for i in range(16)]])
    assert render_main([str(tmp_path / "bare.mid"), "--seed", "1"]) == 0
    assert (tmp_path / "bare.mpe.mid").is_file()
    assert "Artifacts missing" in capsys.readouterr().err


def test_fixed_seed_is_byte_identical(song_dir):
    out1 = song_dir / "one.mid"
    out2 = song_dir / "two.mid"
    source = str(song_dir / "song.mid")
    assert render_main([source, "--seed", "7", "-o", str(out1)]) == 0
    assert render_main([source, "--seed", "7", "-o", str(out2)]) == 0
    assert out1.read_bytes() == out2.read_bytes()
    out3 = song_dir / "three.mid"
    assert render_main([source, "--seed", "8", "-o", str(out3)]) == 0
    assert out1.read_bytes() != out3.read_bytes()


def test_keep_velocity(song_dir):
    out = song_dir / "kept.mid"
    assert render_main([str(song_dir / "song.mid"), "--keep-velocity", "-o", str(out)]) == 0
    mid = mido.MidiFile(str(out))
    velocities = {m.velocity for t in mid.tracks for m in t if m.type == "note_on"}
    assert velocities == {100}  # the synthetic file's constant source velocity


def test_metric_weight_and_features(song_dir):
    mid = mido.MidiFile(str(song_dir / "song.mid"))
    artifacts = load_artifacts(song_dir / "song.mid")
    assert not artifacts.missing
    tracks = extract_tracks(mid)
    annotate(tracks, artifacts, mid.ticks_per_beat)
    melody = tracks[1]  # track 0 is the tempo track
    downbeat = melody.notes[0]  # starts exactly on the first downbeat (0.5 s)
    assert downbeat.metric_weight == 1.0
    assert melody.notes[0].section == "verse"
    assert melody.notes[-1].section == "chorus"
    assert melody.notes[-1].section_intensity == 1.0
    # C over C:maj is the root -> tension 0
    c_notes = [n for n in melody.notes if n.pitch % 12 == 0 and n.start < 15.0]
    assert c_notes and all(n.tension == 0.0 for n in c_notes)


# --- Module F: loudness --------------------------------------------------------


def _crescendo_wav(path, *, seconds=8.0, samplerate=22050):
    t = np.arange(int(seconds * samplerate)) / samplerate
    amplitude = np.linspace(0.02, 0.9, t.size)
    sf.write(str(path), (amplitude * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32), samplerate)


def test_crescendo_maps_to_increasing_velocity(tmp_path):
    wav = tmp_path / "song_guitar.wav"
    _crescendo_wav(wav)
    curve = loudness.analyze_wav(wav)
    levels = [curve.level_at_onset(0.5 + i) for i in range(7)]
    assert all(b >= a for a, b in pairwise(levels))
    velocities = [20.0 + 100.0 * x**0.8 for x in levels]
    assert velocities[-1] > velocities[0] + 30


def test_stem_matching(tmp_path):
    stems = tmp_path / "stems"
    stems.mkdir()
    _crescendo_wav(stems / "song_guitar.wav", seconds=1.0)
    _crescendo_wav(stems / "song_vocals.wav", seconds=1.0)
    found = loudness.find_stem_wavs(stems)
    assert set(found) == {"guitar", "vocals"}

    from conftest import make_track

    guitar = make_track([], name="Guitar (song_guitar)")
    assert loudness.match_track(guitar, found) == stems / "song_guitar.wav"


def test_vel_blend_zero_uses_loudness_only(tmp_path, capsys):
    stems = tmp_path / "stems"
    stems.mkdir()
    _crescendo_wav(stems / "song_guitar.wav", seconds=12.0)
    make_midi(
        tmp_path / "song.mid",
        [[(60, 0.5 + i * 1.0, 0.4) for i in range(10)]],
        names=["guitar"],
    )
    out = tmp_path / "blended.mid"
    assert (
        render_main([str(tmp_path / "song.mid"), "--vel-blend", "0", "--seed", "1", "-o", str(out)])
        == 0
    )
    mid = mido.MidiFile(str(out))
    velocities = [m.velocity for t in mid.tracks for m in t if m.type == "note_on"]
    assert len(velocities) == 10
    assert velocities[-1] > velocities[0] + 20  # crescendo drives the velocities


def test_per_stem_midi_finds_song_context(tmp_path, capsys):
    """A per-stem MIDI under output/<id>/stems/midi/ finds the song's artifacts,
    its stem WAV, and its profile from the filename (real stems layout)."""
    song = tmp_path / "output" / "mysong"
    midi_dir = song / "stems" / "midi"
    wav_dir = song / "stems" / "mysong"
    midi_dir.mkdir(parents=True)
    wav_dir.mkdir(parents=True)
    from conftest import make_artifacts

    make_artifacts(song / "artifacts", "mysong")
    _crescendo_wav(wav_dir / "mysong_vocals.wav", seconds=12.0)
    make_midi(
        midi_dir / "mysong_vocals.mid",
        [[(60 + i % 5, 0.5 + i * 1.0, 0.6) for i in range(10)]],
    )

    assert song_name(midi_dir / "mysong_vocals.mid") == "mysong"
    artifacts = load_artifacts(midi_dir / "mysong_vocals.mid")
    assert not artifacts.missing  # found two levels up

    assert render_main([str(midi_dir / "mysong_vocals.mid"), "--seed", "3"]) == 0
    err = capsys.readouterr().err
    assert "Artifacts missing" not in err
    assert "profile=sung" in err  # stem name from the filename picks the profile
    dest = midi_dir / "mysong_vocals.mpe.mid"
    assert dest.is_file()

    # the stem WAV drives the velocities (crescendo -> rising)
    mid = mido.MidiFile(str(dest))
    velocities = [m.velocity for t in mid.tracks for m in t if m.type == "note_on"]
    assert velocities[-1] > velocities[0]


def test_wind_flag_renders_breath_and_monophony(tmp_path, capsys):
    # two-note chords under a melody: --wind must keep only the top voice
    chords = [[(60 + i, 0.5 + i * 0.7, 0.6), (48 + i, 0.5 + i * 0.7, 0.6)] for i in range(8)]
    make_midi(tmp_path / "duo.mid", [[n for pair in chords for n in pair]])
    out = tmp_path / "wind.mid"
    assert render_main([str(tmp_path / "duo.mid"), "--wind", "--seed", "2", "-o", str(out)]) == 0
    assert "profile=winds" in capsys.readouterr().err
    mid = mido.MidiFile(str(out))
    messages = [m for track in mid.tracks for m in track]
    note_ons = [m for m in messages if m.type == "note_on"]
    assert len(note_ons) == 8  # top voice only
    assert {m.note for m in note_ons} == {60 + i for i in range(8)}
    breath = [m for m in messages if m.type == "control_change" and m.control == 2]
    assert len(breath) > 8  # a real breath stream, not just per-note setup

    # without --wind: full polyphony, no breath CC
    plain = tmp_path / "plain.mid"
    assert render_main([str(tmp_path / "duo.mid"), "--seed", "2", "-o", str(plain)]) == 0
    messages = [m for track in mido.MidiFile(str(plain)).tracks for m in track]
    assert len([m for m in messages if m.type == "note_on"]) == 16
    assert not any(m.type == "control_change" and m.control == 2 for m in messages)


def test_per_stem_midi_in_upstream_layout(tmp_path, capsys):
    """Upstream's stem pipeline nests deeper than the old layout:
    output/<id>/stems/<id>/stem_midis/<id>_vocals.mid, with the WAVs under
    stems/<id>/stems/<id>/. Artifacts and stem audio must still be found."""
    song = tmp_path / "output" / "deep"
    midi_dir = song / "stems" / "deep" / "stem_midis"
    wav_dir = song / "stems" / "deep" / "stems" / "deep"
    midi_dir.mkdir(parents=True)
    wav_dir.mkdir(parents=True)
    from conftest import make_artifacts

    make_artifacts(song / "artifacts", "deep")
    _crescendo_wav(wav_dir / "deep_vocals.wav", seconds=12.0)
    make_midi(
        midi_dir / "deep_vocals.mid",
        [[(60 + i % 5, 0.5 + i * 1.0, 0.6) for i in range(10)]],
    )

    assert song_name(midi_dir / "deep_vocals.mid") == "deep"
    assert not load_artifacts(midi_dir / "deep_vocals.mid").missing

    from sound2midi.expression.cli import _default_stems_dir

    stems_dir = _default_stems_dir(midi_dir / "deep_vocals.mid")
    assert loudness.find_stem_wavs(stems_dir).get("vocals") == wav_dir / "deep_vocals.wav"

    assert render_main([str(midi_dir / "deep_vocals.mid"), "--seed", "3"]) == 0
    err = capsys.readouterr().err
    assert "Artifacts missing" not in err
    assert "profile=sung" in err
    mid = mido.MidiFile(str(midi_dir / "deep_vocals.mpe.mid"))
    velocities = [m.velocity for t in mid.tracks for m in t if m.type == "note_on"]
    assert velocities[-1] > velocities[0]  # the stem WAV's crescendo drives them
