"""Tests for the --airwave gesture synthesis (midi-sink sumi_ctl_t CC map)."""

from __future__ import annotations

import random

import mido
from conftest import make_artifacts, make_midi, make_note, make_track

from sound2midi.expression import airwave
from sound2midi.expression.cli import render_main
from sound2midi.expression.context import Artifacts


def _song_tracks():
    # a rising melody, denser and louder in the second half
    notes = []
    for i in range(16):
        note = make_note(55 + i * 2, 0.5 + i * 0.5, 0.45, phrase_pos=i / 15)
        note.velocity = 60 if i < 8 else 110
        note.phrase_peak = i == 12
        notes.append(note)
    quiet_half = [make_note(40, 0.5 + i * 1.0, 0.9) for i in range(4)]
    loud_half = [make_note(40, 4.5 + i * 0.25, 0.2) for i in range(16)]
    for n in quiet_half + loud_half:
        n.velocity = 50 if n.start < 4.5 else 120
    return [make_track(notes, index=0), make_track(quiet_half + loud_half, index=1)]


def _artifacts():
    return Artifacts(
        meter={
            "beats": [0.5 + 0.5 * i for i in range(20)],
            "first_downbeat": 0.5,
            "felt_beats_per_bar": 4,
            "bpm": 120.0,
        },
        sections=[("verse", 0.0, 4.5), ("chorus", 4.5, 10.0)],
    )


def test_airwave_covers_the_cc_map_deterministically():
    one = airwave.synthesize(_song_tracks(), _artifacts(), random.Random(7))
    two = airwave.synthesize(_song_tracks(), _artifacts(), random.Random(7))
    assert [(t, m.control, m.value) for t, _, _, m in one] == [
        (t, m.control, m.value) for t, _, _, m in two
    ]
    controls = {m.control for _, _, _, m in one}
    assert controls == set(airwave.AIRWAVE_CCS)
    for _, _, _, m in one:
        assert m.channel == airwave.CHANNEL == 0
        assert 0 <= m.value <= 127


def test_vortex_x_follows_the_melodic_contour():
    # the vortex is the attack hand: its centre tracks the tune sideways
    events = airwave.synthesize(_song_tracks(), _artifacts(), random.Random(1))
    vortex_x = [(t, m.value) for t, _, _, m in events if m.control == airwave.CC_VORTEX_X]
    early = [v for t, v in vortex_x if 0.5 <= t <= 2.0]
    late = [v for t, v in vortex_x if 6.5 <= t <= 8.0]
    assert sum(late) / len(late) > sum(early) / len(early) + 30  # the hand moved up-range


def test_pinch_cross_fires_at_the_section_boundary():
    events = airwave.synthesize(_song_tracks(), _artifacts(), random.Random(1))
    cross = [(t, m.value) for t, _, _, m in events if m.control == airwave.CC_PINCH_CROSS]
    near_boundary = [v for t, v in cross if 4.5 <= t <= 4.9]
    assert near_boundary and max(near_boundary) >= 100


def test_ripple_amp_tracks_song_energy():
    events = airwave.synthesize(_song_tracks(), _artifacts(), random.Random(1))
    amp = [(t, m.value) for t, _, _, m in events if m.control == airwave.CC_RIPPLE_AMP]
    quiet = [v for t, v in amp if 2.0 <= t <= 4.0]
    loud = [v for t, v in amp if 6.0 <= t <= 8.0]
    assert sum(loud) / len(loud) > sum(quiet) / len(quiet) + 20


def test_airwave_source_merges_gestures_into_the_same_file(tmp_path, capsys):
    make_midi(tmp_path / "song.mid", [[(60 + i % 7, 0.5 + i * 0.5, 0.45) for i in range(16)]])
    # the gesture source: a separate, steadily ASCENDING line
    make_midi(tmp_path / "gestures.mid", [[(48 + i * 2, 0.5 + i * 0.5, 0.45) for i in range(16)]])
    make_artifacts(tmp_path / "artifacts", "song")
    assert (
        render_main(
            [str(tmp_path / "song.mid"), "--airwave", str(tmp_path / "gestures.mid"), "--seed", "3"]
        )
        == 0
    )
    assert "gestures.mid" in capsys.readouterr().err
    assert not (tmp_path / "song.airwave.mid").exists()  # no companion: same file

    mid = mido.MidiFile(str(tmp_path / "song.mpe.mid"))
    messages = [m for track in mid.tracks for m in track if not m.is_meta]
    gestures = [
        m for m in messages if m.type == "control_change" and m.control in airwave.AIRWAVE_CCS
    ]
    assert gestures and all(m.channel == 0 for m in gestures)
    assert {m.control for m in gestures} == set(airwave.AIRWAVE_CCS)
    assert any(m.type == "note_on" and m.channel >= 1 for m in messages)  # notes intact

    # the swirl follows the GESTURE file's ascending contour
    abs_time = []
    tick = 0
    for track in mid.tracks:
        tick = 0
        for m in track:
            tick += m.time
            if m.type == "control_change" and m.control == airwave.CC_SWIRL_X:
                abs_time.append((tick, m.value))
    abs_time.sort()
    third = len(abs_time) // 3
    early = [v for _, v in abs_time[:third]]
    late = [v for _, v in abs_time[-third:]]
    assert sum(late) / len(late) > sum(early) / len(early) + 25


def test_airwave_missing_source_errors(tmp_path):
    import pytest

    make_midi(tmp_path / "song.mid", [[(60, 0.5, 0.45)]])
    with pytest.raises(SystemExit):
        render_main([str(tmp_path / "song.mid"), "--airwave", str(tmp_path / "nope.mid")])


def test_vortex_attacks_capped_and_swirl_is_the_wind():
    events = airwave.synthesize(_song_tracks(), _artifacts(), random.Random(1))
    by_cc: dict[int, list[tuple[float, int]]] = {}
    for t, _, _, m in events:
        if t < 8.0:  # while the music plays (before the hands rest)
            by_cc.setdefault(m.control, []).append((t, m.value))
    vortex = [v for _, v in by_cc[airwave.CC_VORTEX_STRENGTH]]
    swirl = [v for _, v in by_cc[airwave.CC_SWIRL_STRENGTH]]
    # the vortex is visually strong: brief accent flares only, hard-capped
    # at 47, fully SILENT between hits (no floor, snappy decay)
    assert max(vortex) <= 47
    assert min(vortex) == 0, "the vortex goes silent between attacks"
    ordered = sorted(vortex)
    assert ordered[len(ordered) // 2] < 15, "flares are the exception, not the state"
    # the swirl is the wind: a broad continuous layer using the full range
    assert max(swirl) > 100
    loud_half = [v for t, v in by_cc[airwave.CC_SWIRL_STRENGTH] if t > 5.5]
    assert min(loud_half) > 60, "the wind sustains, it does not pulse away"


def test_ripple_wavelength_is_always_visible():
    events = airwave.synthesize(_song_tracks(), _artifacts(), random.Random(1))
    freq = [m.value for _, _, _, m in events if m.control == airwave.CC_RIPPLE_FREQ]
    assert len(freq) > 5, "the wavelength moves, it is not a single constant"
    assert all(v >= 32 for v in freq), "a zero wavelength renders nothing"
    assert freq[-1] >= 32  # the resting value keeps a real wavelength too


def test_ripple_wavelength_swells_and_decays_with_the_amount():
    events = airwave.synthesize(_song_tracks(), _artifacts(), random.Random(1))
    freq = [(t, m.value) for t, _, _, m in events if m.control == airwave.CC_RIPPLE_FREQ]
    quiet = [v for t, v in freq if 2.0 <= t <= 4.0]
    loud = [v for t, v in freq if 6.0 <= t <= 8.0]
    assert quiet and loud
    assert sum(loud) / len(loud) > sum(quiet) / len(quiet) + 15
    assert max(v for _, v in freq) - min(v for _, v in freq) > 30  # a real swell
