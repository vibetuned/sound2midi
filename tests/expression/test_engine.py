"""Unit tests for the expression engine (acceptance criteria §11)."""

from __future__ import annotations

import random
from itertools import pairwise

from conftest import make_note, make_track

from sound2midi.expression import curves, timing, velocity
from sound2midi.expression.context import Artifacts, phrase_arc
from sound2midi.expression.mpe_encoder import cents_to_bend, encode

STRINGS = curves.load_profile("strings")
SUNG = curves.load_profile("sung")
PLUCK = curves.load_profile("pluck")


def render_notes(notes, profile, *, seed=1, legato="off", bend_range=48):
    track = make_track(notes)
    rng = random.Random(seed)
    groups = curves.detect_legato(track, profile, legato, bend_range)
    velocity.synthesize(track, rng)
    options = curves.RenderOptions(bend_range=bend_range, legato_mode=legato)
    return curves.render_track(track, groups, profile, rng, options)


# --- criterion 3: bend math -------------------------------------------------


def test_bend_100_cents_at_range_48():
    assert abs(cents_to_bend(100.0, 48) - 171) <= 1


def test_bend_clamped():
    assert cents_to_bend(1e9, 2) == 8191
    assert cents_to_bend(-1e9, 2) == -8192
    assert cents_to_bend(0.0, 48) == 0


def test_encoder_resets_at_every_note_off():
    perf = render_notes([make_note(60, 0.0, 1.0), make_note(64, 1.2, 1.0)], STRINGS)
    events = encode(perf)
    offs = [i for i, e in enumerate(events) if e[3].type == "note_off"]
    assert offs
    for i in offs:
        channel = events[i][3].channel
        following = [e[3] for e in events[i + 1 : i + 4]]
        assert any(
            m.type == "pitchwheel" and m.pitch == 0 and m.channel == channel for m in following
        )
        assert any(
            m.type == "aftertouch" and m.value == 0 and m.channel == channel for m in following
        )
        assert any(
            m.type == "control_change" and m.control == 74 and m.channel == channel
            for m in following
        )
    for _, _, _, msg in events:
        if msg.type == "pitchwheel":
            assert -8192 <= msg.pitch <= 8191


# --- criterion 4: no chirp (phase accumulation) ------------------------------


def test_vibrato_rate_stays_in_band_on_long_note():
    (perf,) = render_notes([make_note(60, 0.0, 4.0, tension=0.5)], STRINGS, seed=3)
    interior = [(t, cents) for t, _, _, cents in perf.samples if 0.8 <= t <= 3.2]
    crossings = [
        t1 + (t2 - t1) * (0 - c1) / (c2 - c1)
        for (t1, c1), (t2, c2) in pairwise(interior)
        if c1 < 0 <= c2 or c2 < 0 <= c1
    ]
    assert len(crossings) > 10
    for a, b in zip(crossings[:-2], crossings[2:], strict=False):  # full periods
        rate = 1.0 / (b - a)
        assert 4.5 <= rate <= 6.5, f"instantaneous vibrato rate {rate:.2f} Hz"


# --- criterion 5: humanization ----------------------------------------------


def test_identical_consecutive_notes_differ():
    notes = [make_note(60, 0.0, 1.0), make_note(60, 1.1, 1.0)]
    perf = render_notes(notes, STRINGS, seed=5)
    first = [(p, c, v) for _, p, c, v in perf[0].samples]
    second = [(p, c, v) for _, p, c, v in perf[1].samples]
    assert first[: len(second)] != second[: len(first)]
    assert perf[0].velocity != perf[1].velocity  # ±4 alternation on repeats


# --- criterion 6: delta thresholding ----------------------------------------


def test_delta_threshold_reduction():
    from sound2midi.expression.mpe_encoder import naive_event_count

    notes = [make_note(60 + i % 7, i * 0.55, 0.5, tension=0.5) for i in range(40)]
    perf = render_notes(notes, STRINGS, seed=2)
    events = encode(perf)
    naive = naive_event_count(perf)
    assert len(events) <= 0.4 * naive, f"{len(events)} vs naive {naive}"


# --- criterion 10: rubato ----------------------------------------------------


def _two_tracks():
    melody = make_track(
        [make_note(72 + i % 5, 0.5 + 0.5 * i, 0.45, metric_weight=1.0) for i in range(16)],
        index=0,
    )
    chords = make_track([make_note(48 + i % 3, 0.5 + 0.5 * i, 0.45) for i in range(16)], index=1)
    return [melody, chords]


def test_rubato_zero_is_grid_exact():
    tracks = _two_tracks()
    timing.apply_rubato(tracks, Artifacts(), random.Random(1), strength=0.0)
    for track in tracks:
        for note in track.notes:
            assert note.perf_start == note.start
            assert note.perf_end == note.end


def test_rubato_capped_and_shared():
    tracks = _two_tracks()
    strength = 0.5
    times, displacement = timing._build_warp(tracks, Artifacts(), None, strength)
    assert max(abs(d) for d in displacement) <= timing.DRIFT_CAP_S * strength + 1e-9
    for a, b in pairwise(times):
        assert b > a  # warp sample grid is monotonic

    timing.apply_rubato(tracks, Artifacts(), random.Random(1), strength=strength)
    for track in tracks:
        for note in track.notes:
            # warp cap + melody lead + 5-sigma jitter
            assert abs(note.perf_start - note.start) <= 0.08 * strength + 0.025 * strength + 0.015

    # chord attacks stay within the asynchrony window (same warp on all tracks)
    for a, b in zip(tracks[0].notes, tracks[1].notes, strict=False):
        assert abs(a.perf_start - b.perf_start) <= 0.05


def test_rubato_reproducible():
    one, two = _two_tracks(), _two_tracks()
    timing.apply_rubato(one, Artifacts(), random.Random(9), strength=0.5)
    timing.apply_rubato(two, Artifacts(), random.Random(9), strength=0.5)
    assert [n.perf_start for t in one for n in t.notes] == [
        n.perf_start for t in two for n in t.notes
    ]


# --- criteria 11/12: legato ---------------------------------------------------


def _mono_line(intervals: list[int], *, gap: float = 0.02) -> list:
    notes = []
    pitch, t = 60, 0.0
    for delta in [0, *intervals]:
        pitch += delta
        notes.append(make_note(pitch, t, 0.5))
        t += 0.5 + gap
    return notes


def test_glide_renders_group_as_one_note():
    notes = _mono_line([2, 2, -1])
    perf = render_notes(notes, SUNG, legato="glide")
    assert len(perf) == 1
    assert perf[0].pitch == 60
    assert perf[0].end - perf[0].start > 1.9
    events = encode(perf, bend_range=48)
    for _, _, _, msg in events:
        if msg.type == "pitchwheel":
            assert -8192 <= msg.pitch <= 8191
    # the bend actually travels: final offset ≈ +3 st = 300 cents
    final_cents = perf[0].samples[-1][3]
    assert abs(final_cents - 300.0) < 40.0  # within vibrato depth of the target


def test_glide_budget_splits_chain():
    notes = _mono_line([4, 4, 4])  # cumulative +12 st
    perf = render_notes(notes, SUNG, legato="glide", bend_range=6)  # budget ±4 st
    assert len(perf) > 1  # clean retrigger instead of leaving the budget


def test_legato_off_keeps_all_notes():
    notes = _mono_line([2, 2, -1])
    perf = render_notes(notes, SUNG, legato="off")
    assert len(perf) == len(notes)


def test_overlap_extends_and_softens():
    notes = _mono_line([2, 2])
    perf = render_notes(notes, STRINGS, legato="overlap")
    assert len(perf) == len(notes)
    assert perf[0].end > notes[0].perf_end  # overlaps its successor
    # successor attack transient measurably reduced: initial pressure jump smaller
    first_attack = perf[0].samples[0][1] - perf[0].samples[5][1]
    second_attack = perf[1].samples[0][1] - perf[1].samples[5][1]
    assert second_attack < first_attack


# --- criterion 9: profiles ----------------------------------------------------


def test_auto_profile_mapping():
    assert curves.auto_profile("vocals") == "sung"
    assert curves.auto_profile("guitar") == "pluck"
    assert curves.auto_profile("bass") == "pluck"
    assert curves.auto_profile("piano") == "keys"
    assert curves.auto_profile("other") == "strings"
    assert curves.auto_profile("drums") == "none"
    assert curves.auto_profile("anything", is_drum=True) == "none"


def test_pluck_has_no_vibrato_bend():
    (perf,) = render_notes([make_note(60, 0.0, 2.0)], PLUCK)
    assert all(cents == 0.0 for _, _, _, cents in perf.samples)
    events = encode([perf])
    wheels = [m for _, _, _, m in events if m.type == "pitchwheel"]
    assert all(m.pitch == 0 for m in wheels)  # only the initial value and the reset


# --- velocity ------------------------------------------------------------------


def test_velocity_formula_reacts_to_context():
    strong = make_note(60, 0.0, 0.5, metric_weight=1.0, phrase_pos=0.5)
    weak = make_note(60, 1.0, 0.5, metric_weight=0.1, phrase_pos=0.5)
    track = make_track([strong, weak])
    velocity.synthesize(track, random.Random(0))
    assert strong.velocity > weak.velocity
    assert 20 <= weak.velocity <= 127


def test_phrase_arc_shape():
    assert phrase_arc(0.0) == 0.6
    assert phrase_arc(0.5) > 0.9
    assert abs(phrase_arc(1.0) - 0.6) < 1e-6


def test_channel_stealing_over_15_voices(capsys):
    notes = [make_note(40 + i, 0.01 * i, 5.0) for i in range(20)]  # 20 concurrent
    perf = render_notes(notes, PLUCK)
    events = encode(perf)
    assert "stole the oldest" in capsys.readouterr().err
    sounding: set[tuple[int, int]] = set()
    peak = 0
    for _, _, _, msg in events:
        if msg.type == "note_on":
            key = (msg.channel, msg.note)
            assert key not in sounding  # a stolen note was closed first
            sounding.add(key)
            peak = max(peak, len(sounding))
        elif msg.type == "note_off":
            sounding.discard((msg.channel, msg.note))
    assert peak == 15  # never more than the member channels


def test_match_port():
    import pytest

    from sound2midi.expression.realtime import match_port

    ports = ["IAC Driver Bus 1", "Vital", "ROLI Seaboard", "Network Session 1"]
    assert match_port(ports, "Vital") == 1
    assert match_port(ports, "vital") == 1
    assert match_port(ports, "roli") == 2
    assert match_port(ports, "2") == 2  # by index
    assert match_port(ports, "IAC Driver Bus 1") == 0
    assert match_port(ports, "1") == 1  # a digit is an index first
    with pytest.raises(SystemExit):  # ambiguous substring
        match_port(ports, "i")  # in Vital, ROLI, Session...
    with pytest.raises(SystemExit):
        match_port(ports, "does-not-exist")
