"""Streaming scheduler tests: nothing is lost at the start of a performance."""

from __future__ import annotations

import time

import mido

from sound2midi.expression import realtime


class FakeOut:
    """Captures raw MIDI bytes with the wall-clock moment they were sent."""

    def __init__(self) -> None:
        self.sent: list[tuple[float, mido.Message]] = []
        self.t0 = time.monotonic()

    def send_message(self, data: list[int]) -> None:
        self.sent.append((time.monotonic() - self.t0, mido.Message.from_bytes(data)))

    def notes_on(self) -> list[int]:
        return [m.note for _, m in self.sent if m.type == "note_on"]


def _event(t: float, seq: int, msg: mido.Message):
    return (t, 2, seq, msg)


def _opening_performance():
    """A performance whose first note is pulled slightly before t=0 by rubato."""
    return [
        _event(-0.004, 0, mido.Message("aftertouch", channel=1, value=40)),
        _event(-0.004, 1, mido.Message("note_on", channel=1, note=60, velocity=90)),
        _event(0.010, 2, mido.Message("note_on", channel=2, note=64, velocity=90)),
        _event(0.040, 3, mido.Message("note_off", channel=1, note=60, velocity=64)),
        _event(0.060, 4, mido.Message("note_off", channel=2, note=64, velocity=64)),
    ]


def test_note_before_zero_is_not_dropped():
    """Regression: a note_on at a negative time used to be filtered away while
    its note_off survived, silently losing the opening note under --live."""
    out = FakeOut()
    realtime.stream(_opening_performance(), [out], bend_range=48, handshake=False)
    assert out.notes_on() == [60, 64], "the opening note must survive"
    offs = [m.note for _, m in out.sent if m.type == "note_off"]
    assert sorted(offs) == [60, 64]  # no orphaned note_off


def test_relative_timing_survives_the_negative_origin():
    out = FakeOut()
    realtime.stream(_opening_performance(), [out], bend_range=48, handshake=False)
    at = {m.note: t for t, m in out.sent if m.type == "note_on"}
    # the two note_ons are 14 ms apart in the performance; keep that spacing
    assert 0.008 < at[64] - at[60] < 0.045


def test_start_still_seeks():
    out = FakeOut()
    realtime.stream(_opening_performance(), [out], bend_range=48, start=0.005, handshake=False)
    assert out.notes_on() == [64], "--start seeks past earlier events"


def test_handshake_settles_before_the_first_note():
    out = FakeOut()
    realtime.stream(_opening_performance(), [out], bend_range=48, handshake=True)
    first_note = next(t for t, m in out.sent if m.type == "note_on")
    rpns = [t for t, m in out.sent if m.type == "control_change" and m.control in (6, 100, 101)]
    last_rpn = max(rpns)
    assert first_note - last_rpn >= realtime.HANDSHAKE_SETTLE_S * 0.8
    assert out.notes_on() == [60, 64]
