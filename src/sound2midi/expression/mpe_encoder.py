"""Module D — MPE encoding, shared by the file writer and the realtime stream.

One lower MPE zone: master channel 1 (index 0), member channels 2-16
(indices 1-15). Notes are round-robin allocated to member channels; per-note
expression rides aftertouch (pressure), CC74 (timbre) and pitch bend
(vibrato/glides). Emission is delta-thresholded, and every note_off is followed
by full resets so channel reuse never bleeds expression.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import mido

from sound2midi.beatgrid import sec_to_tick_fn, tempo_changes

DEFAULT_BEND_RANGE = 48
MEMBER_CHANNELS: tuple[int, ...] = tuple(range(1, 16))
MASTER_CHANNEL = 0
CC74_FLOOR = 24
BEND_DELTA = 16  # emit a new pitchwheel only when it moved ≥ this many units
# On top of the delta thresholds, each stream is rate-limited: values that
# would be replaced within the same interval carry no audible information.
PRESSURE_MIN_DT = 1.0 / 30.0
CC74_MIN_DT = 1.0 / 30.0
BEND_MIN_DT = 1.0 / 45.0
FILE_PPQ = 480

# Message ordering at equal times: end-of-note first so channel reuse is clean.
_PRI_OFF = 0
_PRI_SETUP = 1
_PRI_NOTE_ON = 2
_PRI_STREAM = 3


@dataclass
class PerfNote:
    """A performance note: final timing, velocity, and sampled streams.

    ``samples`` rows are ``(t_abs_seconds, pressure 0..127, cc74 0..127,
    bend cents)``; an empty list means velocity-only (drums / profile none).
    """

    track: int
    start: float
    end: float
    pitch: int
    velocity: int
    samples: list[tuple[float, float, float, float]] = field(default_factory=list)


TimedMessage = tuple[float, int, int, mido.Message]  # (time, priority, seq, message)


def cents_to_bend(cents: float, bend_range: int) -> int:
    """Pitch-bend units for a cent offset at the given RPN0 sensitivity."""
    value = round(cents / (100.0 * bend_range) * 8192.0)
    return max(-8192, min(8191, value))


def handshake_messages(bend_range: int) -> list[mido.Message]:
    """MPE zone configuration + pitch-bend sensitivity on the member channels."""
    messages = [
        # RPN 6 on the master channel: lower zone with 15 member channels.
        mido.Message("control_change", channel=MASTER_CHANNEL, control=101, value=0),
        mido.Message("control_change", channel=MASTER_CHANNEL, control=100, value=6),
        mido.Message("control_change", channel=MASTER_CHANNEL, control=6, value=15),
    ]
    for channel in MEMBER_CHANNELS:
        messages += [
            # RPN 0: pitch-bend sensitivity (semitones).
            mido.Message("control_change", channel=channel, control=101, value=0),
            mido.Message("control_change", channel=channel, control=100, value=0),
            mido.Message("control_change", channel=channel, control=6, value=bend_range),
            # RPN null, so later CC6 traffic can't retune anything.
            mido.Message("control_change", channel=channel, control=101, value=127),
            mido.Message("control_change", channel=channel, control=100, value=127),
        ]
    return messages


@dataclass
class _Active:
    note: PerfNote
    channel: int
    end: float


def _allocate_channels(notes: list[PerfNote]) -> tuple[dict[int, int], dict[int, float]]:
    """Round-robin member-channel assignment with oldest-note stealing.

    Returns ``{id(note): channel}`` and ``{id(note): truncated_end}`` (an end
    earlier than the note's own when the note was stolen from).
    """
    channels: dict[int, int] = {}
    ends: dict[int, float] = {}
    active: list[_Active] = []
    cursor = 0
    stolen = 0
    for note in sorted(notes, key=lambda n: n.start):
        active = [a for a in active if a.end > note.start + 1e-9]
        busy = {a.channel for a in active}
        channel = None
        for offset in range(len(MEMBER_CHANNELS)):
            candidate = MEMBER_CHANNELS[(cursor + offset) % len(MEMBER_CHANNELS)]
            if candidate not in busy:
                channel = candidate
                cursor = (cursor + offset + 1) % len(MEMBER_CHANNELS)
                break
        if channel is None:  # all member channels busy: steal the oldest note
            oldest = min(active, key=lambda a: a.note.start)
            active.remove(oldest)
            ends[id(oldest.note)] = note.start
            channel = oldest.channel
            stolen += 1
        channels[id(note)] = channel
        ends[id(note)] = note.end
        active.append(_Active(note=note, channel=channel, end=note.end))
    if stolen:
        print(
            f"warning: more than {len(MEMBER_CHANNELS)} simultaneous voices; "
            f"stole the oldest note {stolen} time(s).",
            file=sys.stderr,
        )
    return channels, ends


def encode(
    notes: list[PerfNote],
    *,
    bend_range: int = DEFAULT_BEND_RANGE,
    cc74_floor: int = CC74_FLOOR,
) -> list[TimedMessage]:
    """Encode performance notes into a time-sorted MPE message stream.

    The handshake is not included — prepend :func:`handshake_messages` at the
    start of the file/stream.
    """
    channels, ends = _allocate_channels(notes)
    events: list[TimedMessage] = []
    seq = 0

    def emit(time: float, priority: int, message: mido.Message) -> None:
        nonlocal seq
        events.append((time, priority, seq, message))
        seq += 1

    for note in sorted(notes, key=lambda n: n.start):
        channel = channels[id(note)]
        end = max(ends[id(note)], note.start + 0.005)
        samples = [s for s in note.samples if s[0] < end - 1e-9]

        first = samples[0] if samples else (note.start, 0.0, float(cc74_floor), 0.0)
        bend = cents_to_bend(first[3], bend_range)
        pressure = round(min(127.0, max(0.0, first[1])))
        cc74 = round(min(127.0, max(0.0, first[2])))
        emit(note.start, _PRI_SETUP, mido.Message("pitchwheel", channel=channel, pitch=bend))
        emit(note.start, _PRI_SETUP, mido.Message("aftertouch", channel=channel, value=pressure))
        emit(
            note.start,
            _PRI_SETUP,
            mido.Message("control_change", channel=channel, control=74, value=cc74),
        )
        emit(
            note.start,
            _PRI_NOTE_ON,
            mido.Message("note_on", channel=channel, note=note.pitch, velocity=note.velocity),
        )

        last_bend, last_pressure, last_cc74 = bend, pressure, cc74
        t_pressure = t_cc74 = t_bend = note.start
        for t, raw_pressure, raw_cc74, cents in samples[1:]:
            value = round(min(127.0, max(0.0, raw_pressure)))
            if value != last_pressure and t - t_pressure >= PRESSURE_MIN_DT:
                emit(t, _PRI_STREAM, mido.Message("aftertouch", channel=channel, value=value))
                last_pressure = value
                t_pressure = t
            value = round(min(127.0, max(0.0, raw_cc74)))
            if value != last_cc74 and t - t_cc74 >= CC74_MIN_DT:
                emit(
                    t,
                    _PRI_STREAM,
                    mido.Message("control_change", channel=channel, control=74, value=value),
                )
                last_cc74 = value
                t_cc74 = t
            value = cents_to_bend(cents, bend_range)
            if abs(value - last_bend) >= BEND_DELTA and t - t_bend >= BEND_MIN_DT:
                emit(t, _PRI_STREAM, mido.Message("pitchwheel", channel=channel, pitch=value))
                last_bend = value
                t_bend = t

        emit(
            end,
            _PRI_OFF,
            mido.Message("note_off", channel=channel, note=note.pitch, velocity=64),
        )
        # Full resets so the next note on this channel starts clean
        # (mido pitchwheel center is 0, not 8192).
        emit(end, _PRI_OFF, mido.Message("pitchwheel", channel=channel, pitch=0))
        emit(end, _PRI_OFF, mido.Message("aftertouch", channel=channel, value=0))
        emit(
            end,
            _PRI_OFF,
            mido.Message("control_change", channel=channel, control=74, value=cc74_floor),
        )

    events.sort(key=lambda e: (e[0], e[1], e[2]))
    return events


def naive_event_count(notes: list[PerfNote]) -> int:
    """Events a naive 60 Hz emitter would send (for the reduction target)."""
    return sum(3 * len(n.samples) + 2 for n in notes)


def write_file(
    events: list[TimedMessage],
    src: mido.MidiFile,
    dest: str,
    *,
    bend_range: int = DEFAULT_BEND_RANGE,
) -> None:
    """Write the MPE performance as a Type 1 file at PPQ 480.

    The source tempo map (and time signatures) are preserved — re-ticked to
    the new PPQ — so the file stays aligned with ``meter.json`` and the audio.
    """
    out = mido.MidiFile(type=1, ticks_per_beat=FILE_PPQ)
    scale = FILE_PPQ / src.ticks_per_beat

    meta_events: list[tuple[int, mido.MetaMessage]] = [
        (round(tick * scale), mido.MetaMessage("set_tempo", tempo=tempo))
        for tick, tempo in tempo_changes(src)
    ]
    for track in src.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.is_meta and msg.type == "time_signature":
                meta_events.append((round(tick * scale), msg.copy()))
    meta_events.sort(key=lambda e: e[0])

    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage("track_name", name="tempo", time=0))
    tick = 0
    for abs_tick, msg in meta_events:
        msg = msg.copy()
        msg.time = abs_tick - tick
        tempo_track.append(msg)
        tick = abs_tick
    out.tracks.append(tempo_track)

    sec_to_tick = sec_to_tick_fn(out)
    performance = mido.MidiTrack()
    performance.append(mido.MetaMessage("track_name", name="MPE performance", time=0))
    tick = 0
    for msg in handshake_messages(bend_range):
        msg.time = 0
        performance.append(msg)
    for time, _, _, msg in events:
        abs_tick = max(0, sec_to_tick(time))
        msg = msg.copy()
        msg.time = max(0, abs_tick - tick)
        performance.append(msg)
        tick = max(tick, abs_tick)
    out.tracks.append(performance)
    out.save(dest)
