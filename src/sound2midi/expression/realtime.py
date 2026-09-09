"""Module E — realtime MPE streaming to a DAW and/or MIDI destinations.

MIDI consumers come in two shapes, and we serve both at once (the same model
as battuta's playback output):

- **source subscribers** (a DAW, listener apps like midi-sink) receive from
  the virtual *source* port we publish — it stays open even when ``--port``
  is used, unless explicitly disabled;
- **destinations** (a ROLI keyboard, an iPad, GarageBand's virtual input)
  never subscribe to anything — they only receive what is sent to their
  output port, so ``connect_to`` / ``--port`` (repeatable, ``all`` for every
  destination) opens and feeds them directly.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterable

import mido

from sound2midi.expression.mpe_encoder import (
    MEMBER_CHANNELS,
    TimedMessage,
    handshake_messages,
)

PORT_NAME = "sound2midi MPE"
_SLEEP_MARGIN_S = 0.002  # sleep to just before the deadline, busy-wait the rest


def list_ports() -> list[str]:
    """The system's MIDI output destinations (DAW inputs, IAC buses, devices)."""
    import rtmidi

    out = rtmidi.MidiOut()  # ty: ignore[unresolved-attribute]
    try:
        return list(out.get_ports())
    finally:
        del out


def match_port(ports: list[str], query: str) -> int:
    """Resolve a ``--port`` query to a port index: index, exact name, or a
    unique case-insensitive substring. Exits with the port list otherwise."""
    if query.isdigit() and int(query) < len(ports):
        return int(query)
    lowered = query.lower()
    exact = [i for i, p in enumerate(ports) if p.lower() == lowered]
    if len(exact) == 1:
        return exact[0]
    matches = [i for i, p in enumerate(ports) if lowered in p.lower()]
    if len(matches) == 1:
        return matches[0]
    listing = "\n".join(f"  [{i}] {p}" for i, p in enumerate(ports)) or "  (none)"
    reason = "matches several ports" if len(matches) > 1 else "matches no port"
    raise SystemExit(f"--port {query!r} {reason}. Available MIDI destinations:\n{listing}")


def open_outputs(
    port_name: str = PORT_NAME,
    *,
    connect_to: list[str] | None = None,
    virtual: bool = True,
) -> list:
    """Open the MIDI output(s): the virtual source plus any destinations.

    ``virtual`` publishes a virtual *source* port that subscribing apps (a
    DAW, source listeners) receive from; it stays up alongside any
    ``connect_to`` connections. ``connect_to`` additionally sends to existing
    *destinations* (each by index, name, or a unique substring; ``all`` for
    every destination — hardware like a ROLI keyboard or an iPad included),
    which never subscribe to sources and would otherwise receive nothing.
    """
    import rtmidi

    outputs = []
    if virtual:
        if sys.platform == "win32":
            if not connect_to:
                print(
                    "Windows does not support virtual MIDI ports. Install loopMIDI\n"
                    "(https://www.tobias-erichsen.de/software/loopmidi.html), create\n"
                    "a port, and re-run with  --port <loopMIDI port name>.",
                    file=sys.stderr,
                )
                raise SystemExit(1)
        else:
            out = rtmidi.MidiOut()  # ty: ignore[unresolved-attribute]
            out.open_virtual_port(port_name)
            print(f"Virtual MIDI source open: {port_name}", file=sys.stderr)
            outputs.append(out)

    if connect_to:
        ports = list_ports()
        if any(query.lower() == "all" for query in connect_to):
            indices = list(range(len(ports)))
            if not indices:
                raise SystemExit("--port all: no MIDI destinations found.")
        else:
            indices = sorted({match_port(ports, query) for query in connect_to})
        for index in indices:  # one MidiOut per destination — each can open one port
            out = rtmidi.MidiOut()  # ty: ignore[unresolved-attribute]
            out.open_port(index)
            print(f"Connected to MIDI destination: {ports[index]}", file=sys.stderr)
            outputs.append(out)

    if not outputs:
        raise SystemExit("No MIDI outputs: --no-virtual needs at least one --port.")
    return outputs


def send_message(outputs: list, msg: mido.Message) -> None:
    data = msg.bytes()
    for out in outputs:
        out.send_message(data)


def panic(outputs: list) -> None:
    """All-notes-off + bend/pressure/CC74 resets on every member channel.

    Best-effort per output: it runs from ``stream``'s ``finally``, and one
    dead destination (a device unplugged or asleep mid-session) must not
    prevent the others from being cleaned up.
    """
    messages = [
        msg.bytes()
        for channel in (0, *MEMBER_CHANNELS)
        for msg in (
            mido.Message("control_change", channel=channel, control=123, value=0),
            mido.Message("control_change", channel=channel, control=120, value=0),
            mido.Message("pitchwheel", channel=channel, pitch=0),
            mido.Message("aftertouch", channel=channel, value=0),
        )
    ]
    for out in outputs:
        try:
            for data in messages:
                out.send_message(data)
        except Exception as exc:
            print(f"warning: panic failed on one MIDI output: {exc}", file=sys.stderr)


def _progress_bar(total: float, description: str):
    """A tqdm bar over playback seconds (auto-disabled when stderr is not a TTY)."""
    from tqdm import tqdm

    return tqdm(
        total=round(total, 1),
        desc=description,
        disable=None,  # tqdm: None = only when stderr is a TTY
        leave=False,
        bar_format="{desc} {percentage:3.0f}%|{bar}| {n:.0f}/{total:.0f}s",
    )


def stream(
    events: Iterable[TimedMessage],
    outputs: list,
    *,
    bend_range: int,
    start: float = 0.0,
    loop: bool = False,
    handshake: bool = True,
    description: str = "playing",
) -> None:
    """Send the handshake, then the events on a monotonic-clock schedule.

    Ctrl-C always leaves the synth clean (panic in ``finally``). Pass
    ``handshake=False`` when streaming a pre-rendered ``.mpe.mid``, which
    already carries its own handshake messages. A progress bar tracks the
    position in the performance; it refreshes only while the scheduler is
    sleeping ahead of the next deadline, so send timing is unaffected.
    """
    playlist = [e for e in events if e[0] >= start]
    if not playlist:
        print("Nothing to play after --start.", file=sys.stderr)
        return
    duration = playlist[-1][0] - start
    try:
        while True:
            if handshake:
                for msg in handshake_messages(bend_range):
                    send_message(outputs, msg)
            bar = _progress_bar(duration, description)
            try:
                t0 = time.monotonic()
                last_draw = 0.0
                for event_time, _, _, msg in playlist:
                    deadline = t0 + (event_time - start)
                    while True:
                        now = time.monotonic()
                        delay = deadline - now
                        if delay <= _SLEEP_MARGIN_S:
                            break
                        if now - last_draw >= 0.1:  # refresh() bypasses tqdm throttling
                            bar.n = min(now - t0, bar.total)
                            bar.refresh()
                            last_draw = now
                        time.sleep(min(delay - _SLEEP_MARGIN_S, 0.2))
                    while time.monotonic() < deadline:
                        pass
                    send_message(outputs, msg)
                bar.n = bar.total
                bar.refresh()
            finally:
                bar.close()
            if not loop:
                break
    finally:
        panic(outputs)
