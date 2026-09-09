"""CLI entry points: ``sound2midi-mpe`` (render) and ``sound2midi-mpe-play`` (stream)."""

from __future__ import annotations

import argparse
import contextlib
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import mido

from sound2midi.beatgrid import tick_to_sec_fn
from sound2midi.expression import context, curves, loudness, timing, velocity
from sound2midi.expression import mpe_encoder as encoder
from sound2midi.expression.mpe_encoder import PerfNote, TimedMessage


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("midi", type=Path, help="Source MIDI file.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Artifacts directory (default: artifacts/ next to the MIDI).",
    )
    parser.add_argument(
        "--profile",
        choices=curves.available_profiles(),
        default=None,
        help="Force one instrument profile for every track "
        "(default: auto-select per track from its name).",
    )
    parser.add_argument(
        "--track-profile",
        action="append",
        default=[],
        metavar="NAME=PROFILE",
        help="Override the profile for one track, by track name or index. Repeatable.",
    )
    parser.add_argument(
        "--bend-range",
        type=int,
        default=encoder.DEFAULT_BEND_RANGE,
        help="MPE pitch-bend sensitivity in semitones (default: %(default)s).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible output.")
    parser.add_argument(
        "--rubato",
        type=float,
        default=0.5,
        metavar="STRENGTH",
        help="Expressive-timing strength 0-1; 0 disables (default: %(default)s).",
    )
    parser.add_argument(
        "--legato",
        choices=curves.LEGATO_MODES,
        default="auto",
        help="Legato rendering: auto = per profile (default: %(default)s).",
    )
    parser.add_argument(
        "--keep-velocity",
        action="store_true",
        help="Keep the source velocities instead of synthesizing them.",
    )
    parser.add_argument(
        "--vel-blend",
        type=float,
        default=0.3,
        metavar="W",
        help="Weight of the synthesized velocity vs stem loudness (default: %(default)s).",
    )
    parser.add_argument(
        "--loudness",
        choices=("rms", "lufs"),
        default="rms",
        help="Stem loudness measure (default: %(default)s).",
    )
    parser.add_argument(
        "--loudness-pressure",
        action="store_true",
        help="Blend the stem loudness curve into the pressure stream (50/50).",
    )
    parser.add_argument(
        "--stems-dir",
        type=Path,
        default=None,
        help="Stem WAV directory (default: stems/ next to the MIDI, when present).",
    )
    parser.add_argument(
        "--solo-track", type=int, default=None, metavar="N", help="Render only track N."
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Velocity only, expression streams muted (for A/B comparison).",
    )


def _profile_overrides(entries: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for entry in entries:
        name, sep, profile = entry.partition("=")
        if not sep:
            raise SystemExit(f"--track-profile expects NAME=PROFILE, got {entry!r}")
        overrides[name.strip().lower()] = profile.strip()
    return overrides


def _program_profile(program: int) -> str:
    """Fallback profile from the GM program when a track has no usable name."""
    if program < 8:
        return "keys"
    if 24 <= program <= 39:  # guitars + basses
        return "pluck"
    if 88 <= program <= 95:
        return "pads"
    if 64 <= program <= 79:
        return "winds"
    return "strings"


def _resolve_profile(
    track: context.TrackData,
    forced: str | None,
    overrides: dict[str, str],
    hint: str | None,
) -> curves.Profile:
    if track.is_drum:
        return curves.load_profile("none")
    name = overrides.get(str(track.index)) or overrides.get(track.name.lower())
    if name is None:
        name = forced
    if name is None:
        auto = curves.auto_profile(track.name, is_drum=track.is_drum)
        if auto == "strings" and hint is not None:
            # a per-stem MIDI: the filename says which stem every track is
            auto = curves.auto_profile(hint)
        if auto == "strings" and not track.name.strip():
            auto = _program_profile(track.program)
        name = auto
    return curves.load_profile(name)


def _default_stems_dir(midi_path: Path) -> Path:
    """The song's stem-WAV root: ``stems/`` next to the MIDI, or the ``stems``
    tree a per-stem MIDI already lives in (``.../stems/midi/<song>_vocals.mid``)."""
    for base in list(midi_path.resolve().parents)[:3]:
        if base.name == "stems":
            return base
        candidate = base / "stems"
        if candidate.is_dir():
            return candidate
    return midi_path.parent / "stems"


@dataclass
class Performance:
    mid: mido.MidiFile
    events: list[TimedMessage]
    perf_notes: list[PerfNote]
    bend_range: int


def build_performance(args: argparse.Namespace) -> Performance:
    """The shared render pipeline: features → rubato → legato → velocity → curves."""
    midi_path: Path = args.midi
    if not midi_path.is_file():
        raise SystemExit(f"MIDI file not found: {midi_path}")
    mid = mido.MidiFile(str(midi_path))
    rng = random.Random(args.seed)

    artifacts = context.load_artifacts(midi_path, args.artifacts_dir)
    if artifacts.missing:
        _log(f"Artifacts missing (using fallbacks): {', '.join(artifacts.missing)}")

    tracks = context.extract_tracks(mid)
    context.annotate(tracks, artifacts, mid.ticks_per_beat)

    if args.rubato > 0:
        if not timing.is_quantized(tracks, artifacts.grid(), mid.ticks_per_beat):
            _log(
                "Source timing looks human (raw transcription); "
                "consider --rubato 0 to keep it untouched."
            )
        timing.apply_rubato(tracks, artifacts, rng, strength=min(1.0, max(0.0, args.rubato)))

    overrides = _profile_overrides(args.track_profile)
    hint = context.stem_hint(midi_path)
    stems_dir = args.stems_dir if args.stems_dir is not None else _default_stems_dir(midi_path)
    stem_wavs = loudness.find_stem_wavs(stems_dir)

    if args.legato in ("auto", "glide") and args.bend_range < 24:
        _log(
            f"warning: glide legato needs a wide bend range to be usable; "
            f"--bend-range {args.bend_range} will break most chains "
            f"(the default is {encoder.DEFAULT_BEND_RANGE})."
        )

    perf_notes: list[PerfNote] = []
    for track in tracks:
        if not track.notes or (args.solo_track is not None and track.index != args.solo_track):
            continue
        profile = _resolve_profile(track, args.profile, overrides, hint)

        groups = curves.detect_legato(track, profile, args.legato, args.bend_range)

        stem_curve = None
        wav = loudness.match_track(track, stem_wavs, hint=hint)
        if wav is not None:
            stem_curve = loudness.analyze_wav(wav, mode=args.loudness)
        if not args.keep_velocity:
            loudness_x: list[float | None] | None = None
            if stem_curve is not None:
                loudness_x = [stem_curve.level_at_onset(n.perf_start) for n in track.notes]
            velocity.synthesize(track, rng, loudness_x=loudness_x, vel_blend=args.vel_blend)

        options = curves.RenderOptions(
            bend_range=args.bend_range,
            legato_mode=args.legato,
            loudness_pressure=args.loudness_pressure,
            loudness_curve=stem_curve,
            dry=getattr(args, "dry", False),
        )
        label = track.name or f"track {track.index}"
        _log(f"  {label}: profile={profile.name}, {len(track.notes)} notes")
        perf_notes.extend(curves.render_track(track, groups, profile, rng, options))

    events = encoder.encode(perf_notes, bend_range=args.bend_range)
    return Performance(mid=mid, events=events, perf_notes=perf_notes, bend_range=args.bend_range)


def render_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sound2midi-mpe",
        description="Render a MIDI file + song artifacts into an expressive MPE .mpe.mid.",
    )
    _add_common_args(parser)
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Output path (default: <name>.mpe.mid)."
    )
    args = parser.parse_args(argv)

    performance = build_performance(args)
    dest = args.output or args.midi.with_name(args.midi.stem + ".mpe.mid")
    encoder.write_file(
        performance.events, performance.mid, str(dest), bend_range=performance.bend_range
    )

    naive = encoder.naive_event_count(performance.perf_notes)
    if naive:
        reduction = 100.0 * (1.0 - len(performance.events) / naive)
        _log(f"Emitted {len(performance.events)} events ({reduction:.0f}% below naive 60 Hz).")
    print(f"MPE performance written to {dest}")
    return 0


def _file_events(midi_path: Path) -> tuple[list[TimedMessage], int]:
    """A pre-rendered file's channel messages with absolute times, for streaming."""
    mid = mido.MidiFile(str(midi_path))
    tick_to_sec = tick_to_sec_fn(mid)
    events: list[TimedMessage] = []
    seq = 0
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if not msg.is_meta:
                events.append((tick_to_sec(tick), 0, seq, msg))
                seq += 1
    events.sort(key=lambda e: (e[0], e[2]))
    return events, len(mid.tracks)


def play_main(argv: list[str] | None = None) -> int:
    from sound2midi.expression import realtime

    # --list-ports needs no MIDI file, so handle it before argparse's
    # required-positional check.
    if "--list-ports" in (argv if argv is not None else sys.argv[1:]):
        ports = realtime.list_ports()
        if not ports:
            print("No MIDI output destinations found.")
        for i, port in enumerate(ports):
            print(f"[{i}] {port}")
        return 0

    parser = argparse.ArgumentParser(
        prog="sound2midi-mpe-play",
        description="Stream an MPE performance to a DAW over MIDI.",
    )
    _add_common_args(parser)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Render a plain MIDI on the fly instead of expecting a .mpe.mid.",
    )
    parser.add_argument("--loop", action="store_true", help="Loop playback.")
    parser.add_argument(
        "--start", type=float, default=0.0, metavar="SECONDS", help="Start position."
    )
    parser.add_argument(
        "--port",
        action="append",
        default=None,
        metavar="NAME",
        help="ALSO send to existing MIDI destination(s) — hardware and "
        "network devices included (index, name, or unique substring; see "
        "--list-ports). Repeatable: --port ROLI --port iPad; --port all "
        "for every destination. The virtual source stays open alongside "
        "so subscribing apps keep receiving (disable with --no-virtual).",
    )
    parser.add_argument(
        "--no-virtual",
        action="store_true",
        help="Do not publish the virtual source port (avoids doubled notes "
        "when --port already targets an app that also subscribes to sources).",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="List the available MIDI destinations and exit.",
    )
    parser.add_argument(
        "--port-name", default=None, help="Virtual port name (default: sound2midi MPE)."
    )
    args = parser.parse_args(argv)

    if args.live:
        events = build_performance(args).events
    else:
        # Without --live, the file is streamed exactly as it is: a .mpe.mid
        # plays with its baked-in expression (and its own handshake), a plain
        # MIDI plays plain — never silently rendered.
        events, _ = _file_events(args.midi)
        if not args.midi.name.endswith(".mpe.mid"):
            _log(
                "Streaming the file as-is (it carries no MPE expression); "
                "pass --live to render MPE on the fly, or render a .mpe.mid "
                "first with sound2midi-mpe."
            )
        if args.solo_track is not None:
            _log("--solo-track only applies with --live; playing the full file.")

    outputs = realtime.open_outputs(
        args.port_name or realtime.PORT_NAME,
        connect_to=args.port,
        virtual=not args.no_virtual,
    )
    if args.port is None:
        _log(
            "note: only apps subscribed to MIDI sources (a DAW, listener apps) "
            "receive the virtual port; add --port (repeatable, or --port all) to "
            "also reach hardware destinations like a ROLI keyboard or an iPad."
        )
    _log("Streaming (Ctrl-C stops) ...")
    try:
        realtime.stream(
            events,
            outputs,
            bend_range=args.bend_range,
            start=args.start,
            loop=args.loop,
            handshake=args.live,  # file playback sends only what the file contains
            description=args.midi.name,
        )
    except KeyboardInterrupt:
        _log("Stopped.")
    finally:
        for out in outputs:
            # a vanished device must not block closing the rest
            with contextlib.suppress(Exception):
                out.close_port()
    return 0
