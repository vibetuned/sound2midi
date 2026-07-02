"""Command-line entry point: ``sound2midi <youtube-url-or-audio-file>``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from sound2midi import __version__
from sound2midi.amt import (
    MODEL_TYPES,
    default_amt_home,
    detect_key,
    setup,
    transcribe,
    transcribe_stems,
)
from sound2midi.download import download_audio, probe_id


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sound2midi",
        description="Download a YouTube video's audio and transcribe it to MIDI "
        "with instrument-agnostic-amt.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    parser.add_argument(
        "source",
        nargs="?",
        help="YouTube (or other yt-dlp) URL, or a path to a local audio file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Explicit output MIDI path (overrides the output-dir layout).",
    )
    parser.add_argument(
        "-O",
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Base directory; each song gets its own subfolder (default: %(default)s).",
    )
    parser.add_argument(
        "-t",
        "--type",
        dest="model_type",
        choices=MODEL_TYPES,
        default="default",
        help="AMT model variant (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=None,
        help="Inference device. Defaults to CUDA when available.",
    )
    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Disable mixed-precision inference (it is on by default).",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("bf16", "fp16"),
        default="bf16",
        help="Mixed-precision dtype when AMP is enabled (default: %(default)s).",
    )
    parser.add_argument(
        "--audio-format",
        default="wav",
        help="Audio format to extract from the video (default: %(default)s).",
    )
    parser.add_argument(
        "--amt-home",
        type=Path,
        default=None,
        help="Where to keep the AMT checkout + venv "
        "(default: $SOUND2MIDI_AMT_HOME or ~/.cache/sound2midi/...).",
    )
    stems = parser.add_argument_group("stem-separated mode (replicates the Colab workflow)")
    stems.add_argument(
        "--stems",
        action="store_true",
        help="Separate the audio into stems, transcribe each with its matching model, "
        "then merge. Slower and downloads several model checkpoints. The non-stem "
        "result is unaffected, so you can compare.",
    )
    stems.add_argument(
        "--no-drums",
        dest="transcribe_drums",
        action="store_false",
        help="With --stems: skip the drum stem.",
    )
    stems.add_argument(
        "--cleanup-stems",
        action="store_true",
        help="With --stems: delete the separated stem WAVs when done.",
    )

    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Clone the AMT repo and build its environment, then exit.",
    )
    parser.add_argument(
        "--reinstall",
        action="store_true",
        help="Rebuild the AMT venv from scratch.",
    )
    parser.add_argument(
        "--infer-arg",
        dest="infer_args",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra argument forwarded verbatim to infer.py. Repeatable, e.g. "
        "--infer-arg=--velocity --infer-arg=110.",
    )
    parser.add_argument(
        "--no-key",
        dest="detect_key",
        action="store_false",
        help="Skip key detection (it runs by default via skey, saved as <song>.key.json).",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Re-download and re-transcribe even if the audio / MIDI already exist.",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Reduce output.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    amt_home = args.amt_home or default_amt_home()

    if args.setup_only:
        python = setup(amt_home, reinstall=args.reinstall)
        print(f"AMT ready at {amt_home}\n  venv python: {python}")
        return 0

    if not args.source:
        parser.error("a source URL or audio file is required (or use --setup-only).")

    source: str = args.source
    local = Path(source)

    # Everything for one song lives under <output-dir>/<song-name>/.
    if local.exists():
        song_name = local.stem
    else:
        if not args.quiet:
            print(f"Resolving audio: {source} ...")
        song_name = probe_id(source)

    song_dir = args.output.parent if args.output else args.output_dir / song_name
    song_dir.mkdir(parents=True, exist_ok=True)

    if local.exists():
        audio_path = local  # use the user's file in place
        if not args.quiet:
            print(f"Using local audio: {audio_path}")
    else:
        audio_path = download_audio(
            source,
            song_dir,
            audio_format=args.audio_format,
            quiet=args.quiet,
            overwrite=args.force,
            expected_id=song_name,
        )
        if not args.quiet:
            print(f"Audio: {audio_path}")

    if args.output:
        output_midi = args.output
    elif args.stems:
        output_midi = song_dir / f"{song_name}.stems.mid"
    else:
        output_midi = song_dir / f"{song_name}.mid"

    if output_midi.exists() and not args.force:
        print(f"MIDI already exists: {output_midi}  (use --force to regenerate)")
    else:
        if args.reinstall:
            setup(amt_home, reinstall=True)

        if args.stems:
            if not args.quiet:
                print(f"Stem-separated transcription -> {output_midi} ...")
            transcribe_stems(
                audio_path,
                output_midi,
                home=amt_home,
                device=args.device,
                transcribe_drums=args.transcribe_drums,
                cleanup_stems=args.cleanup_stems,
                force=args.force,
                output_root=song_dir / "stems",
            )
        else:
            if not args.quiet:
                print(f"Transcribing -> {output_midi} (model: {args.model_type}) ...")
            transcribe(
                audio_path,
                output_midi,
                home=amt_home,
                model_type=args.model_type,
                device=args.device,
                amp=args.amp,
                amp_dtype=args.amp_dtype,
                extra_args=args.infer_args,
                quiet=args.quiet,
            )
        print(f"Done. MIDI written to {output_midi}")

    if args.detect_key:
        key_json = song_dir / f"{song_name}.key.json"
        if args.force or not key_json.exists():
            if not args.quiet:
                print("Detecting key (skey) ...")
            try:
                key = detect_key(
                    audio_path, home=amt_home, device=args.device, output_json=key_json
                )
                print(f"Detected key: {key}  (saved to {key_json})")
            except Exception as exc:  # non-fatal: don't lose the transcription
                print(f"Key detection failed: {exc}", file=sys.stderr)
        elif not args.quiet:
            print(f"Key already detected: {key_json}")

    return 0
