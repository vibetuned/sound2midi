"""Command-line entry point: ``sound2midi <youtube-url-or-audio-file>``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from sound2midi import __version__
from sound2midi import muscriptor as muscriptor_backend
from sound2midi.amt import (
    MODEL_TYPES,
    default_amt_home,
    default_analysis_home,
    detect_chords,
    detect_key,
    detect_meter,
    setup,
    transcribe,
    transcribe_stems,
)
from sound2midi.download import download_audio, probe_id
from sound2midi.sections import default_songformer_home, detect_sections


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sound2midi",
        description="Download a YouTube video's audio and transcribe it to MIDI "
        "with tsumugi or MuScriptor.",
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
        "--transcriber",
        choices=("tsumugi", "muscriptor"),
        default="tsumugi",
        help="Transcription model (default: %(default)s). 'tsumugi' is the "
        "instrument-agnostic Semi-CRF model (ex instrument-agnostic-amt); "
        "'muscriptor' is Kyutai/Mirelo's transformer decoder, whose weights are "
        "gated on Hugging Face under a non-commercial licence.",
    )
    parser.add_argument(
        "-t",
        "--type",
        dest="model_type",
        default=None,
        metavar="MODEL",
        help="Model variant. tsumugi: "
        + ", ".join(MODEL_TYPES)
        + " (default: default). muscriptor: "
        + ", ".join(muscriptor_backend.MODEL_SIZES)
        + " (default: medium), or a safetensors path / hf:// URL.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu", "mps"),
        default=None,
        help="Inference device. Defaults to the model's own auto-detection "
        "(CUDA, then Apple GPU 'mps', then CPU).",
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
        help="Where to keep the tsumugi checkout + venv "
        "(default: $SOUND2MIDI_AMT_HOME or ~/.cache/sound2midi/tsumugi).",
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
    stems.add_argument(
        "--low-vram",
        action="store_true",
        help="With --stems: keep the models in CPU memory and move one at a "
        "time onto the GPU (slower, but fits in less VRAM).",
    )
    stems.add_argument(
        "--no-stem-velocity",
        dest="stem_velocity",
        action="store_false",
        help="With --stems: skip upstream's per-note velocity model.",
    )

    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Build the chosen transcriber's environment, then exit.",
    )
    parser.add_argument(
        "--reinstall",
        action="store_true",
        help="Rebuild the transcriber's venv from scratch.",
    )
    parser.add_argument(
        "--infer-arg",
        dest="infer_args",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra argument forwarded verbatim to the transcriber's CLI. "
        "Repeatable, e.g. --infer-arg=--velocity --infer-arg=110.",
    )
    parser.add_argument(
        "--no-key",
        dest="detect_key",
        action="store_false",
        help="Skip key detection (it runs by default via skey, saved as <song>.key.json).",
    )
    parser.add_argument(
        "--no-meter",
        dest="detect_meter",
        action="store_false",
        help="Skip tempo/time-signature detection (runs by default via beat-this, "
        "saved as <song>.meter.json).",
    )
    parser.add_argument(
        "--chords",
        action="store_true",
        help="Detect the chord progression (lv-chordia, large vocabulary), saved as "
        "<song>.chords.json. Opt-in; installs into the tsumugi venv on first use.",
    )
    parser.add_argument(
        "--sections",
        action="store_true",
        help="Detect song structure (intro/verse/chorus/...) with SongFormer, saved as "
        "<song>.sections.json. Opt-in: first use clones SongFormer into its own cached "
        "venv and downloads several GB of checkpoints.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Re-download and re-transcribe even if the audio / MIDI already exist.",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Reduce output.")
    return parser


def _resolve_model(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    """The model variant for the chosen transcriber, validated with its own set."""
    if args.transcriber == "muscriptor":
        model = args.model_type or muscriptor_backend.DEFAULT_MODEL
        # Sizes are keywords; anything path- or URL-shaped is passed through.
        if model not in muscriptor_backend.MODEL_SIZES and not (
            "/" in model or model.endswith(".safetensors")
        ):
            parser.error(
                f"--type {model!r} is not a MuScriptor model; choose from "
                f"{', '.join(muscriptor_backend.MODEL_SIZES)}, or give a "
                "safetensors path / hf:// URL."
            )
        return model
    model = args.model_type or "default"
    if model not in MODEL_TYPES:
        parser.error(
            f"--type {model!r} is not a tsumugi model; choose from {', '.join(MODEL_TYPES)}."
        )
    return model


def _model_tag(args: argparse.Namespace, model: str) -> str:
    """The per-model output folder name, so runs can be compared side by side.

    Stem mode uses one model per stem rather than a single ``--type``, so it is
    tagged by the mode instead.
    """
    return f"{args.transcriber}-{'stems' if args.stems else model}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    amt_home = args.amt_home or default_amt_home()
    muscriptor_home = muscriptor_backend.default_muscriptor_home()
    # The detectors keep their own venv, so they behave the same for either
    # transcriber (and skey's torch pin cannot disturb the transcriber's).
    analysis_home = default_analysis_home()
    model = _resolve_model(args, parser)
    if args.stems and args.transcriber != "tsumugi":
        parser.error("--stems is a tsumugi workflow (it uses its per-stem models).")

    if args.setup_only:
        if args.transcriber == "muscriptor":
            python = muscriptor_backend.setup(muscriptor_home, reinstall=args.reinstall)
            print(f"MuScriptor ready at {muscriptor_home}\n  venv python: {python}")
        else:
            python = setup(amt_home, reinstall=args.reinstall)
            print(f"tsumugi ready at {amt_home}\n  venv python: {python}")
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

    # The audio and the analysis artifacts are model-independent and stay in the
    # song folder; each model's MIDI (and its stems) gets its own subfolder, so
    # transcribing the same song with another model compares instead of
    # overwriting. -o/--output still names an exact path.
    run_dir = song_dir / _model_tag(args, model)
    if args.output:
        output_midi = args.output
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".stems.mid" if args.stems else ".mid"
        output_midi = run_dir / f"{song_name}{suffix}"

    if output_midi.exists() and not args.force:
        print(f"MIDI already exists: {output_midi}  (use --force to regenerate)")
    else:
        if args.reinstall:
            if args.transcriber == "muscriptor":
                muscriptor_backend.setup(muscriptor_home, reinstall=True)
            else:
                setup(amt_home, reinstall=True)

        if args.transcriber == "muscriptor":
            if not args.quiet:
                print(f"Transcribing with MuScriptor -> {output_midi} (model: {model}) ...")
            muscriptor_backend.transcribe(
                audio_path,
                output_midi,
                home=muscriptor_home,
                model=model,
                device=args.device,
                extra_args=args.infer_args,
                quiet=args.quiet,
            )
        elif args.stems:
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
                output_root=output_midi.parent / "stems",
                low_vram=args.low_vram,
                predict_velocity=args.stem_velocity,
            )
        else:
            if not args.quiet:
                print(f"Transcribing -> {output_midi} (model: {model}) ...")
            transcribe(
                audio_path,
                output_midi,
                home=amt_home,
                model_type=model,
                device=args.device,
                amp=args.amp,
                amp_dtype=args.amp_dtype,
                extra_args=args.infer_args,
                quiet=args.quiet,
            )
        print(f"Done. MIDI written to {output_midi}")

    if args.detect_key:
        key_json = song_dir / "artifacts" / f"{song_name}.key.json"
        if args.force or not key_json.exists():
            if not args.quiet:
                print("Detecting key (skey) ...")
            try:
                key = detect_key(
                    audio_path,
                    home=analysis_home,
                    device=args.device,
                    output_json=key_json,
                )
                print(f"Detected key: {key}  (saved to {key_json})")
            except Exception as exc:  # non-fatal: don't lose the transcription
                print(f"Key detection failed: {exc}", file=sys.stderr)
        elif not args.quiet:
            print(f"Key already detected: {key_json}")

    if args.detect_meter:
        meter_json = song_dir / "artifacts" / f"{song_name}.meter.json"
        if args.force or not meter_json.exists():
            if not args.quiet:
                print("Detecting tempo + time signature (beat-this) ...")
            try:
                meter = detect_meter(
                    audio_path,
                    home=analysis_home,
                    midi_path=output_midi,
                    device=args.device,
                    output_json=meter_json,
                )
                print(
                    f"Detected meter: {meter['time_signature']} at {meter['bpm']} bpm  "
                    f"(saved to {meter_json})"
                )
            except Exception as exc:  # non-fatal: don't lose the transcription
                print(f"Meter detection failed: {exc}", file=sys.stderr)
        elif not args.quiet:
            print(f"Meter already detected: {meter_json}")

    if args.chords:
        chords_json = song_dir / "artifacts" / f"{song_name}.chords.json"
        if args.force or not chords_json.exists():
            if not args.quiet:
                print("Detecting chords (lv-chordia) ...")
            try:
                chords = detect_chords(audio_path, home=analysis_home, output_json=chords_json)
                print(
                    f"Detected {chords['n_chords']} chords "
                    f"({chords['distinct']} distinct)  (saved to {chords_json})"
                )
            except Exception as exc:  # non-fatal: don't lose the transcription
                print(f"Chord detection failed: {exc}", file=sys.stderr)
        elif not args.quiet:
            print(f"Chords already detected: {chords_json}")

    if args.sections:
        sections_json = song_dir / "artifacts" / f"{song_name}.sections.json"
        if args.force or not sections_json.exists():
            if not args.quiet:
                print("Detecting song structure (SongFormer) ...")
            try:
                sections = detect_sections(
                    audio_path,
                    home=default_songformer_home(),
                    device=args.device,
                    output_json=sections_json,
                )
                print(f"Detected sections: {sections['structure']}  (saved to {sections_json})")
            except Exception as exc:  # non-fatal: don't lose the transcription
                print(f"Section detection failed: {exc}", file=sys.stderr)
        elif not args.quiet:
            print(f"Sections already detected: {sections_json}")

    return 0
