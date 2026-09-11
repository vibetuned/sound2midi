"""Stem-separated transcription, delegating to tsumugi's own pipeline.

This script is NOT part of the importable ``sound2midi`` package and is not linted
or type-checked with the rest of the project. It is executed by the *tsumugi*
virtualenv (via ``sound2midi.amt.transcribe_stems``) because it imports that
checkout's ``instrument_agnostic_amt`` package, which is a uv workspace and is
deliberately never installed into this project's environment.

Upstream now ships the Colab stem workflow itself, as
``instrument_agnostic_amt.cli.infer_stem.run_stem_separated_transcription``
(separate -> transcribe each stem with its matching model -> optionally reclassify
instruments and predict velocity -> merge). This file is therefore a thin adapter:
run that, then place the merged MIDI where sound2midi expects it.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _bootstrap_amt_repo() -> None:
    """Put the tsumugi checkout on sys.path so its package resolves."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--amt-repo", required=True)
    known, _ = parser.parse_known_args()
    sys.path.insert(0, known.amt_repo)


_bootstrap_amt_repo()

from instrument_agnostic_amt.cli.infer_stem import (  # noqa: E402
    run_stem_separated_transcription,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stem-separated tsumugi transcription.")
    parser.add_argument("--amt-repo", required=True, help="Path to the tsumugi checkout.")
    parser.add_argument("--audio", required=True, help="Input audio file.")
    parser.add_argument("--output-midi", required=True, help="Path for the merged MIDI.")
    parser.add_argument("--output-root", default="stem_outputs", help="Intermediate output dir.")
    parser.add_argument("--device", default=None, help="auto, cuda, mps or cpu.")
    parser.add_argument("--checkpoint", default=None, help="Optional AMT checkpoint override.")
    parser.add_argument("--window-batch-size", type=int, default=4)
    parser.add_argument("--max-midi-melodic-instruments", type=int, default=15)
    parser.add_argument("--merge-onset-ms", type=float, default=20.0)
    parser.add_argument("--no-transcribe-drums", action="store_true", help="Skip the drum stem.")
    parser.add_argument(
        "--cleanup-stems", action="store_true", help="Delete separated stem WAVs when done."
    )
    parser.add_argument(
        "--force", action="store_true", help="Discard cached stems and redo the separation."
    )
    parser.add_argument(
        "--low-vram", action="store_true", help="Keep models on CPU, moving one at a time."
    )
    parser.add_argument(
        "--no-velocity", action="store_true", help="Skip the per-note velocity model."
    )
    parser.add_argument(
        "--refine-instruments", action="store_true", help="Reclassify instruments per stem."
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if args.force and output_root.exists():
        # Upstream reuses whatever separated stems it finds; --force means redo.
        shutil.rmtree(output_root, ignore_errors=True)

    result = run_stem_separated_transcription(
        args.audio,
        checkpoint_path=args.checkpoint,
        output_root=output_root,
        device=args.device or "auto",
        window_batch_size=args.window_batch_size,
        max_midi_melodic_instruments=args.max_midi_melodic_instruments,
        merge_onset_ms=args.merge_onset_ms,
        transcribe_drum_stems=not args.no_transcribe_drums,
        cleanup_separated_stems=args.cleanup_stems,
        low_vram_mode=args.low_vram,
        predict_velocity=not args.no_velocity,
        refine_instruments=args.refine_instruments,
    )

    merged = Path(str(result["merged_midi_path"]))
    output_midi = Path(args.output_midi)
    output_midi.parent.mkdir(parents=True, exist_ok=True)
    if merged.resolve() != output_midi.resolve():
        shutil.copyfile(merged, output_midi)

    print(f"stem_midis_dir={result['stem_midi_dir']}")
    print(f"merged_midi={output_midi}")
    print(f"Merged MIDI written to {output_midi}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
