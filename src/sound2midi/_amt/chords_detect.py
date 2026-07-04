"""Detect the chord progression of a song with lv-chordia.

Runs INSIDE the AMT venv (lv-chordia rides its torch), invoked by
``sound2midi.amt.detect_chords``. Not part of this package's lint/type-check
surface.

lv-chordia (openmirlab) packages the ISMIR 2019 *Large-Vocabulary Chord
Transcription via Chord Structure Decomposition* model (Jiang, Chen, Li, Xia):
an ensemble of 5 networks with HMM decoding, producing Harte-style labels like
``C:maj``, ``A:min7`` or ``C:maj/3`` (inversions as chord degrees), plus ``N``
for no-chord spans.

Quirk: the library resolves relative paths against its own package directory,
so the audio path must be absolute.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect chords with lv-chordia.")
    parser.add_argument("--audio", required=True)
    parser.add_argument(
        "--chord-dict",
        default="submission",
        choices=("submission", "ismir2017", "full"),
        help="Chord vocabulary (default: %(default)s, the large ISMIR 2019 set).",
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    audio = Path(args.audio).resolve()
    if not audio.is_file():
        print(f"Audio file not found: {audio}", file=sys.stderr)
        return 2

    from lv_chordia.chord_recognition import chord_recognition

    raw = chord_recognition(str(audio), chord_dict_name=args.chord_dict)

    chords = [
        {
            "label": str(seg["chord"]),
            "start": round(float(seg["start_time"]), 3),
            "end": round(float(seg["end_time"]), 3),
        }
        for seg in raw
        if float(seg["end_time"]) > float(seg["start_time"])
    ]

    result = {
        "chords": chords,
        "model": "lv-chordia",
        "vocabulary": args.chord_dict,
        "audio": str(audio),
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")

    labeled = [c for c in chords if c["label"] != "N"]
    summary = {
        "n_chords": len(labeled),
        "n_segments": len(chords),
        "distinct": len({c["label"] for c in labeled}),
    }
    print(json.dumps(summary))  # parseable result line for the parent process
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
