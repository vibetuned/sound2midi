"""Detect tempo, meter (time signature), and the beat grid of a song.

Runs INSIDE the AMT venv (beat-this + torch live there), invoked by
``sound2midi.amt.detect_meter``. Not part of this package's lint/type-check surface.

Approach:
  1. Beat This! (CPJKU, ISMIR 2024) -> beat and downbeat times.
  2. Meter numerator = mode of beats-per-bar between consecutive downbeats.
  3. Compound-meter test: snap the transcribed MIDI's between-beat onsets to a duple
     vs a triple subdivision grid; if triple clearly wins, the meter is compound
     (2 -> 6/8, 3 -> 9/8, 4 -> 12/8) with the tracked beat as a dotted quarter.
  4. Tempo = 60 / median inter-beat interval (felt-beat bpm).

The artifact JSON includes the full beat/downbeat grid so the exporter can retime
notes to the beat grid (beat-aligned notation).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def vote_numerator(beats: np.ndarray, downbeats: np.ndarray) -> tuple[int, float]:
    """Mode of beats-per-bar across bars, plus the fraction of bars agreeing."""
    if len(downbeats) < 2:
        return 4, 0.0
    counts = []
    for a, b in zip(downbeats[:-1], downbeats[1:]):
        counts.append(int(((beats > a) & (beats < b)).sum()) + 1)
    vals, freq = np.unique(counts, return_counts=True)
    best = int(vals[freq.argmax()])
    confidence = float(freq.max() / freq.sum())
    if best < 2 or best > 12:  # implausible; fall back
        return 4, 0.0
    return best, confidence


def subdivision_votes(
    beats: np.ndarray, onsets: np.ndarray, *, edge: float = 0.12, margin: float = 0.02
) -> tuple[int, int]:
    """Count between-beat onsets better explained by duple vs triple subdivision.

    Onsets within ``edge`` of a beat are ignored (they are on-beat). An onset votes
    only when one grid fits at least ``margin`` (in beat fractions) better.
    """
    duple_grid = np.array([0.25, 0.5, 0.75])
    triple_grid = np.array([1 / 3, 2 / 3])
    duple = triple = 0
    for a, b in zip(beats[:-1], beats[1:]):
        span = b - a
        if span <= 0:
            continue
        interior = onsets[(onsets > a) & (onsets < b)]
        for t in interior:
            phase = (t - a) / span
            if phase < edge or phase > 1 - edge:
                continue
            d_duple = float(np.abs(duple_grid - phase).min())
            d_triple = float(np.abs(triple_grid - phase).min())
            if d_duple + margin < d_triple:
                duple += 1
            elif d_triple + margin < d_duple:
                triple += 1
    return duple, triple


def midi_onsets(midi_path: Path) -> np.ndarray:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    onsets = [n.start for inst in pm.instruments for n in inst.notes]
    return np.asarray(sorted(onsets))


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect tempo + time signature.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--midi", default=None, help="Transcribed MIDI (compound-meter test).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    audio = Path(args.audio).resolve()
    if not audio.is_file():
        print(f"Audio file not found: {audio}", file=sys.stderr)
        return 2

    from beat_this.inference import File2Beats

    f2b = File2Beats(checkpoint_path="final0", device=args.device, dbn=False)
    beats_raw, downbeats_raw = f2b(str(audio))
    beats = np.asarray(beats_raw, dtype=float)
    downbeats = np.asarray(downbeats_raw, dtype=float)

    if len(beats) < 8:
        print(f"Too few beats detected ({len(beats)}); cannot infer meter.", file=sys.stderr)
        return 1

    ibi = np.diff(beats)
    felt_bpm = float(60.0 / np.median(ibi))
    numerator, confidence = vote_numerator(beats, downbeats)

    duple = triple = 0
    compound = False
    if args.midi and Path(args.midi).is_file():
        onsets = midi_onsets(Path(args.midi))
        if len(onsets):
            duple, triple = subdivision_votes(beats, onsets)
            # conservative: require a clear triple majority before calling it compound
            compound = triple > 1.5 * duple and triple >= 20 and 2 <= numerator <= 4

    if compound:
        time_num, time_den = numerator * 3, 8
    else:
        time_num, time_den = numerator, 4

    result = {
        "time_signature": f"{time_num}/{time_den}",
        "numerator": time_num,
        "denominator": time_den,
        "felt_beats_per_bar": numerator,
        "compound": compound,
        "bpm": round(felt_bpm, 2),
        "bar_confidence": round(confidence, 3),
        "subdivision_votes": {"duple": duple, "triple": triple},
        "first_downbeat": round(float(downbeats[0]), 4) if len(downbeats) else None,
        "beats": [round(float(b), 4) for b in beats],
        "downbeats": [round(float(d), 4) for d in downbeats],
        "model": "beat-this",
        "audio": str(audio),
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")

    summary = {k: result[k] for k in ("time_signature", "bpm", "compound", "bar_confidence")}
    summary["n_beats"] = len(beats)
    print(json.dumps(summary))  # parseable result line for the parent process
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
