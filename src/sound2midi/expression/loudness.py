"""Module F — dynamics extracted from the separated stem WAVs.

The stems at ``output/<id>/stems/`` share the source audio's timeline with the
per-stem MIDIs, so a stem's loudness curve can drive note velocities (and,
opt-in, pressure) with no alignment step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from sound2midi.expression.context import STEM_KEYWORDS, TrackData

WINDOW_S = 0.050
HOP_S = 0.010
ONSET_WINDOW_S = 0.080


@dataclass
class StemCurve:
    """A stem's normalized short-term loudness: ``values[i]`` at ``times[i]``."""

    times: np.ndarray  # window centers, seconds
    values: np.ndarray  # normalized to [0, 1] (5th percentile → 0, 98th → 1)

    def level_at_onset(self, start: float) -> float:
        """Max of the curve over the note's first 80 ms (spec §7.3)."""
        mask = (self.times >= start) & (self.times <= start + ONSET_WINDOW_S)
        if not mask.any():
            index = int(np.clip(np.searchsorted(self.times, start), 0, len(self.values) - 1))
            return float(self.values[index])
        return float(self.values[mask].max())

    def resample(self, t0: float, t1: float, n: int) -> list[float]:
        """The curve resampled to ``n`` points across ``[t0, t1]``."""
        targets = np.linspace(t0, t1, max(2, n))
        return np.interp(targets, self.times, self.values).tolist()


def _normalize(db: np.ndarray) -> np.ndarray:
    finite = db[np.isfinite(db)]
    if finite.size < 4:
        return np.full_like(db, 0.5)
    lo, hi = np.percentile(finite, 5.0), np.percentile(finite, 98.0)
    if hi - lo < 1e-6:
        return np.full_like(db, 0.5)
    return np.clip((np.nan_to_num(db, neginf=lo) - lo) / (hi - lo), 0.0, 1.0)


def analyze_wav(path: Path, *, mode: str = "rms") -> StemCurve:
    """Short-term loudness of one stem: RMS dB (default) or LUFS-S (pyloudnorm)."""
    data, samplerate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)

    if mode == "lufs":
        import pyloudnorm

        block, hop = 0.400, 0.100
        meter = pyloudnorm.Meter(samplerate, block_size=block)
        step, size = int(hop * samplerate), int(block * samplerate)
        times, values = [], []
        for start in range(0, max(1, len(mono) - size), step):
            times.append((start + size / 2) / samplerate)
            values.append(meter.integrated_loudness(mono[start : start + size]))
        return StemCurve(times=np.asarray(times), values=_normalize(np.asarray(values)))

    window, hop = int(WINDOW_S * samplerate), int(HOP_S * samplerate)
    if len(mono) < window:
        mono = np.pad(mono, (0, window - len(mono)))
    squared = np.concatenate(([0.0], np.cumsum(mono.astype(np.float64) ** 2)))
    starts = np.arange(0, len(mono) - window + 1, hop)
    rms = np.sqrt((squared[starts + window] - squared[starts]) / window)
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(rms)
    times = (starts + window / 2) / samplerate
    return StemCurve(times=times, values=_normalize(db))


def find_stem_wavs(stems_dir: Path) -> dict[str, Path]:
    """Map stem keywords to WAV paths under the song's stems directory.

    The separation step writes ``<song>_<stem>.wav``; match on that suffix
    (or a bare ``<stem>.wav``) anywhere below ``stems_dir``.
    """
    found: dict[str, Path] = {}
    if not stems_dir.is_dir():
        return found
    for wav in sorted(stems_dir.rglob("*.wav")):
        stem_name = wav.stem.lower()
        for keyword in STEM_KEYWORDS:
            if stem_name == keyword or stem_name.endswith(f"_{keyword}"):
                found.setdefault(keyword, wav)
                break
    return found


def match_track(
    track: TrackData, stem_wavs: dict[str, Path], *, hint: str | None = None
) -> Path | None:
    """The stem WAV a MIDI track belongs to.

    Matched by the stem name embedded in the track name; ``hint`` (the stem a
    per-stem MIDI file belongs to, from its filename) is the fallback when the
    track names don't carry one.
    """
    lowered = track.name.lower()
    if track.is_drum:
        for keyword in ("drums", "drum"):
            if keyword in stem_wavs:
                return stem_wavs[keyword]
        return None
    for keyword in STEM_KEYWORDS:
        if keyword in lowered and keyword in stem_wavs:
            return stem_wavs[keyword]
    if hint is not None:
        return stem_wavs.get(hint)
    return None
