"""Entry point for the ``sound2midi-play`` MIDI player.

Kept dependency-light so that a missing ``player`` extra yields a friendly message
rather than an import traceback. The Qt window itself lives in ``_window``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

_MISSING_DEPS = (
    "The MIDI player needs extra dependencies (PySide6, pyfluidsynth, mido, numpy).\n"
    "Install them with:\n\n    uv sync --extra player\n"
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sound2midi-play",
        description="Play a MIDI file with per-track solo/mute (separate vs together).",
    )
    parser.add_argument("midi", nargs="?", help="MIDI file to open on launch.")
    parser.add_argument("--soundfont", help="Path to a .sf2/.sf3 soundfont.")
    parser.add_argument(
        "--driver",
        help="FluidSynth audio driver (e.g. pulseaudio, pipewire, alsa, jack).",
    )
    args = parser.parse_args(argv)

    try:
        from sound2midi.player._window import run
    except ImportError as exc:
        print(f"{_MISSING_DEPS}\n(import error: {exc})", file=sys.stderr)
        return 1

    return run(midi=args.midi, soundfont=args.soundfont, driver=args.driver)


if __name__ == "__main__":
    raise SystemExit(main())
