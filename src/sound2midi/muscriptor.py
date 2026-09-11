"""Transcribe with MuScriptor (Kyutai / Mirelo), an alternative to tsumugi.

MuScriptor is a transformer decoder that emits note events directly, published
as the ``muscriptor`` PyPI package in three sizes — ``small`` (103M), ``medium``
(307M, the default) and ``large`` (1.4B). Like the tsumugi checkout it gets its
own uv-managed virtualenv (it pulls its own torch) and is driven as a subprocess.

The weights live on Hugging Face under a CC BY-NC 4.0 (non-commercial) licence
that has to be accepted once per model, and downloading them needs an
authenticated Hugging Face session — see :data:`AUTH_HELP`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from sound2midi.amt import _run, _uv, _venv_python

PACKAGE = "muscriptor"
# Pin the venv to a Python with broad wheel coverage; MuScriptor supports 3.10+
# (and only <=3.12 on Intel macOS, where torch stopped shipping x86_64 wheels).
MUSCRIPTOR_PYTHON_VERSION = "3.12"

# The published model variants (``--model`` accepts these bare size keywords).
MODEL_SIZES = ("small", "medium", "large")
DEFAULT_MODEL = "medium"

# Output formats the CLI can write. ``midi`` is a single file; ``sheets`` is a
# directory of engraved PDFs plus MusicXML and MIDI, and needs MuseScore 4+.
OUTPUT_FORMATS = ("midi", "json", "jsonl", "sheets")

# MuScriptor prints its own (good) download/auth instructions; this only adds
# what it leaves out — that the licence is non-commercial.
AUTH_HELP = (
    "note: MuScriptor's weights are published under CC BY-NC 4.0 (non-commercial "
    "use); accepting the licence on the model page grants access automatically."
)


def default_muscriptor_home() -> Path:
    """Where the MuScriptor venv lives, overridable via ``SOUND2MIDI_MUSCRIPTOR_HOME``."""
    override = os.environ.get("SOUND2MIDI_MUSCRIPTOR_HOME")
    if override:
        return Path(override).expanduser()
    cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache).expanduser() if cache else Path.home() / ".cache"
    return base / "sound2midi" / "muscriptor"


def ensure_env(home: Path, *, reinstall: bool = False) -> Path:
    """Create the MuScriptor venv and install the package; return its python."""
    venv_dir = home / ".venv"
    marker = venv_dir / ".sound2midi-deps-installed"
    python = _venv_python(home)

    if reinstall and venv_dir.exists():
        shutil.rmtree(venv_dir)

    if marker.exists() and python.exists():
        return python

    home.mkdir(parents=True, exist_ok=True)
    uv = _uv()
    _run([uv, "venv", "--python", MUSCRIPTOR_PYTHON_VERSION, str(venv_dir)])
    _run([uv, "pip", "install", "--python", str(python), PACKAGE])

    marker.write_text("ok\n")
    return python


def setup(home: Path, *, reinstall: bool = False) -> Path:
    """Ensure the MuScriptor environment is ready. Returns the venv python path."""
    return ensure_env(home, reinstall=reinstall)


def transcribe(
    audio_path: Path,
    output_midi: Path,
    *,
    home: Path,
    model: str = DEFAULT_MODEL,
    device: str | None = None,
    output_format: str = "midi",
    extra_args: Sequence[str] = (),
    quiet: bool = False,
) -> Path:
    """Run ``muscriptor transcribe`` on ``audio_path``, writing ``output_midi``.

    ``model`` is a size keyword (``small``/``medium``/``large``), a local
    safetensors path, or an ``hf://`` / ``http(s)://`` URL. Omitting ``device``
    lets MuScriptor choose (CUDA, then Apple Silicon MPS, then CPU).
    """
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"Unknown format {output_format!r}; choose from {OUTPUT_FORMATS}.")

    python = setup(home)
    output_midi.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        str(python),
        "-m",
        PACKAGE,
        "transcribe",
        str(audio_path.resolve()),
        "--output",
        str(output_midi.resolve()),
        "--format",
        output_format,
        "--model",
        model,
    ]
    if device:
        cmd += ["--device", device]
    cmd += list(extra_args)

    try:
        _run(cmd)
    except subprocess.CalledProcessError as exc:
        # A gated/unauthenticated download is by far the most common failure, and
        # MuScriptor has already explained it — exit cleanly rather than dumping
        # a traceback over its instructions.
        print(AUTH_HELP, file=sys.stderr)
        raise SystemExit(exc.returncode or 1) from None

    if not output_midi.exists():
        raise FileNotFoundError(f"MuScriptor finished but no output was written to {output_midi}.")
    return output_midi
