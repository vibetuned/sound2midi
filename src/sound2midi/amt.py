"""Manage the vendored instrument-agnostic-amt checkout and run its inference script.

The upstream project (https://github.com/anime-song/instrument-agnostic-amt) is a
script-based repo, not a pip-installable package: ``infer.py`` imports sibling modules
and pins a CUDA-specific build of torch. We therefore keep it in its own directory with
its own uv-managed virtualenv and call ``infer.py`` as a subprocess.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

AMT_REPO_URL = "https://github.com/anime-song/instrument-agnostic-amt.git"
AMT_REF = "main"
# Pin the AMT venv to a Python with broad ML-wheel coverage (torch, miditoolkit, ...).
AMT_PYTHON_VERSION = "3.12"

MODEL_TYPES = (
    "default",
    "bass",
    "vocal",
    "guitar",
    "vocal_harmony",
    "drums",
    "other",
)


def default_amt_home() -> Path:
    """Where the AMT repo + venv live, overridable via ``SOUND2MIDI_AMT_HOME``."""
    override = os.environ.get("SOUND2MIDI_AMT_HOME")
    if override:
        return Path(override).expanduser()
    cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache).expanduser() if cache else Path.home() / ".cache"
    return base / "sound2midi" / "instrument-agnostic-amt"


def _uv() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "`uv` was not found on PATH. Install it from https://docs.astral.sh/uv/."
        )
    return uv


def _run(cmd: Sequence[str], *, cwd: Path | None = None) -> None:
    printable = " ".join(str(c) for c in cmd)
    print(f"$ {printable}", file=sys.stderr, flush=True)
    subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, check=True)


def _venv_python(home: Path) -> Path:
    if os.name == "nt":
        return home / ".venv" / "Scripts" / "python.exe"
    return home / ".venv" / "bin" / "python"


def ensure_repo(home: Path, *, ref: str = AMT_REF) -> Path:
    """Clone the AMT repo into ``home`` if it is not already there."""
    if (home / ".git").exists():
        return home
    if home.exists() and any(home.iterdir()):
        raise RuntimeError(
            f"{home} exists but is not a git checkout of the AMT repo. "
            "Remove it or pass a different --amt-home."
        )
    home.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", "--branch", ref, AMT_REPO_URL, str(home)])
    return home


def ensure_env(home: Path, *, reinstall: bool = False) -> Path:
    """Create the AMT venv and install its requirements; return its python executable."""
    venv_dir = home / ".venv"
    marker = venv_dir / ".sound2midi-deps-installed"
    python = _venv_python(home)

    if reinstall and venv_dir.exists():
        shutil.rmtree(venv_dir)

    if marker.exists() and python.exists():
        return python

    uv = _uv()
    _run([uv, "venv", "--python", AMT_PYTHON_VERSION, str(venv_dir)])

    requirements = home / "requirements.txt"
    if not requirements.exists():
        raise FileNotFoundError(f"No requirements.txt found in AMT repo at {home}.")
    _run([uv, "pip", "install", "--python", str(python), "-r", str(requirements)])

    marker.write_text("ok\n")
    return python


def setup(home: Path, *, ref: str = AMT_REF, reinstall: bool = False) -> Path:
    """Ensure the AMT repo and its environment are ready. Returns the venv python path."""
    ensure_repo(home, ref=ref)
    return ensure_env(home, reinstall=reinstall)


def transcribe(
    audio_path: Path,
    output_midi: Path,
    *,
    home: Path,
    model_type: str = "default",
    device: str | None = None,
    amp: bool = True,
    amp_dtype: str = "bf16",
    extra_args: Sequence[str] = (),
    quiet: bool = False,
) -> Path:
    """Run ``infer.py`` on ``audio_path`` and write MIDI to ``output_midi``."""
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type {model_type!r}; choose from {MODEL_TYPES}.")

    python = setup(home)
    output_midi.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        str(python),
        "infer.py",
        "--audio",
        str(audio_path.resolve()),
        "--output-midi",
        str(output_midi.resolve()),
        "--type",
        model_type,
    ]
    if device:
        cmd += ["--device", device]
    if amp:
        cmd += ["--amp", "--amp-dtype", amp_dtype]
    if quiet:
        cmd += ["--disable-tqdm"]
    cmd += list(extra_args)

    _run(cmd, cwd=home)

    if not output_midi.exists():
        raise FileNotFoundError(f"infer.py finished but no MIDI was written to {output_midi}.")
    return output_midi


# Extra packages the Colab stem workflow needs on top of requirements.txt.
STEM_DEPS = ("stem-splitter", "librosa")


def _stem_pipeline_script() -> Path:
    """Path to the in-venv stem pipeline script shipped with this package."""
    return Path(__file__).resolve().parent / "_amt" / "stem_pipeline.py"


def ensure_stem_deps(home: Path, *, reinstall: bool = False) -> Path:
    """Ensure the AMT env plus the stem-separation extras are installed."""
    python = ensure_env(home, reinstall=reinstall)
    marker = home / ".venv" / ".sound2midi-stem-deps-installed"
    if marker.exists() and not reinstall:
        return python
    _run([_uv(), "pip", "install", "--python", str(python), *STEM_DEPS])
    marker.write_text("ok\n")
    return python


def transcribe_stems(
    audio_path: Path,
    output_midi: Path,
    *,
    home: Path,
    device: str | None = None,
    window_batch_size: int = 4,
    max_midi_melodic_instruments: int = 15,
    merge_onset_ms: float = 20.0,
    transcribe_drums: bool = True,
    cleanup_stems: bool = False,
    force: bool = False,
    output_root: Path | None = None,
) -> Path:
    """Separate the audio into stems, transcribe each, and merge into ``output_midi``.

    Replicates the upstream Colab's stem workflow. Per-stem MIDIs are kept under
    ``output_root`` for inspection / playback.

    The pipeline is resumable: separated stems and per-stem MIDIs that already exist
    are reused (unless ``force``), and if the child process dies on a signal (e.g. an
    intermittent native SIGSEGV in the torch/CUDA stack) it is retried once, picking up
    from the stems already completed.
    """
    ensure_repo(home)
    python = ensure_stem_deps(home)
    script = _stem_pipeline_script()

    if output_root is None:
        output_root = output_midi.parent / f"{output_midi.stem}_stems"
    output_midi.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        str(python),
        str(script),
        "--amt-repo",
        str(home),
        "--audio",
        str(audio_path.resolve()),
        "--output-midi",
        str(output_midi.resolve()),
        "--output-root",
        str(output_root.resolve()),
        "--window-batch-size",
        str(window_batch_size),
        "--max-midi-melodic-instruments",
        str(max_midi_melodic_instruments),
        "--merge-onset-ms",
        str(merge_onset_ms),
    ]
    if device:
        cmd += ["--device", device]
    if not transcribe_drums:
        cmd += ["--no-transcribe-drums"]
    if cleanup_stems:
        cmd += ["--cleanup-stems"]
    if force:
        cmd += ["--force"]

    attempts = 2
    for attempt in range(1, attempts + 1):
        try:
            _run(cmd, cwd=home)
            break
        except subprocess.CalledProcessError as exc:
            crashed_on_signal = exc.returncode is not None and exc.returncode < 0
            if crashed_on_signal and attempt < attempts:
                print(
                    f"Stem pipeline died on signal {-exc.returncode} "
                    f"(attempt {attempt}/{attempts}); retrying, resuming completed stems...",
                    file=sys.stderr,
                )
                continue
            raise

    if not output_midi.exists():
        raise FileNotFoundError(f"Stem pipeline finished but no MIDI was written to {output_midi}.")
    return output_midi


# deezer/skey (key detection) deps, installed into the AMT venv (reuses its torch).
KEY_DEPS = ("nnAudio==0.3.3", "git+https://github.com/deezer/skey.git")


def _key_detect_script() -> Path:
    return Path(__file__).resolve().parent / "_amt" / "key_detect.py"


def ensure_key_deps(home: Path, *, reinstall: bool = False) -> Path:
    """Ensure the AMT env plus skey (key detection) are installed."""
    python = ensure_env(home, reinstall=reinstall)
    marker = home / ".venv" / ".sound2midi-key-deps-installed"
    if marker.exists() and not reinstall:
        return python
    _run([_uv(), "pip", "install", "--python", str(python), *KEY_DEPS])
    marker.write_text("ok\n")
    return python


def detect_key(
    audio_path: Path,
    *,
    home: Path,
    device: str | None = None,
    output_json: Path | None = None,
) -> str:
    """Detect the musical key of ``audio_path`` with skey; optionally save JSON.

    Returns the key label, e.g. ``"C Major"`` or ``"A minor"``.
    """
    python = ensure_key_deps(home)
    cmd: list[str] = [
        str(python),
        str(_key_detect_script()),
        "--audio",
        str(audio_path.resolve()),
        "--device",
        device or "cuda",  # skey falls back to CPU if CUDA is unavailable
    ]
    if output_json is not None:
        cmd += ["--output-json", str(output_json.resolve())]

    print(f"$ {' '.join(cmd)}", file=sys.stderr, flush=True)
    result = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, check=True)

    for line in reversed(result.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                key = json.loads(line).get("key")
            except json.JSONDecodeError:
                continue
            if key:
                return str(key)
    raise RuntimeError(
        f"skey produced no key.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr[-800:]}"
    )
