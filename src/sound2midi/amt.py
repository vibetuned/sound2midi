"""Manage the tsumugi (ex instrument-agnostic-amt) checkout and run its inference CLI.

The upstream project (https://github.com/anime-song/tsumugi) is a uv workspace, not a
pip-installable package (``[tool.uv] package = false``): its modules are imported from
the checkout root and its torch pin is resolved per platform by ``uv.lock``. We keep it
in its own directory with its own uv-managed virtualenv and call its inference CLI as a
subprocess.

Upstream renamed the project from ``instrument-agnostic-amt`` to ``tsumugi`` and moved
from ``requirements.txt`` + a root ``infer.py`` to ``uv.lock`` + the
``instrument_agnostic_amt.amt.cli.infer`` module; the Python package name is unchanged.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

AMT_REPO_URL = "https://github.com/anime-song/tsumugi.git"
AMT_REF = "main"
# The inference entry point inside the checkout (upstream package layout).
AMT_INFER_MODULE = "instrument_agnostic_amt.amt.cli.infer"
# The pre-rename cache directory, kept only to point users at the stale copy.
LEGACY_AMT_DIRNAME = "instrument-agnostic-amt"

# Upstream checkpoint variants. The ``_v1_5`` / ``_v2`` models are the newer
# retrains and are what the upstream Colab defaults to per stem.
MODEL_TYPES = (
    "default",
    "bass",
    "bass_v2",
    "vocal",
    "guitar",
    "guitar_v1_5",
    "vocal_harmony",
    "vocal_harmony_v1_5",
    "drums",
    "drums_v1_5",
    "other",
    "other_v1_5",
)


# The audio-analysis detectors (skey, beat-this, lv-chordia) used to ride the
# transcriber's venv to reuse its torch. They can no longer: skey pins
# ``torch>=2.7,<2.8`` while tsumugi pins ``torch==2.13``, so sharing one venv
# silently downgrades the transcriber. They now get their own environment,
# which also means they work the same whichever transcriber is selected.
ANALYSIS_PYTHON_VERSION = "3.12"


def default_analysis_home() -> Path:
    """Where the analysis venv lives, overridable via ``SOUND2MIDI_ANALYSIS_HOME``."""
    override = os.environ.get("SOUND2MIDI_ANALYSIS_HOME")
    if override:
        return Path(override).expanduser()
    cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache).expanduser() if cache else Path.home() / ".cache"
    return base / "sound2midi" / "analysis"


def ensure_analysis_env(home: Path, *, reinstall: bool = False) -> Path:
    """Create the analysis venv (no project deps of its own); return its python."""
    venv_dir = home / ".venv"
    python = _venv_python(home)
    if reinstall and venv_dir.exists():
        shutil.rmtree(venv_dir)
    if python.exists():
        return python
    home.mkdir(parents=True, exist_ok=True)
    _run([_uv(), "venv", "--python", ANALYSIS_PYTHON_VERSION, str(venv_dir)])
    return python


def default_amt_home() -> Path:
    """Where the AMT repo + venv live, overridable via ``SOUND2MIDI_AMT_HOME``."""
    override = os.environ.get("SOUND2MIDI_AMT_HOME")
    if override:
        return Path(override).expanduser()
    cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache).expanduser() if cache else Path.home() / ".cache"
    return base / "sound2midi" / "tsumugi"


def _uv() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "`uv` was not found on PATH. Install it from https://docs.astral.sh/uv/."
        )
    return uv


def _ffmpeg_lib_dirs() -> list[str]:
    """Directories holding FFmpeg's shared libraries, for torchcodec on macOS.

    torchaudio 2.11 decodes through torchcodec, which dlopen()s the system
    FFmpeg libraries at decode time. Homebrew keeps those outside the default
    dylib search path, so torchcodec finds none of its supported versions and
    audio loading dies with "Could not load this library".
    """
    candidates: list[str] = []
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:  # .../Cellar/ffmpeg/<version>/bin/ffmpeg -> .../lib
        candidates.append(str(Path(ffmpeg).resolve().parent.parent / "lib"))
    prefix = os.environ.get("HOMEBREW_PREFIX")
    for base in (prefix, "/opt/homebrew", "/usr/local"):
        if base:
            candidates.append(f"{base}/opt/ffmpeg/lib")
            candidates.append(f"{base}/lib")
    seen: list[str] = []
    for directory in candidates:
        if directory not in seen and Path(directory).is_dir():
            seen.append(directory)
    return seen


def _child_env() -> dict[str, str] | None:
    """The environment for model subprocesses (None to inherit unchanged)."""
    if sys.platform != "darwin":
        return None
    dirs = _ffmpeg_lib_dirs()
    if not dirs:
        return None
    env = os.environ.copy()
    existing = env.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    parts = [d for d in dirs if d not in existing.split(":")]
    if existing:
        parts.append(existing)
    env["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(parts)
    return env


def _run(cmd: Sequence[str], *, cwd: Path | None = None) -> None:
    printable = " ".join(str(c) for c in cmd)
    print(f"$ {printable}", file=sys.stderr, flush=True)
    subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        check=True,
        env=_child_env(),
    )


def _venv_python(home: Path) -> Path:
    if os.name == "nt":
        return home / ".venv" / "Scripts" / "python.exe"
    return home / ".venv" / "bin" / "python"


def _default_device() -> str:
    """Default inference device: CUDA where it can exist, CPU on macOS."""
    return "cpu" if sys.platform == "darwin" else "cuda"


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
    legacy = home.parent / LEGACY_AMT_DIRNAME
    if legacy.is_dir():
        print(
            f"note: upstream renamed the project to tsumugi; the pre-rename checkout at "
            f"{legacy} is no longer used and can be deleted.",
            file=sys.stderr,
        )
    _run(["git", "clone", "--depth", "1", "--branch", ref, AMT_REPO_URL, str(home)])
    return home


def ensure_env(home: Path, *, reinstall: bool = False) -> Path:
    """Sync the tsumugi venv from its lockfile; return its python executable.

    Upstream replaced the CUDA-pinned ``requirements.txt`` with ``uv.lock``, whose
    torch index is selected by platform marker (Linux/Windows resolve the CUDA
    wheels, macOS the platform ones), so nothing has to be rewritten here. The
    ``stem`` extra carries stem-splitter/librosa for the stem workflow.
    """
    venv_dir = home / ".venv"
    marker = venv_dir / ".sound2midi-deps-installed"
    python = _venv_python(home)

    if reinstall and venv_dir.exists():
        shutil.rmtree(venv_dir)

    if marker.exists() and python.exists():
        return python

    if not (home / "uv.lock").exists():
        raise FileNotFoundError(
            f"No uv.lock found in the tsumugi checkout at {home}. "
            "Remove that directory so it can be re-cloned."
        )
    _run([_uv(), "sync", "--locked", "--extra", "stem"], cwd=home)

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
    """Run the tsumugi inference CLI on ``audio_path``, writing ``output_midi``.

    ``device`` is passed through when given; omitting it lets upstream pick
    (``auto`` resolves CUDA, then Apple Silicon MPS, then CPU).
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type {model_type!r}; choose from {MODEL_TYPES}.")

    python = setup(home)
    output_midi.parent.mkdir(parents=True, exist_ok=True)

    # ``-m`` with cwd=home: the checkout root is on sys.path, and upstream is a
    # uv workspace whose package is deliberately not installed into the venv.
    cmd: list[str] = [
        str(python),
        "-m",
        AMT_INFER_MODULE,
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
        raise FileNotFoundError(
            f"tsumugi inference finished but no MIDI was written to {output_midi}."
        )
    return output_midi


def _stem_pipeline_script() -> Path:
    """Path to the in-venv stem pipeline script shipped with this package."""
    return Path(__file__).resolve().parent / "_amt" / "stem_pipeline.py"


def ensure_stem_deps(home: Path, *, reinstall: bool = False) -> Path:
    """The stem workflow's deps ride the ``stem`` extra that :func:`ensure_env` syncs."""
    return ensure_env(home, reinstall=reinstall)


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
    low_vram: bool = False,
    predict_velocity: bool = True,
) -> Path:
    """Separate the audio into stems, transcribe each, and merge into ``output_midi``.

    Runs upstream's own stem workflow. Per-stem MIDIs are kept under ``output_root``
    for inspection / playback, and ``predict_velocity`` adds upstream's per-note
    velocity model (real dynamics rather than a constant).

    Separated stems that already exist are reused, so a run interrupted after the
    separation stage picks up from there; ``force`` discards them and starts over.
    If the child dies on a signal (e.g. an intermittent native SIGSEGV in the
    torch stack) it is retried once.
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
    if low_vram:
        cmd += ["--low-vram"]
    if not predict_velocity:
        cmd += ["--no-velocity"]

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
    """Ensure the analysis env plus skey (key detection) are installed."""
    python = ensure_analysis_env(home, reinstall=reinstall)
    marker = home / ".venv" / ".sound2midi-key-deps-installed"
    if marker.exists() and not reinstall:
        return python
    _run([_uv(), "pip", "install", "--python", str(python), *KEY_DEPS])
    marker.write_text("ok\n")
    return python


# Beat/meter detection deps (Beat This!, CPJKU) — plain PyTorch, rides the AMT venv.
METER_DEPS = ("beat-this",)


def _meter_detect_script() -> Path:
    return Path(__file__).resolve().parent / "_amt" / "meter_detect.py"


def ensure_meter_deps(home: Path, *, reinstall: bool = False) -> Path:
    """Ensure the analysis env plus beat-this (meter detection) are installed."""
    python = ensure_analysis_env(home, reinstall=reinstall)
    marker = home / ".venv" / ".sound2midi-meter-deps-installed"
    if marker.exists() and not reinstall:
        return python
    _run([_uv(), "pip", "install", "--python", str(python), *METER_DEPS])
    marker.write_text("ok\n")
    return python


def detect_meter(
    audio_path: Path,
    *,
    home: Path,
    midi_path: Path | None = None,
    device: str | None = None,
    output_json: Path | None = None,
) -> dict:
    """Detect tempo + time signature of ``audio_path``; optionally save the artifact.

    Returns the summary dict, e.g. ``{"time_signature": "4/4", "bpm": 176.5, ...}``.
    The JSON artifact additionally contains the full beat/downbeat grid.
    """
    python = ensure_meter_deps(home)
    cmd: list[str] = [
        str(python),
        str(_meter_detect_script()),
        "--audio",
        str(audio_path.resolve()),
        "--device",
        device or _default_device(),
    ]
    if midi_path is not None and midi_path.exists():
        cmd += ["--midi", str(midi_path.resolve())]
    if output_json is not None:
        cmd += ["--output-json", str(output_json.resolve())]

    print(f"$ {' '.join(cmd)}", file=sys.stderr, flush=True)
    result = subprocess.run(
        [str(c) for c in cmd], capture_output=True, text=True, check=True, env=_child_env()
    )

    for line in reversed(result.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                summary = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "time_signature" in summary:
                return summary
    raise RuntimeError(
        f"meter detection produced no result.\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr[-800:]}"
    )


# lv-chordia (openmirlab's package of the ISMIR 2019 Large-Vocabulary Chord
# Transcription model) — plain pip package, rides the AMT venv's torch.
CHORD_DEPS = ("lv-chordia",)


def _chords_detect_script() -> Path:
    return Path(__file__).resolve().parent / "_amt" / "chords_detect.py"


def ensure_chord_deps(home: Path, *, reinstall: bool = False) -> Path:
    """Ensure the analysis env plus lv-chordia (chord recognition) are installed."""
    python = ensure_analysis_env(home, reinstall=reinstall)
    marker = home / ".venv" / ".sound2midi-chord-deps-installed"
    if marker.exists() and not reinstall:
        return python
    _run([_uv(), "pip", "install", "--python", str(python), *CHORD_DEPS])
    marker.write_text("ok\n")
    return python


def detect_chords(
    audio_path: Path,
    *,
    home: Path,
    output_json: Path | None = None,
    chord_dict: str = "submission",
) -> dict:
    """Detect the song's chord progression with lv-chordia; optionally save JSON.

    Returns the summary dict, e.g. ``{"n_chords": 71, "distinct": 9, ...}``. The
    JSON artifact contains the full labeled chord segments.
    """
    python = ensure_chord_deps(home)
    cmd: list[str] = [
        str(python),
        str(_chords_detect_script()),
        "--audio",
        str(audio_path.resolve()),
        "--chord-dict",
        chord_dict,
    ]
    if output_json is not None:
        cmd += ["--output-json", str(output_json.resolve())]

    print(f"$ {' '.join(cmd)}", file=sys.stderr, flush=True)
    result = subprocess.run(
        [str(c) for c in cmd], capture_output=True, text=True, check=True, env=_child_env()
    )

    for line in reversed(result.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                summary = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "n_chords" in summary:
                return summary
    raise RuntimeError(
        f"chord detection produced no result.\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr[-800:]}"
    )


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
        device or _default_device(),  # skey falls back to CPU if CUDA is unavailable
    ]
    if output_json is not None:
        cmd += ["--output-json", str(output_json.resolve())]

    print(f"$ {' '.join(cmd)}", file=sys.stderr, flush=True)
    result = subprocess.run(
        [str(c) for c in cmd], capture_output=True, text=True, check=True, env=_child_env()
    )

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
