"""Manage the SongFormer checkout/venv and run song-structure detection.

SongFormer (https://github.com/ASLP-lab/SongFormer, ASLP-lab) segments a song
into labeled sections (intro / verse / pre-chorus / chorus / bridge / inst /
outro / silence). Like the AMT project it is a script-based repo, not a pip
package: its inference code imports sibling modules, vendors MusicFM as a git
submodule, and pins ``torch==2.4.0`` — a build with no kernels for Blackwell
(RTX 50) GPUs. It therefore gets the same treatment as the AMT repo: its own
checkout with its own uv-managed venv, where we install the runtime subset of
its requirements against the ``torch==2.7.0+cu128`` build the AMT venv already
uses, and run inference as a subprocess
(``sound2midi/_songformer/sections_detect.py``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from sound2midi.amt import _run, _uv, _venv_python

SONGFORMER_REPO_URL = "https://github.com/ASLP-lab/SongFormer.git"
SONGFORMER_REF = "main"
# Only the musicfm submodule is needed at inference time (MuQ comes from PyPI).
SONGFORMER_SUBMODULE = "src/third_party/musicfm"
# numpy==1.25 (upstream pin, required by the old msaf stack) has no 3.12 wheels.
SONGFORMER_PYTHON_VERSION = "3.10"

# Same torch build as the AMT venv: cu128 has Blackwell (RTX 50) kernels, which
# the upstream torch==2.4.0 pin lacks.
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
TORCH_DEPS = ("torch==2.7.0+cu128", "torchaudio==2.7.0+cu128")

# The runtime subset of the upstream requirements.txt, at the upstream pins:
# training/eval/demo-only packages (lightning, wandb, gradio, pesq, ...) are
# dropped. msaf stays — models/SongFormer.py imports it at module level.
SECTIONS_DEPS = (
    "numpy==1.25.0",
    "scipy==1.15.2",
    "librosa==0.11.0",
    "soundfile==0.13.1",
    "omegaconf==2.3.0",
    "ema-pytorch==0.7.7",
    "x-transformers==2.4.14",
    "muq==0.1.0",
    "msaf==0.1.80",
    "safetensors==0.5.3",
    "einops==0.8.1",
    "transformers==4.51.1",
    "huggingface-hub==0.30.1",
    "tqdm==4.67.1",
    "requests",
)

# Downloaded by the repo's utils/fetch_pretrained.py, relative to src/SongFormer.
CHECKPOINT_FILES = (
    Path("ckpts") / "SongFormer.safetensors",
    Path("ckpts") / "MusicFM" / "msd_stats.json",
    Path("ckpts") / "MusicFM" / "pretrained_msd.pt",
)


def default_songformer_home() -> Path:
    """Where the SongFormer repo + venv live, overridable via ``SOUND2MIDI_SONGFORMER_HOME``."""
    override = os.environ.get("SOUND2MIDI_SONGFORMER_HOME")
    if override:
        return Path(override).expanduser()
    cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache).expanduser() if cache else Path.home() / ".cache"
    return base / "sound2midi" / "songformer"


def _package_dir(home: Path) -> Path:
    """The directory the repo's relative ckpts/configs paths are anchored to."""
    return home / "src" / "SongFormer"


def ensure_repo(home: Path, *, ref: str = SONGFORMER_REF) -> Path:
    """Clone the SongFormer repo (+ musicfm submodule) into ``home`` if needed."""
    if not (home / ".git").exists():
        if home.exists() and any(home.iterdir()):
            raise RuntimeError(
                f"{home} exists but is not a git checkout of the SongFormer repo. "
                "Remove it or set SOUND2MIDI_SONGFORMER_HOME elsewhere."
            )
        home.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--depth", "1", "--branch", ref, SONGFORMER_REPO_URL, str(home)])
    if not (home / SONGFORMER_SUBMODULE / "model").exists():
        _run(
            ["git", "submodule", "update", "--init", "--depth", "1", SONGFORMER_SUBMODULE],
            cwd=home,
        )
    return home


def ensure_env(home: Path, *, reinstall: bool = False) -> Path:
    """Create the SongFormer venv and install its runtime deps; return its python."""
    venv_dir = home / ".venv"
    marker = venv_dir / ".sound2midi-sections-deps-installed"
    python = _venv_python(home)

    if reinstall and venv_dir.exists():
        shutil.rmtree(venv_dir)

    if marker.exists() and python.exists():
        return python

    uv = _uv()
    _run([uv, "venv", "--python", SONGFORMER_PYTHON_VERSION, str(venv_dir)])
    # torch first, from the cu128 index, so nothing pulls the PyPI build transitively.
    _run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--extra-index-url",
            TORCH_INDEX_URL,
            *TORCH_DEPS,
        ]
    )
    _run([uv, "pip", "install", "--python", str(python), *SECTIONS_DEPS])

    marker.write_text("ok\n")
    return python


def ensure_checkpoints(home: Path, python: Path) -> None:
    """Fetch the SongFormer + MusicFM checkpoints (skips files already present)."""
    pkg = _package_dir(home)
    if all((pkg / f).exists() for f in CHECKPOINT_FILES):
        return
    _run([str(python), str(Path("utils") / "fetch_pretrained.py")], cwd=pkg)
    missing = [str(f) for f in CHECKPOINT_FILES if not (pkg / f).exists()]
    if missing:
        raise RuntimeError(f"Checkpoint download finished but files are missing: {missing}")


def setup(home: Path, *, reinstall: bool = False) -> Path:
    """Ensure the SongFormer repo, env, and checkpoints are ready; return venv python."""
    ensure_repo(home)
    python = ensure_env(home, reinstall=reinstall)
    ensure_checkpoints(home, python)
    return python


def _sections_detect_script() -> Path:
    return Path(__file__).resolve().parent / "_songformer" / "sections_detect.py"


def detect_sections(
    audio_path: Path,
    *,
    home: Path,
    device: str | None = None,
    output_json: Path | None = None,
) -> dict:
    """Detect the song's structure; optionally save the artifact JSON.

    Returns the summary dict, e.g. ``{"n_segments": 9, "structure": "intro verse
    chorus ...", ...}``. The JSON artifact contains the full labeled segments.
    """
    python = setup(home)
    cmd: list[str] = [
        str(python),
        str(_sections_detect_script()),
        "--repo",
        str(home),
        "--audio",
        str(audio_path.resolve()),
        "--device",
        device or "cuda",  # the script falls back to CPU if CUDA is unusable
    ]
    if output_json is not None:
        cmd += ["--output-json", str(output_json.resolve())]

    print(f"$ {' '.join(cmd)}", file=sys.stderr, flush=True)
    # stderr passes through so model-download/load progress stays visible; the
    # summary contract is a JSON line on stdout, like the key/meter detectors.
    # As with the stem pipeline, the torch/CUDA stack occasionally dies on a
    # native SIGSEGV; a plain retry is reliable.
    attempts = 2
    for attempt in range(1, attempts + 1):
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True)
            break
        except subprocess.CalledProcessError as exc:
            crashed_on_signal = exc.returncode is not None and exc.returncode < 0
            if crashed_on_signal and attempt < attempts:
                print(
                    f"Section detection died on signal {-exc.returncode} "
                    f"(attempt {attempt}/{attempts}); retrying ...",
                    file=sys.stderr,
                )
                continue
            raise

    for line in reversed(result.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                summary = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "n_segments" in summary:
                return summary
    raise RuntimeError(f"section detection produced no result.\nstdout:\n{result.stdout}")
