"""Download audio from YouTube (or any yt-dlp supported URL) and extract it to a file."""

from __future__ import annotations

from pathlib import Path

from yt_dlp import YoutubeDL


def probe_id(url: str) -> str:
    """Resolve the source id without downloading any media (cheap metadata call)."""
    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if not info:
        raise RuntimeError(f"yt-dlp returned no metadata for {url!r}.")
    video_id = info.get("id")
    if not video_id:
        raise RuntimeError(f"yt-dlp did not return an id for {url!r}.")
    return str(video_id)


def download_audio(
    url: str,
    out_dir: Path,
    *,
    audio_format: str = "wav",
    quiet: bool = False,
    overwrite: bool = False,
    expected_id: str | None = None,
) -> Path:
    """Download the best audio stream of ``url`` and extract it to ``out_dir``.

    The file is named after the source id (e.g. ``dQw4w9WgXcQ.wav``). If that file
    already exists it is reused instead of downloading again, unless ``overwrite``.
    Pass ``expected_id`` to skip the metadata probe when the id is already known.

    Returns the path to the extracted audio file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    video_id = expected_id or probe_id(url)
    audio_path = out_dir / f"{video_id}.{audio_format}"
    if audio_path.exists() and not overwrite:
        if not quiet:
            print(f"Using cached audio: {audio_path}")
        return audio_path

    ydl_opts: dict[str, object] = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "overwrites": overwrite,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
                "preferredquality": "0",  # best quality for lossy; ignored for wav
            }
        ],
        "quiet": quiet,
        "no_warnings": quiet,
        "noprogress": quiet,
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Expected extracted audio at {audio_path}, but it was not produced. "
            "Is ffmpeg installed and on PATH?"
        )
    return audio_path
