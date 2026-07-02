"""Detect the musical key of an audio file with deezer/skey.

Runs INSIDE the AMT venv (skey + its torch live there), invoked by
``sound2midi.amt.detect_key``. Not part of this package's lint/type-check surface.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect musical key with skey.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    from skey import detect_key  # installed in the AMT venv

    audio = str(Path(args.audio).resolve())
    if not Path(audio).is_file():
        print(f"Audio file not found: {audio}", file=sys.stderr)
        return 2

    # skey prints "Predicted key for <file>: <key>" to stdout for a single file.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ret = detect_key(audio_path=audio, device=args.device, cli=False)
    out = buf.getvalue()

    key = None
    if isinstance(ret, list) and ret:
        key = str(ret[0])
    if not key:
        match = re.search(r"Predicted key for[^:]*:\s*(.+)", out)
        key = match.group(1).strip() if match else None
    if not key:
        print(f"Could not parse key from skey output:\n{out}", file=sys.stderr)
        return 1

    result = {"key": key, "model": "skey", "audio": audio, "device": args.device}
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))  # the parseable result line for the parent process
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
