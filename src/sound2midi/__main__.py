"""Enable `python -m sound2midi`."""

from __future__ import annotations

import sys

from sound2midi.cli import main

if __name__ == "__main__":
    sys.exit(main())
