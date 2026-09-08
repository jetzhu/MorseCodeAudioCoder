"""``python -m morse``: run the command-line entry point.

Also the entry script of the PyInstaller build (``packaging/morse-console.spec``),
so it does nothing but hand over to :func:`morse.app.main`, the single entry
point named in ``docs/INTERFACES.md``.
"""
from __future__ import annotations

import sys

from morse.app import main

if __name__ == "__main__":
    sys.exit(main())
