# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the self-contained desktop build of morse-console.

One-folder, console-mode build (docs/INTERFACES.md, section "Packaging";
docs/PLAN.md, section 10): the entry script is ``morse/__main__.py``, which
only calls :func:`morse.app.main`, so every command-line flag of
``python -m morse.app`` works on ``morse-console.exe`` as well
(``--list-devices``, ``--wav``, ``--no-ui``, or no arguments for the Qt UI).

Build from the project root::

    .venv/Scripts/python -m PyInstaller --noconfirm --clean packaging/morse-console.spec

The result is ``dist/morse-console/`` (zipped by .github/workflows/release.yml
as ``morse-console-windows-x64.zip``).  Both ``dist/`` and ``build/`` are
ignored by git.

What has to be collected by hand and why:

* ``morse.ui`` is imported through ``importlib.import_module`` in
  ``morse/app.py``, which static analysis cannot see, so every submodule of
  ``morse`` is listed as a hidden import.
* pyqtgraph loads its colour maps, icons and Qt templates from data files and
  picks binding-specific template modules at run time, so the whole package
  is collected with :func:`collect_all`.  The examples are left out: they are
  dead weight and importing them creates a ``QApplication``.
* sounddevice is a single module; the PortAudio DLL lives in the companion
  ``_sounddevice_data`` package and is located at run time through
  ``_sounddevice_data.__path__``, so the DLL for this machine's architecture
  is collected into that same relative directory (``collect_dynamic_libs``)
  together with the package's README (``collect_data_files``).  The wheel
  also carries the DLLs for the other Windows architectures and the macOS
  dylib, which are dropped.
* PySide6 itself needs nothing extra: PyInstaller's own PySide6 hooks bundle
  the Qt libraries and plugins for every ``PySide6.Qt*`` module that is
  imported, including the ``platforms`` plugins (``qwindows.dll`` and
  ``qoffscreen.dll``).

Excluded: the ``tests`` and ``tools`` trees, tkinter and matplotlib (neither
is used; both are picked up by scipy/numpy heuristics otherwise), and the Qt
bindings pyqtgraph supports but this app does not use.  UPX is off because it
corrupts Qt DLLs.

Trimmed after analysis: three Qt pieces that a widgets-only app never loads
but that the Qt hooks bring in through plugin dependencies, about 38 MB in
total.  ``opengl32sw.dll`` is the software OpenGL rasteriser (nothing here
asks Qt for an OpenGL context: pyqtgraph runs with ``useOpenGL`` off and no
QOpenGLWidget is created); the virtual-keyboard input-context plugin drags
Qt6Quick, Qt6Qml* and Qt6VirtualKeyboard along; the PDF image-format plugin
drags Qt6Pdf.  A missing plugin is simply not loaded by Qt.  Every other Qt
library and plugin (platforms, styles, image formats, TLS) stays.
"""
from __future__ import annotations

import os
import platform

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

# SPECPATH is set by PyInstaller to the directory of this file (packaging/).
ROOT = os.path.dirname(SPECPATH)  # noqa: F821  (PyInstaller global)
ENTRY = os.path.join(ROOT, "morse", "__main__.py")
NAME = "morse-console"

# --- morse: every submodule, because morse.ui is imported by name at run time
morse_hidden = collect_submodules("morse")

# --- pyqtgraph: data files, binding templates and submodules, minus examples
pg_datas, pg_binaries, pg_hidden = collect_all(
    "pyqtgraph",
    filter_submodules=lambda name: not name.startswith("pyqtgraph.examples"),
    exclude_datas=["**/examples/*"],
)

# --- sounddevice: the PortAudio DLL for this architecture, next to the module
_PORTAUDIO_SUFFIX = {"amd64": "64bit", "x86_64": "64bit", "arm64": "arm64", "aarch64": "arm64", "x86": "32bit"}
_suffix = _PORTAUDIO_SUFFIX.get(platform.machine().lower(), "64bit")
sd_binaries = [
    (src, dest)
    for src, dest in collect_dynamic_libs("_sounddevice_data")
    if os.path.basename(src).lower() in (f"libportaudio{_suffix}.dll", f"libportaudio{_suffix}-asio.dll")
]
if not sd_binaries:
    raise SystemExit(f"no PortAudio DLL for {platform.machine()} found in the sounddevice wheel")
sd_datas = collect_data_files("_sounddevice_data", includes=["**/README.md"])

a = Analysis(
    [ENTRY],
    pathex=[ROOT],
    binaries=pg_binaries + sd_binaries,
    datas=pg_datas + sd_datas,
    hiddenimports=morse_hidden + pg_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tests",
        "tools",
        "tkinter",
        "_tkinter",
        "matplotlib",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "pyqtgraph.examples",
    ],
    noarchive=False,
    optimize=0,
)

# --- drop Qt pieces that only plugin dependencies pulled in (see the docstring)
_DROP_NAMES = {
    "opengl32sw.dll",  # software OpenGL fallback, 20 MB
    "qtvirtualkeyboardplugin.dll",  # platforminputcontexts plugin; needs Qt6Quick/Qml/VirtualKeyboard
    "qpdf.dll",  # imageformats plugin; needs Qt6Pdf
}
_DROP_PREFIXES = ("qt6quick", "qt6qml", "qt6virtualkeyboard", "qt6pdf")


def _wanted(entry: tuple) -> bool:
    """Keep a TOC entry ``(dest_name, src_name, typecode)`` unless it is on the drop list."""
    base = os.path.basename(entry[0]).lower()
    return base not in _DROP_NAMES and not base.startswith(_DROP_PREFIXES)


a.binaries = [entry for entry in a.binaries if _wanted(entry)]
a.datas = [entry for entry in a.datas if _wanted(entry)]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=NAME,
)
