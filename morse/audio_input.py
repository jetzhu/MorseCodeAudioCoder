"""Microphone capture through ``sounddevice``.

This is the only module in the ``morse`` package that talks to sounddevice /
PortAudio. The import is deferred into the functions that need it so that
importing this module never loads PortAudio or touches audio hardware; that
keeps the DSP and decoder tests hardware-free and lets tests substitute a fake
``sounddevice`` module.

Only one input stream may be open at a time on the target laptop: opening the
default MME endpoint while the WDM-KS endpoint is open fails with "Device
unavailable" (see docs/PLAN.md, section 8).
"""
from __future__ import annotations

import queue
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import numpy as np

# Name fragments used by resolve_device(); see docs/PLAN.md section 8 for why
# the raw WDM-KS endpoint is preferred over the processed default microphone.
PREFERRED_HOSTAPI_FRAGMENT = "WDM-KS"
PREFERRED_DEVICE_FRAGMENT = "Microphone Array 1"


@dataclass
class DeviceInfo:
    """One input-capable PortAudio device."""

    index: int
    name: str
    hostapi: str
    max_input_channels: int
    default_samplerate: float


def _sd() -> Any:
    """Import sounddevice lazily so module import stays hardware-free."""
    import sounddevice  # deliberately local: loads PortAudio

    return sounddevice


def _hostapi_names(sd: Any) -> list[str]:
    return [str(api["name"]) for api in sd.query_hostapis()]


def _all_devices(sd: Any) -> list[dict[str, Any]]:
    """All PortAudio devices (input and output) in index order."""
    return list(sd.query_devices())


def _device_info(index: int, dev: dict[str, Any], hostapis: list[str]) -> DeviceInfo:
    api_index = int(dev.get("hostapi", -1))
    api_name = hostapis[api_index] if 0 <= api_index < len(hostapis) else ""
    return DeviceInfo(
        index=index,
        name=str(dev["name"]),
        hostapi=api_name,
        max_input_channels=int(dev["max_input_channels"]),
        default_samplerate=float(dev["default_samplerate"]),
    )


def list_devices() -> list[DeviceInfo]:
    """Return every input-capable device, with its host API name resolved.

    Indices are the PortAudio device indices (as accepted by
    ``sounddevice.InputStream(device=...)``), so they are not contiguous.
    """
    sd = _sd()
    hostapis = _hostapi_names(sd)
    return [
        _device_info(index, dev, hostapis)
        for index, dev in enumerate(_all_devices(sd))
        if int(dev["max_input_channels"]) > 0
    ]


def _is_preferred_hostapi(info: DeviceInfo) -> bool:
    return PREFERRED_HOSTAPI_FRAGMENT.lower() in info.hostapi.lower()


def _default_input_index(sd: Any) -> int:
    """PortAudio's default input device index, or -1 when there is none."""
    try:
        index = sd.default.device[0]
    except (AttributeError, IndexError, TypeError):
        return -1
    if index is None:
        return -1
    return int(index)


def resolve_device(spec: str | int | None) -> int:
    """Turn a device specification into a PortAudio device index.

    * ``int`` -- returned unchanged (that device index).
    * ``str`` -- case-insensitive substring of the device name, searched over
      input-capable devices only. When several devices match, one whose host
      API name contains ``'WDM-KS'`` wins; otherwise the lowest index wins.
      Raises ``ValueError`` when nothing matches.
    * ``None`` -- a device whose name contains ``'Microphone Array 1'`` on a
      WDM-KS host API if one exists, else the system default input device.
      Raises ``RuntimeError`` when there is no default input device either.
    """
    if isinstance(spec, bool):  # bool is an int subclass; refuse the ambiguity
        raise TypeError("device spec must be int, str or None, not bool")
    if isinstance(spec, int):
        return spec

    devices = list_devices()

    if isinstance(spec, str):
        needle = spec.lower()
        matches = [d for d in devices if needle in d.name.lower()]
        if not matches:
            raise ValueError(
                f"no input device name contains {spec!r}; "
                f"available: {[d.name for d in devices]}"
            )
        preferred = [d for d in matches if _is_preferred_hostapi(d)]
        return (preferred or matches)[0].index

    if spec is None:
        wanted = PREFERRED_DEVICE_FRAGMENT.lower()
        for d in devices:
            if wanted in d.name.lower() and _is_preferred_hostapi(d):
                return d.index
        index = _default_input_index(_sd())
        if index < 0:
            raise RuntimeError("no default input device is available")
        return index

    raise TypeError(f"device spec must be int, str or None, not {type(spec).__name__}")


def _device_name(sd: Any, index: int) -> str:
    devices = _all_devices(sd)
    if not 0 <= index < len(devices):
        raise ValueError(f"device index {index} out of range (0..{len(devices) - 1})")
    return str(devices[index]["name"])


class AudioInput:
    """Mono float32 capture in fixed-size blocks delivered through a queue.

    The PortAudio callback copies ``indata[:, 0]`` into a bounded
    ``queue.Queue`` and does nothing else; when the queue is full the block is
    discarded and ``dropped`` is incremented. The consumer (UI timer or a
    loop) calls :meth:`read_blocks` to drain the queue without blocking.

    The device is resolved in the constructor so that a bad specification
    fails early and ``device_index`` / ``device_name`` are available before
    :meth:`start` is called. Nothing is opened until :meth:`start`.
    """

    def __init__(
        self,
        device: int | str | None = None,
        fs: int = 48000,
        block_size: int = 480,
        queue_size: int = 1000,
    ) -> None:
        self.fs = int(fs)
        self.block_size = int(block_size)
        self.queue_size = int(queue_size)
        self.dropped: int = 0
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=self.queue_size)
        self._stream: Any = None
        self.device_index: int = resolve_device(device)
        self.device_name: str = _device_name(_sd(), self.device_index)

    # -- PortAudio callback -------------------------------------------------

    def _callback(self, indata: np.ndarray, frames: int, time: Any, status: Any) -> None:
        """Copy the first channel into the queue; count drops. Nothing else happens here."""
        column = indata[:, 0] if indata.ndim > 1 else indata
        block = np.array(column, dtype=np.float32)  # always a fresh copy
        try:
            self._queue.put_nowait(block)
        except queue.Full:
            self.dropped += 1

    # -- lifecycle ------------------------------------------------------------

    @property
    def running(self) -> bool:
        """True while the input stream is open."""
        return self._stream is not None

    def start(self) -> None:
        """Open and start the input stream. No-op if already running."""
        if self._stream is not None:
            return
        sd = _sd()
        stream = sd.InputStream(
            device=self.device_index,
            samplerate=self.fs,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        try:
            stream.start()
        except Exception:
            stream.close()
            raise
        self._stream = stream

    def stop(self) -> None:
        """Stop and close the stream. No-op if not running."""
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
        finally:
            stream.close()

    def read_blocks(self) -> list[np.ndarray]:
        """Drain every queued block, oldest first, without blocking."""
        blocks: list[np.ndarray] = []
        get = self._queue.get_nowait
        while True:
            try:
                blocks.append(get())
            except queue.Empty:
                return blocks

    def __enter__(self) -> AudioInput:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()
