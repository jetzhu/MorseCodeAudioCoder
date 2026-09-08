"""Tests for morse.audio_input.

Everything except the final smoke test runs against a fake ``sounddevice``
module installed in ``sys.modules``: audio_input imports sounddevice lazily
inside each function, so the fake is what it sees, and no PortAudio stream is
ever opened.
"""
from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from morse import audio_input
from morse.audio_input import AudioInput, DeviceInfo, list_devices, resolve_device

ROOT = Path(__file__).resolve().parents[1]

HOSTAPIS = (
    {"name": "MME"},
    {"name": "Windows WASAPI"},
    {"name": "Windows WDM-KS"},
)
MME, WASAPI, WDMKS = 0, 1, 2


def _dev(name: str, hostapi: int, ins: int, outs: int = 0, sr: float = 48000.0) -> dict[str, Any]:
    return {
        "name": name,
        "hostapi": hostapi,
        "max_input_channels": ins,
        "max_output_channels": outs,
        "default_samplerate": sr,
    }


# Mirrors the laptop from docs/PLAN.md section 8: the processed default mic on
# MME/WASAPI and the raw "Microphone Array 1" endpoint on WDM-KS.
DEVICES_WITH_ARRAY1 = [
    _dev("Microsoft Sound Mapper - Input", MME, 2, 0, 44100.0),               # 0
    _dev("Microphone Array (Intel Smart Sound Technology)", MME, 2),          # 1 default
    _dev("Speakers (Realtek(R) Audio)", MME, 0, 2),                           # 2
    _dev("Microphone Array (Intel Smart Sound Technology)", WASAPI, 2),       # 3
    _dev("Speakers (Realtek(R) Audio)", WASAPI, 0, 2),                        # 4
    _dev("Microphone Array 1 (Intel Smart Sound Technology)", WDMKS, 2),      # 5
    _dev("Microphone Array 3 (Intel Smart Sound Technology)", WDMKS, 2),      # 6
    _dev("Speakers (Realtek HD Audio output)", WDMKS, 0, 2),                  # 7
]

# Same machine without the raw WDM-KS array endpoint ("Microphone Array 1"
# only exists on MME here, which must not count).
DEVICES_WITHOUT_ARRAY1 = [
    _dev("Microsoft Sound Mapper - Input", MME, 2),                           # 0
    _dev("Microphone Array 1 (Intel Smart Sound Technology)", MME, 2),        # 1 default
    _dev("Speakers (Realtek(R) Audio)", MME, 0, 2),                           # 2
    _dev("Microphone Array (Intel Smart Sound Technology)", WASAPI, 2),       # 3
    _dev("Headset Microphone (USB Audio)", WDMKS, 1),                         # 4
]


class FakeInputStream:
    """Records constructor kwargs and lifecycle calls; never touches audio."""

    instances: list[FakeInputStream] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.started = 0
        self.stopped = 0
        self.closed = 0
        FakeInputStream.instances.append(self)

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def close(self) -> None:
        self.closed += 1


def install_fake_sd(
    monkeypatch: pytest.MonkeyPatch,
    devices: list[dict[str, Any]],
    default_input: int = 1,
    hostapis: tuple[dict[str, Any], ...] = HOSTAPIS,
) -> types.ModuleType:
    """Install a fake ``sounddevice`` module with query_devices/query_hostapis/default."""
    fake = types.ModuleType("sounddevice")

    def query_devices(device: int | None = None, kind: str | None = None) -> Any:
        if device is None:
            return list(devices)
        return devices[device]

    def query_hostapis(index: int | None = None) -> Any:
        if index is None:
            return hostapis
        return hostapis[index]

    fake.query_devices = query_devices  # type: ignore[attr-defined]
    fake.query_hostapis = query_hostapis  # type: ignore[attr-defined]
    fake.default = SimpleNamespace(device=[default_input, -1])  # type: ignore[attr-defined]
    fake.InputStream = FakeInputStream  # type: ignore[attr-defined]
    FakeInputStream.instances.clear()
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    return fake


# -- import hygiene -----------------------------------------------------------


def test_importing_module_does_not_import_sounddevice() -> None:
    code = "import sys, morse.audio_input; print('sounddevice' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


# -- list_devices ---------------------------------------------------------------


def test_list_devices_returns_only_input_capable_with_hostapi_names(monkeypatch) -> None:
    install_fake_sd(monkeypatch, DEVICES_WITH_ARRAY1)
    devices = list_devices()
    assert [d.index for d in devices] == [0, 1, 3, 5, 6]
    assert all(isinstance(d, DeviceInfo) for d in devices)
    assert all(d.max_input_channels > 0 for d in devices)
    by_index = {d.index: d for d in devices}
    assert by_index[0].hostapi == "MME"
    assert by_index[0].default_samplerate == 44100.0
    assert by_index[3].hostapi == "Windows WASAPI"
    assert by_index[5].hostapi == "Windows WDM-KS"
    assert by_index[5].name.startswith("Microphone Array 1")
    assert isinstance(by_index[5].default_samplerate, float)


def test_list_devices_empty_when_no_inputs(monkeypatch) -> None:
    install_fake_sd(monkeypatch, [_dev("Speakers", MME, 0, 2)], default_input=-1)
    assert list_devices() == []


# -- resolve_device -------------------------------------------------------------


def test_resolve_device_int_is_returned_unchanged(monkeypatch) -> None:
    install_fake_sd(monkeypatch, DEVICES_WITH_ARRAY1)
    assert resolve_device(3) == 3
    assert resolve_device(0) == 0


def test_resolve_device_str_prefers_wdm_ks(monkeypatch) -> None:
    install_fake_sd(monkeypatch, DEVICES_WITH_ARRAY1)
    # Matches 1 (MME), 3 (WASAPI), 5 and 6 (WDM-KS); the first WDM-KS wins.
    assert resolve_device("microphone array") == 5
    assert resolve_device("MICROPHONE ARRAY") == 5
    # A more specific name picks the specific WDM-KS device.
    assert resolve_device("array 3") == 6


def test_resolve_device_str_without_wdm_ks_match_takes_first(monkeypatch) -> None:
    install_fake_sd(monkeypatch, DEVICES_WITH_ARRAY1)
    assert resolve_device("sound mapper") == 0
    # 'Intel' also matches 1, 3, 5, 6 -> 5 is WDM-KS; restrict to a name that
    # exists only on MME/WASAPI to exercise the fallback:
    install_fake_sd(monkeypatch, DEVICES_WITHOUT_ARRAY1)
    assert resolve_device("microphone array") == 1


def test_resolve_device_str_ignores_output_only_devices(monkeypatch) -> None:
    install_fake_sd(monkeypatch, DEVICES_WITH_ARRAY1)
    with pytest.raises(ValueError):
        resolve_device("speakers")


def test_resolve_device_str_no_match_raises(monkeypatch) -> None:
    install_fake_sd(monkeypatch, DEVICES_WITH_ARRAY1)
    with pytest.raises(ValueError, match="no such device"):
        resolve_device("no such device")


def test_resolve_device_none_prefers_microphone_array_1_on_wdm_ks(monkeypatch) -> None:
    install_fake_sd(monkeypatch, DEVICES_WITH_ARRAY1, default_input=1)
    assert resolve_device(None) == 5


def test_resolve_device_none_falls_back_to_default_input(monkeypatch) -> None:
    # "Microphone Array 1" exists here but only on MME, so the default wins.
    install_fake_sd(monkeypatch, DEVICES_WITHOUT_ARRAY1, default_input=3)
    assert resolve_device(None) == 3


def test_resolve_device_none_without_default_raises(monkeypatch) -> None:
    install_fake_sd(monkeypatch, DEVICES_WITHOUT_ARRAY1, default_input=-1)
    with pytest.raises(RuntimeError):
        resolve_device(None)


def test_resolve_device_rejects_other_types(monkeypatch) -> None:
    install_fake_sd(monkeypatch, DEVICES_WITH_ARRAY1)
    with pytest.raises(TypeError):
        resolve_device(2.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        resolve_device(True)  # type: ignore[arg-type]


# -- AudioInput: construction -----------------------------------------------------


def test_audio_input_resolves_device_on_construction(monkeypatch) -> None:
    install_fake_sd(monkeypatch, DEVICES_WITH_ARRAY1)
    ai = AudioInput()
    assert ai.device_index == 5
    assert ai.device_name == "Microphone Array 1 (Intel Smart Sound Technology)"
    assert ai.fs == 48000 and ai.block_size == 480
    assert ai.dropped == 0
    assert ai.running is False
    assert FakeInputStream.instances == []  # nothing opened yet

    ai2 = AudioInput(device="sound mapper", fs=44100, block_size=441)
    assert (ai2.device_index, ai2.device_name) == (0, "Microsoft Sound Mapper - Input")


def test_audio_input_bad_index_raises(monkeypatch) -> None:
    install_fake_sd(monkeypatch, DEVICES_WITH_ARRAY1)
    with pytest.raises(ValueError):
        AudioInput(device=99)


# -- AudioInput: callback / queue / dropped -------------------------------------


def _fake_indata(block_size: int, value: float, channels: int = 1) -> np.ndarray:
    data = np.full((block_size, channels), value, dtype=np.float32)
    if channels > 1:
        data[:, 1:] = -1.0  # other channels must be ignored
    return data


def test_callback_queues_a_copy_of_channel_zero(monkeypatch) -> None:
    install_fake_sd(monkeypatch, DEVICES_WITH_ARRAY1)
    ai = AudioInput(device=5, block_size=8, queue_size=4)

    indata = _fake_indata(8, 0.25, channels=2)
    ai._callback(indata, 8, None, None)
    indata[:] = 0.0  # mutate the buffer PortAudio would reuse

    blocks = ai.read_blocks()
    assert len(blocks) == 1
    block = blocks[0]
    assert block.shape == (8,)
    assert block.dtype == np.float32
    np.testing.assert_array_equal(block, np.full(8, 0.25, dtype=np.float32))
    assert ai.dropped == 0
    assert ai.read_blocks() == []  # drained


def test_callback_preserves_order_and_read_blocks_drains(monkeypatch) -> None:
    install_fake_sd(monkeypatch, DEVICES_WITH_ARRAY1)
    ai = AudioInput(device=5, block_size=4, queue_size=10)
    for k in range(5):
        ai._callback(_fake_indata(4, k / 10), 4, None, None)
    blocks = ai.read_blocks()
    assert [float(b[0]) for b in blocks] == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
    assert ai.read_blocks() == []
    assert ai.dropped == 0


def test_callback_counts_dropped_blocks_when_queue_full(monkeypatch) -> None:
    install_fake_sd(monkeypatch, DEVICES_WITH_ARRAY1)
    ai = AudioInput(device=5, block_size=4, queue_size=2)
    for k in range(5):
        ai._callback(_fake_indata(4, k / 10), 4, None, None)
    assert ai.dropped == 3
    blocks = ai.read_blocks()
    assert len(blocks) == 2
    assert [float(b[0]) for b in blocks] == pytest.approx([0.0, 0.1])  # oldest survive
    # After draining, new blocks are accepted again.
    ai._callback(_fake_indata(4, 0.9), 4, None, None)
    assert ai.dropped == 3
    assert len(ai.read_blocks()) == 1


def test_callback_ignores_status_flags(monkeypatch) -> None:
    """The contract: copy indata[:, 0] into the queue and do nothing else.

    PortAudio status flags (input_overflow etc.) must not change what is
    queued, must not count as drops, and must not grow the public surface:
    ``dropped`` is the only counter INTERFACES.md lists.
    """
    install_fake_sd(monkeypatch, DEVICES_WITH_ARRAY1)
    ai = AudioInput(device=5, block_size=4, queue_size=4)
    ai._callback(_fake_indata(4, 0.1), 4, None, SimpleNamespace(input_overflow=True))
    ai._callback(_fake_indata(4, 0.2), 4, None, SimpleNamespace(input_overflow=False))
    ai._callback(_fake_indata(4, 0.3), 4, None, None)
    assert ai.dropped == 0
    assert not hasattr(ai, "overflows")
    blocks = ai.read_blocks()
    assert [float(b[0]) for b in blocks] == pytest.approx([0.1, 0.2, 0.3])


def test_callback_casts_to_float32(monkeypatch) -> None:
    install_fake_sd(monkeypatch, DEVICES_WITH_ARRAY1)
    ai = AudioInput(device=5, block_size=4, queue_size=4)
    ai._callback(np.zeros((4, 1), dtype=np.float64), 4, None, None)
    (block,) = ai.read_blocks()
    assert block.dtype == np.float32


# -- AudioInput: start / stop / context manager -----------------------------------


def test_start_opens_stream_with_contract_parameters(monkeypatch) -> None:
    install_fake_sd(monkeypatch, DEVICES_WITH_ARRAY1)
    ai = AudioInput(device=5, fs=48000, block_size=480)
    ai.start()
    assert ai.running is True
    assert len(FakeInputStream.instances) == 1
    stream = FakeInputStream.instances[0]
    assert stream.kwargs["device"] == 5
    assert stream.kwargs["samplerate"] == 48000
    assert stream.kwargs["blocksize"] == 480
    assert stream.kwargs["channels"] == 1
    assert stream.kwargs["dtype"] == "float32"
    assert stream.kwargs["callback"] == ai._callback
    assert stream.started == 1
    ai.stop()


def test_start_and_stop_are_idempotent(monkeypatch) -> None:
    install_fake_sd(monkeypatch, DEVICES_WITH_ARRAY1)
    ai = AudioInput(device=5)
    ai.stop()  # before start: harmless
    ai.start()
    ai.start()
    assert len(FakeInputStream.instances) == 1
    stream = FakeInputStream.instances[0]
    assert stream.started == 1
    ai.stop()
    ai.stop()
    assert stream.stopped == 1
    assert stream.closed == 1
    assert ai.running is False
    # Restart opens a fresh stream.
    ai.start()
    assert len(FakeInputStream.instances) == 2
    ai.stop()


def test_start_closes_stream_when_start_fails(monkeypatch) -> None:
    install_fake_sd(monkeypatch, DEVICES_WITH_ARRAY1)

    class FailingStream(FakeInputStream):
        def start(self) -> None:
            raise OSError("Device unavailable")

    sys.modules["sounddevice"].InputStream = FailingStream  # type: ignore[attr-defined]
    ai = AudioInput(device=5)
    with pytest.raises(OSError):
        ai.start()
    assert ai.running is False
    assert FakeInputStream.instances[0].closed == 1


def test_context_manager_starts_and_stops(monkeypatch) -> None:
    install_fake_sd(monkeypatch, DEVICES_WITH_ARRAY1)
    with AudioInput(device=5) as ai:
        assert ai.running is True
        assert len(FakeInputStream.instances) == 1
    assert ai.running is False
    stream = FakeInputStream.instances[0]
    assert stream.stopped == 1 and stream.closed == 1


def test_context_manager_stops_on_exception(monkeypatch) -> None:
    install_fake_sd(monkeypatch, DEVICES_WITH_ARRAY1)
    with pytest.raises(KeyError):
        with AudioInput(device=5) as ai:
            raise KeyError("boom")
    assert ai.running is False
    assert FakeInputStream.instances[0].closed == 1


# -- smoke test against the real sounddevice --------------------------------------


def test_real_list_devices_smoke() -> None:
    try:
        import sounddevice  # noqa: F401  (may fail without PortAudio)
    except Exception as exc:  # ImportError or OSError from the PortAudio loader
        pytest.skip(f"sounddevice not importable: {exc}")
    try:
        devices = audio_input.list_devices()
    except Exception as exc:
        pytest.skip(f"sounddevice raised: {exc}")
    assert isinstance(devices, list)
    assert all(isinstance(d, DeviceInfo) for d in devices)
