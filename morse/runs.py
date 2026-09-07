"""Shared data type: one run of tone ON or OFF, measured in 10 ms blocks."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Run:
    """A stretch of consecutive blocks with the same detector verdict."""

    on: bool
    blocks: int
    block_ms: float = 10.0

    @property
    def ms(self) -> float:
        return self.blocks * self.block_ms

    def __str__(self) -> str:
        return f"{'ON' if self.on else 'off'} {self.ms:.0f} ms"
