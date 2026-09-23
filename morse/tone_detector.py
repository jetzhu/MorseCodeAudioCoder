"""Adaptive ON/OFF tone detector, version 2 (docs/INTERFACES.md, 2026-09-07).

Feed one Goertzel power value (dB) per block.  The detector keeps two level
estimates in the *linear* power domain, a noise level ``N`` fed by OFF blocks
and a signal level ``S`` fed by ON blocks, places a hysteresis band above
``N`` whose height is sized to the tracked signal-to-noise ratio, and turns
the block verdicts into runs of consecutive ON or OFF blocks.  Runs shorter
than ``min_run_blocks`` are folded into their neighbours, and a run is handed
out only once the run after it is long enough that no merge can change it.

Why the levels are linear means
-------------------------------
The Goertzel power of stationary noise in one 10 ms block is exponentially
distributed (chi-square with two degrees of freedom).  In dB that spread has a
standard deviation near 5.6 dB and, over a minute, spikes about 9-10 dB above
the *linear* mean and dips 30-40 dB below it.  A tracker that follows the
minimum (version 1) therefore sits far under the noise and its thresholds
land inside the spread; a linear-domain mean sits exactly where the
statistics are known, so ``min_lift_db = 12`` above it clears every spike a
minute of noise produces (``exp(-10**1.2)`` of blocks, about one in ten
million) while ``min_off_lift_db = 6`` is exceeded by only 2 % of noise
blocks, which is what keeps a single spike from becoming a two-block run.

Departures from the contract text, and why
------------------------------------------
Both fixtures and 60 s Gaussian-noise runs were used to check the contract's
defaults; two of them do not deliver the acceptance list and are changed here:

* ``noise_alpha_up`` and ``noise_alpha_down`` default to the same value,
  0.1, instead of 0.02 up / 0.2 down.  An asymmetric EMA of an exponentially
  distributed input is biased: with 0.2 down and 0.02 up, ``N`` settles about
  4 dB (0.39x) *below* the linear mean, so the ON threshold sits only 8 dB
  above the noise and the OFF threshold at the noise mean.  Measured on real
  Goertzel noise that gave 8-14 false ON runs per minute at every level above
  the silence clamp and three extra runs on the fixtures.  Equal alphas make
  ``N`` an unbiased mean; the rule ``alpha = noise_alpha_down if p < N else
  noise_alpha_up`` is kept for callers who want asymmetry.
* The common alpha is 0.1 rather than 0.02.  Both recordings fade in from
  digital silence over 600-700 ms, twice the 300 ms warm-up, and a 0.02
  tracker lags that ramp by 8-10 dB, which the rising room noise then
  crosses.  At 0.1 the tracker follows the ramp (worst margin 2.3 dB on the
  fixtures), steady-noise jitter of ``N`` stays near 1 dB, and recovery from
  a 25 dB noise jump takes 1-4 s instead of 9 s or more.

Everything else -- lifts, silence clamp, warm-up, stuck-ON timeout,
hysteresis, debounce and finality -- follows the contract literally.  Two
consequences worth knowing: a tone present from the very first block is
learned as noise during the warm-up, and noise that arrives more than
``min_lift_db`` above the tracked level after the warm-up (for example a
long digitally silent lead-in followed by loud room noise) is ON until the
timeout or until two consecutive dips end the run; the design cannot tell
that onset from a tone.
"""
from __future__ import annotations

import math

from morse.runs import Run

__all__ = ["ToneDetector", "TONALITY_MIN_DB", "is_tonal", "tonality_db"]

_DB_FLOOR = 1e-12
"""Added to linear power before ``log10`` (matches ``morse.dsp.Goertzel.power_db``)."""

_WARMUP_ALPHA = 0.3
"""Noise-level smoothing used for every block during the warm-up."""


def _to_db(power: float) -> float:
    """Linear power to dB, ``10*log10(x + 1e-12)``."""
    return 10.0 * math.log10(power + _DB_FLOOR)


def _from_db(power_db: float) -> float:
    """dB to linear power, ``10**(x/10)``."""
    return 10.0 ** (power_db / 10.0)


class _Segment:
    """A mutable run under construction; converted to a frozen Run when emitted."""

    __slots__ = ("on", "blocks")

    def __init__(self, on: bool, blocks: int) -> None:
        self.on = on
        self.blocks = blocks

    def to_run(self, block_ms: float) -> Run:
        return Run(on=self.on, blocks=self.blocks, block_ms=block_ms)


class ToneDetector:
    """Turn a stream of per-block tone powers into debounced ON/OFF runs.

    Parameters
    ----------
    block_ms:
        Duration of one block; copied into every emitted ``Run``.
    on_frac, off_frac:
        Fractions of the tracked signal-to-noise ratio that set the ON and
        OFF lifts above the noise level (``off_frac <= on_frac``).
    min_run_blocks:
        A completed run shorter than this is merged into the runs on either
        side.  A run becomes final once the run after it has reached this
        length.
    min_lift_db, max_lift_db:
        Bounds of the ON lift: ``lift_on = clamp(on_frac * snr, min_lift_db,
        max_lift_db)``.
    min_off_lift_db:
        Lower bound of the OFF lift: ``lift_off = clamp(off_frac * snr,
        min_off_lift_db, max_lift_db - 6)``.
    release_db, hysteresis_db:
        The OFF threshold is ``max(N + lift_off, S - release_db)`` and the ON
        threshold ``max(N + lift_on, lo + hysteresis_db)``: at low SNR both
        are referenced to the noise level, at high SNR to the tracked signal,
        so a loud tone is released as soon as its tail has dropped
        ``release_db`` rather than when it has decayed to the noise lift.
    noise_alpha_up, noise_alpha_down:
        EMA coefficients for the noise level, applied to OFF blocks above and
        below the current level respectively.  Keep them equal unless a bias
        is intended (see the module docstring).
    signal_alpha:
        EMA coefficient for the signal level, applied to ON blocks.
    signal_decay_db:
        How far the signal level sinks toward the noise level per OFF block.
    silence_floor_db:
        The noise level is never tracked below this, so digital silence does
        not pull the thresholds down.
    warmup_blocks:
        Blocks after construction or ``reset()`` during which every verdict
        is OFF and the noise level is smoothed with alpha 0.3.
    max_on_blocks:
        An ON run that reaches this length is ended (the next block is OFF)
        and the noise level is set to the signal level: no Morse mark lasts
        that long, so the level was noise.  With 10 ms blocks the ended run
        is exactly ``max_on_blocks * block_ms`` = 5000 ms long.
    """

    def __init__(
        self,
        block_ms: float = 10.0,
        on_frac: float = 0.6,
        off_frac: float = 0.4,
        min_run_blocks: int = 2,
        min_lift_db: float = 12.0,
        max_lift_db: float = 30.0,
        min_off_lift_db: float = 6.0,
        noise_alpha_up: float = 0.1,
        noise_alpha_down: float = 0.1,
        signal_alpha: float = 0.1,
        signal_decay_db: float = 0.1,
        silence_floor_db: float = -100.0,
        warmup_blocks: int = 30,
        max_on_blocks: int = 500,
        release_db: float = 12.0,
        hysteresis_db: float = 3.0,
    ) -> None:
        if release_db < 0 or hysteresis_db < 0:
            raise ValueError("release_db and hysteresis_db must be non-negative")
        if block_ms <= 0:
            raise ValueError("block_ms must be positive")
        if not 0.0 <= off_frac <= on_frac <= 1.0:
            raise ValueError("need 0 <= off_frac <= on_frac <= 1")
        if min_run_blocks < 1:
            raise ValueError("min_run_blocks must be at least 1")
        if min_lift_db < 0 or min_off_lift_db < 0:
            raise ValueError("min_lift_db and min_off_lift_db must be non-negative")
        if max_lift_db < min_lift_db:
            raise ValueError("max_lift_db must be at least min_lift_db")
        for name, alpha in (
            ("noise_alpha_up", noise_alpha_up),
            ("noise_alpha_down", noise_alpha_down),
            ("signal_alpha", signal_alpha),
        ):
            if not 0.0 < alpha <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if signal_decay_db < 0:
            raise ValueError("signal_decay_db must be non-negative")
        if warmup_blocks < 0:
            raise ValueError("warmup_blocks must be non-negative")
        if max_on_blocks < 1:
            raise ValueError("max_on_blocks must be at least 1")

        self.block_ms = float(block_ms)
        self.on_frac = float(on_frac)
        self.off_frac = float(off_frac)
        self.min_run_blocks = int(min_run_blocks)
        self.min_lift_db = float(min_lift_db)
        self.max_lift_db = float(max_lift_db)
        self.min_off_lift_db = float(min_off_lift_db)
        self.noise_alpha_up = float(noise_alpha_up)
        self.noise_alpha_down = float(noise_alpha_down)
        self.signal_alpha = float(signal_alpha)
        self.signal_decay_db = float(signal_decay_db)
        self.silence_floor_db = float(silence_floor_db)
        self.warmup_blocks = int(warmup_blocks)
        self.max_on_blocks = int(max_on_blocks)
        self.release_db = float(release_db)
        self.hysteresis_db = float(hysteresis_db)

        self._silence_floor = _from_db(self.silence_floor_db)
        self._signal_decay = _from_db(-self.signal_decay_db)

        self._primed = False
        self._state = False
        self._blocks_seen = 0
        self._n = self._silence_floor
        self._s = self._silence_floor
        self._hi = 0.0
        self._lo = 0.0
        # Completed-but-not-yet-final runs followed by the in-progress run.
        # By construction this never holds more than two segments.
        self._segments: list[_Segment] = []
        self._update_thresholds()

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> bool:
        """Current raw ON/OFF verdict (before debouncing)."""
        return self._state

    @property
    def warming_up(self) -> bool:
        """True during the first ``warmup_blocks`` after construction or ``reset()``."""
        return self._blocks_seen < self.warmup_blocks

    @property
    def floor_db(self) -> float:
        """Tracked noise level ``N`` (linear mean of OFF-block power) in dB."""
        return _to_db(self._n)

    @property
    def peak_db(self) -> float:
        """Tracked signal level ``S`` (linear mean of ON-block power) in dB."""
        return _to_db(self._s)

    @property
    def threshold_hi_db(self) -> float:
        """``max(N + lift_on, lo + hysteresis_db)``: power above which an OFF detector switches ON."""
        return self._hi

    @property
    def threshold_lo_db(self) -> float:
        """``max(N + lift_off, S - release_db)``: power below which an ON detector switches OFF."""
        return self._lo

    @property
    def current_run(self) -> Run:
        """The in-progress run (not yet final). Zero blocks before any update."""
        if not self._segments:
            return Run(on=self._state, blocks=0, block_ms=self.block_ms)
        return self._segments[-1].to_run(self.block_ms)

    # --------------------------------------------------------------- feeding
    def update(self, power_db: float, tonal: bool = True) -> list[Run]:
        """Feed one block's tone power in dB.

        Returns the runs that became final during this block, oldest first.
        Usually empty; occasionally one run (or more if the debounce settings
        allow several to complete at once).  A non-finite value is treated as
        digital silence.

        ``tonal`` says whether the block's energy sits in the tone bin (see
        :func:`is_tonal`).  A block that is not tonal, a keyboard click or
        speech, can never switch an OFF detector ON, but it still teaches the
        noise level, so room noise rising after a silent lead-in is followed
        as usual.  While ON the flag is ignored: a click during a mark does
        not chop it.
        """
        p_db = float(power_db)
        if math.isnan(p_db) or p_db == math.inf:
            p_db = -math.inf
        p = _from_db(p_db)

        if not self._primed:
            self._n = max(p, self._silence_floor)
            self._s = self._n
            self._primed = True
            self._update_thresholds()
        warm = self._blocks_seen < self.warmup_blocks
        self._blocks_seen += 1

        # Verdict against the thresholds of the levels as they stand.
        if warm:
            on = False
        else:
            on = self._state
            if on:
                if p_db < self._lo:
                    on = False
            elif p_db > self._hi and tonal:
                on = True
            if on and self._segments:
                current = self._segments[-1]
                if current.on and current.blocks >= self.max_on_blocks:
                    # Stuck ON: no Morse mark lasts this long, so what the
                    # signal tracker learned was the noise level.
                    on = False
                    self._n = self._s

        # Level update: ON blocks feed S, OFF blocks feed N and let S sink.
        if on:
            self._s += self.signal_alpha * (p - self._s)
        else:
            if warm:
                alpha = _WARMUP_ALPHA
            elif p < self._n:
                alpha = self.noise_alpha_down
            else:
                alpha = self.noise_alpha_up
            self._n = max(self._n + alpha * (p - self._n), self._silence_floor)
            self._s = max(self._s * self._signal_decay, self._n)
        self._state = on
        self._update_thresholds()

        segs = self._segments
        if not segs:
            segs.append(_Segment(on, 1))
        elif segs[-1].on == on:
            segs[-1].blocks += 1
        else:
            done = segs[-1]
            if done.blocks < self.min_run_blocks:
                # A glitch: fold it, together with this block, into the run
                # before it (which has the same state as this block because
                # runs alternate). With no run before it, it simply joins
                # the run that is starting now.
                if len(segs) >= 2:
                    segs.pop()
                    segs[-1].blocks += done.blocks + 1
                else:
                    done.on = on
                    done.blocks += 1
            else:
                segs.append(_Segment(on, 1))

        if len(segs) > 1 and segs[-1].blocks >= self.min_run_blocks:
            # The current run can no longer be merged away, so everything
            # before it has its final length.
            out = [seg.to_run(self.block_ms) for seg in segs[:-1]]
            del segs[:-1]
            return out
        return []

    def flush(self) -> list[Run]:
        """Emit everything pending, including the in-progress run.

        The tracked levels, the verdict and the warm-up state are kept so
        that a stream can continue; only the run bookkeeping starts over.
        """
        out = [seg.to_run(self.block_ms) for seg in self._segments]
        self._segments.clear()
        return out

    def reset(self) -> None:
        """Forget everything: levels, verdict, pending runs; restart the warm-up."""
        self._primed = False
        self._state = False
        self._blocks_seen = 0
        self._n = self._silence_floor
        self._s = self._silence_floor
        self._segments.clear()
        self._update_thresholds()

    # -------------------------------------------------------------- internals
    def _update_thresholds(self) -> None:
        # Noise-referenced lifts protect against noise spikes at low SNR;
        # the signal-referenced release (S - release_db) lets go of a loud
        # tone as soon as its room tail has dropped release_db, instead of
        # waiting for the tail to fall all the way to the noise-referenced
        # lift.  hi always sits hysteresis_db above lo.
        floor = _to_db(self._n)
        peak = _to_db(self._s)
        snr = peak - floor
        lift_on = min(self.max_lift_db, max(self.min_lift_db, self.on_frac * snr))
        lift_off = min(self.max_lift_db - 6.0, max(self.min_off_lift_db, self.off_frac * snr))
        lo = max(floor + lift_off, peak - self.release_db)
        self._lo = lo
        self._hi = max(floor + lift_on, lo + self.hysteresis_db)

    def __repr__(self) -> str:
        return (
            f"ToneDetector(state={'ON' if self._state else 'off'}, "
            f"floor={self.floor_db:.1f} dB, peak={self.peak_db:.1f} dB, "
            f"lo={self._lo:.1f} dB, hi={self._hi:.1f} dB, "
            f"warming_up={self.warming_up}, current={self.current_run})"
        )


TONALITY_MIN_DB: float = -15.0
"""Blocks whose tone power sits further below the block's total power than this are broadband."""

_PURE_TONE_OFFSET_DB = 3.0103
"""A pure sine's normalised Goertzel power is 3 dB above its mean square (A^2 vs A^2/2)."""


def tonality_db(power_db: float, level_dbfs: float) -> float:
    """How much of a block's energy sits in the tone bin, in dB; 0 for a pure tone at ``f0``.

    ``power_db`` is the normalised Goertzel power (0 dB = full-scale sine) and
    ``level_dbfs`` is ``10*log10(mean square)`` of the same block.  White noise
    or a click spreads its energy over the whole band, so its 100 Hz bin holds
    about 1/240 of it: around -24 dB here.  A beeper concentrates its energy
    in the bin: around 0 dB, still above -10 dB with a room tail on top.
    """
    return power_db - level_dbfs - _PURE_TONE_OFFSET_DB


def is_tonal(power_db: float, level_dbfs: float, min_tonality_db: float = TONALITY_MIN_DB) -> bool:
    """Whether a block's energy is concentrated in the tone bin: the ``tonal`` flag for ``update``.

    A beeper block scores near 0 dB, white noise or a keyboard click near
    -24 dB; the default threshold sits at -15 dB.  A block that fails is a
    broadband transient: it may teach the detector the noise level but must
    never switch it ON.  Mirrored by ``isTonal`` in ``web/js/detector.js``.
    """
    return tonality_db(power_db, level_dbfs) >= min_tonality_db
