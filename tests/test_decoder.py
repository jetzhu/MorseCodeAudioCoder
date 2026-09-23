"""Tests for morse.decoder.MorseDecoder using hand-built run sequences."""
from __future__ import annotations

import random

import pytest

from morse.decoder import (
    DEFAULT_DIT_MS,
    DEFAULT_MAX_MARK_MS,
    MorseDecoder,
    _percentile,
    nearest_rank_percentile,
)
from morse.runs import Run
from morse.table import MORSE_TABLE

Pattern = list[tuple[bool, float]]

# --------------------------------------------------------------------- helpers


def text_to_code(text: str) -> str:
    """'SOS X' -> '... --- ... / -..-' using the shared table."""
    words = []
    for word in text.upper().split(" "):
        words.append(" ".join(MORSE_TABLE[ch] for ch in word))
    return " / ".join(words)


def code_to_pattern(code: str, dit_ms: float) -> Pattern:
    """Turn a dot/dash string ('... --- / -..-') into keyed (on, ms) pairs.

    Standard timing: dit 1, dah 3, intra-letter gap 1, letter gap 3, word gap 7.
    """
    pattern: Pattern = []
    for wi, word in enumerate(code.split(" / ")):
        if wi:
            pattern.append((False, 7 * dit_ms))
        for li, letter in enumerate(word.split(" ")):
            if li:
                pattern.append((False, 3 * dit_ms))
            for si, symbol in enumerate(letter):
                if si:
                    pattern.append((False, dit_ms))
                pattern.append((True, dit_ms if symbol == "." else 3 * dit_ms))
    return pattern


def pattern_to_runs(
    pattern: Pattern,
    jitter: float = 0.0,
    offset_ms: float = 0.0,
    seed: int = 0,
    block_ms: float = 10.0,
) -> list[Run]:
    """Distort a keyed pattern and quantise it to detector blocks.

    ``jitter`` is a fraction: each duration is scaled by a uniform factor in
    ``1 +- jitter``. ``offset_ms`` mimics room reverb and detector release:
    marks are lengthened by it and gaps shortened by it.
    """
    rng = random.Random(seed)
    runs: list[Run] = []
    for on, ms in pattern:
        if jitter:
            ms *= 1.0 + rng.uniform(-jitter, jitter)
        ms += offset_ms if on else -offset_ms
        runs.append(Run(on, max(1, round(ms / block_ms)), block_ms))
    return runs


def make_runs(text: str, wpm: float, **kwargs: float) -> list[Run]:
    """Runs for ``text`` keyed at ``wpm``; kwargs go to :func:`pattern_to_runs`."""
    return pattern_to_runs(code_to_pattern(text_to_code(text), 1200.0 / wpm), **kwargs)


def decode_runs(runs: list[Run], flush: bool = True, **decoder_kwargs) -> tuple[str, MorseDecoder]:
    """Feed every run, optionally flush with a long idle, return (emitted, decoder)."""
    dec = MorseDecoder(**decoder_kwargs)
    emitted = ""
    for run in runs:
        emitted += dec.feed(run)
    if flush:
        emitted += dec.idle(1_000_000.0)
    return emitted, dec


PLAN_SECTION_4_RUNS: list[Run] = [
    # Loopback fixture after debounce: marks 110/270 ms, gaps 50/210 ms for
    # keyed 80/240 ms (docs/PLAN.md section 4), i.e. d = 30 ms.
    Run(True, 11), Run(False, 5), Run(True, 11), Run(False, 5), Run(True, 11),
    Run(False, 21),
    Run(True, 27), Run(False, 5), Run(True, 27), Run(False, 5), Run(True, 27),
    Run(False, 21),
    Run(True, 11), Run(False, 5), Run(True, 11), Run(False, 5), Run(True, 11),
]


# ------------------------------------------------------------- helper sanity


def test_make_runs_sos_15wpm_matches_standard_timing():
    runs = make_runs("SOS", 15)
    ms = [(r.on, r.ms) for r in runs]
    assert ms == [
        (True, 80), (False, 80), (True, 80), (False, 80), (True, 80),
        (False, 240),
        (True, 240), (False, 80), (True, 240), (False, 80), (True, 240),
        (False, 240),
        (True, 80), (False, 80), (True, 80), (False, 80), (True, 80),
    ]


def test_make_runs_word_gap_and_reverb_offset():
    runs = make_runs("E E", 10, offset_ms=30)
    assert [(r.on, r.ms) for r in runs] == [(True, 150), (False, 810), (True, 150)]


def test_percentile_is_nearest_rank_and_never_interpolates():
    """The contract's "10th percentile" is the nearest-rank order statistic:
    sort, take the element at 1-based rank ceil(q/100 * n). The values numpy's
    default linear interpolation would give are noted; a port that used them
    would not reproduce dit_ms/offset_ms in vectors.json."""
    assert nearest_rank_percentile is _percentile
    assert _percentile([120, 240, 360], 10) == 120  # interpolated: 144, a dit blended with a dah
    assert _percentile([120, 360], 10) == 120  # interpolated: 144
    assert _percentile(range(1, 11), 10) == 1  # n <= 10: the minimum; interpolated: 1.9
    assert _percentile(range(1, 12), 10) == 2  # ceil(1.1) = 2; interpolated: 2.0
    assert _percentile(range(1, 21), 10) == 2  # ceil(2.0) = 2; interpolated: 2.9
    assert _percentile(range(1, 22), 10) == 3  # ceil(2.1) = 3; interpolated: 3.0
    assert _percentile(range(1, 31), 10) == 3  # the default window; interpolated: 3.9
    assert _percentile([5], 10) == 5
    assert _percentile([7, 1], 10) == 1
    assert _percentile([3.5, 1.5, 2.5], 50) == 2.5  # ceil(1.5) = 2
    assert _percentile([3.5, 1.5, 2.5], 100) == 3.5
    assert _percentile([3.5, 1.5, 2.5], 0) == 1.5  # rank 0 clamps to the minimum
    with pytest.raises(ValueError):
        _percentile([], 10)


def test_percentile_differs_from_numpy_default_where_it_matters():
    np = pytest.importorskip("numpy")
    for values in (range(1, 31), range(1, 11), [120, 360]):
        ours = _percentile(values, 10)
        theirs = float(np.percentile(list(values), 10))
        assert ours != theirs, f"{list(values)}: nearest-rank {ours} vs interpolated {theirs}"


def test_percentile_result_is_always_an_observed_value():
    rng = random.Random(7)
    for n in range(1, 40):
        values = [rng.uniform(50.0, 400.0) for _ in range(n)]
        for q in (0, 10, 50, 90, 100):
            assert _percentile(values, q) in values


# ------------------------------------------------------------ clean decoding


def test_sos_clean_15wpm():
    emitted, dec = decode_runs(make_runs("SOS", 15))
    assert emitted == "SOS "
    assert dec.text == "SOS "
    assert dec.dit_ms == pytest.approx(80.0)
    assert dec.offset_ms == pytest.approx(0.0)
    assert dec.wpm == pytest.approx(15.0)
    assert dec.letter_count == 3
    assert dec.unknown_count == 0
    assert dec.buffer == ""


def test_hello_world_8wpm_with_word_spacing():
    emitted, dec = decode_runs(make_runs("HELLO WORLD", 8))
    assert emitted == "HELLO WORLD "
    assert dec.dit_ms == pytest.approx(150.0)
    assert dec.letter_count == 10


def test_feed_returns_only_new_text_and_text_accumulates():
    dec = MorseDecoder()
    pieces = [dec.feed(run) for run in make_runs("SOS SOS", 15)]
    pieces.append(dec.idle(10_000))
    assert sum(p.count("S") for p in pieces) == 4, "each letter is returned exactly once"
    assert "".join(pieces) == dec.text == "SOS SOS "
    # marks and intra-letter gaps return nothing; a word gap returns letter + space
    assert pieces[:5] == ["", "", "", "", ""]
    assert all(p in ("", "S", "O", "S ") for p in pieces)


@pytest.mark.parametrize("text", ["HELLO WORLD", "PARIS PARIS PARIS", "SOS SOS SOS"])
@pytest.mark.parametrize("seed", range(20))
def test_jitter_20_percent_12wpm_still_decodes(text: str, seed: int):
    emitted, dec = decode_runs(make_runs(text, 12, jitter=0.2, seed=seed))
    assert emitted.strip() == text
    # M is the 10th percentile of jittered dits, so it sits below the true dit
    assert 0.75 * 100 <= dec.dit_ms <= 1.05 * 100


def test_digits_and_punctuation_decode():
    # 'C' opens with a dah. The first runs are held back until both mark
    # classes have been seen, so no WPM seed is needed any more (it still works).
    emitted, _ = decode_runs(make_runs("CQ DE W1AW 73 K.", 15, jitter=0.1, seed=3), wpm=15)
    assert emitted.strip() == "CQ DE W1AW 73 K."
    emitted, _ = decode_runs(make_runs("CQ DE W1AW 73 K.", 15, jitter=0.1, seed=3))
    assert emitted.strip() == "CQ DE W1AW 73 K."
    emitted, _ = decode_runs(make_runs("ITS 73 DE W1AW K.", 15, jitter=0.1, seed=3))
    assert emitted.strip() == "ITS 73 DE W1AW K."


# ------------------------------------------------------- hold-back and dah rule


@pytest.mark.parametrize("text", ["OSO", "TEST", "MOM", "MORSE CODE", "OK", "0 TO 9", "TT EEE"])
@pytest.mark.parametrize("wpm", [8, 15, 25])
def test_messages_opening_with_dah_letters_decode_without_a_seed(text: str, wpm: int):
    emitted, dec = decode_runs(make_runs(text, wpm))
    assert emitted.strip() == text
    assert dec.dit_ms == pytest.approx(1200 / wpm, rel=0.06)


@pytest.mark.parametrize("wpm", [2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("text", ["SOS HELLO", "HELLO WORLD", "TEST 123"])
def test_slow_keying_down_to_2_wpm(text: str, wpm: int):
    emitted, dec = decode_runs(make_runs(text, wpm))
    assert emitted.strip() == text
    assert dec.dit_ms == pytest.approx(1200 / wpm, rel=0.06)


def test_runs_are_held_back_until_the_estimate_is_trusted():
    dec = MorseDecoder()
    runs = make_runs("SOS", 3)  # 400 ms dits: the old seed read the first one as a dah
    assert dec.timing_ready is False
    # dit, gap, dit: two equal marks, one class. Not trusted yet, nothing emitted,
    # but the held runs read as two dits under the interim estimate.
    out = "".join(dec.feed(r) for r in runs[:3])
    assert out == "" and dec.buffer == "" and dec.timing_ready is False
    assert dec.pending_count == 3
    assert dec.provisional == ".."
    # gap, third dit: three marks are enough. Everything replays into the buffer.
    out = "".join(dec.feed(r) for r in runs[3:5])
    assert dec.timing_ready is True and dec.pending_count == 0
    assert out == "" and dec.buffer == "..."
    assert dec.provisional == ""
    # The letter gap closes S; the first dah of O is classified as it arrives.
    assert dec.feed(runs[5]) == "S"
    assert dec.feed(runs[6]) == "" and dec.buffer == "-"
    assert dec.dit_ms == pytest.approx(400.0)


def test_three_marks_of_one_class_are_enough_to_trust():
    dec = MorseDecoder()
    out = "".join(dec.feed(r) for r in make_runs("EI", 10))  # dits only: E I
    assert dec.timing_ready is True  # trusted at the 3rd mark
    out += dec.idle(1_000_000)
    assert out.strip() == "EI"


def test_idle_flushes_held_runs_with_the_best_estimate():
    # A lone mark: the seed decides (below 300 ms is E, else T).
    emitted, _ = decode_runs(make_runs("E", 15))
    assert emitted == "E "
    emitted, _ = decode_runs(make_runs("T", 8))
    assert emitted == "T "
    # One dit-only letter: dit rule.
    emitted, _ = decode_runs(make_runs("S", 3))
    assert emitted == "S "
    # One dah-only letter: marks are 3x the intra gaps, the dah rule applies.
    emitted, dec = decode_runs(make_runs("O", 15))
    assert emitted == "O "
    assert dec.dit_ms == pytest.approx(80.0)


def test_held_runs_wait_for_a_slow_letter_gap_before_flushing():
    # E (600 ms) then a letter gap (1800 ms) then E at 2 WPM. The old rule
    # flushed after 7 x 150 = 1050 ms of silence and called the dit a dah.
    dec = MorseDecoder()
    e_mark, gap, e_mark2 = make_runs("EE", 2)
    assert dec.feed(e_mark) == ""
    assert dec.idle(1700.0) == ""  # limit is max(7 x 150, 3.5 x 600) = 2100 ms
    assert dec.feed(gap) == ""
    out = dec.feed(e_mark2)
    assert dec.timing_ready is False  # two equal marks: still one class
    out += dec.idle(1_000_000)
    assert out == "EE "


def test_overlong_mark_clears_held_runs_too():
    dec = MorseDecoder()
    for r in make_runs("S", 3)[:3]:  # dit, gap, dit: still held
        dec.feed(r)
    assert dec.pending_count == 3
    assert dec.feed(Run(True, 600)) == ""  # 6 s: not a symbol
    assert dec.pending_count == 0 and dec.buffer == ""
    assert dec.unknown_count == 0


def test_adopt_timing_transfers_speed_and_offset():
    _, source = decode_runs(make_runs("PARIS PARIS", 15, offset_ms=30))
    fresh = MorseDecoder()
    assert fresh.timing_ready is False
    fresh.adopt_timing(source)
    assert fresh.timing_ready is True
    assert fresh.dit_ms == pytest.approx(source.dit_ms)
    assert fresh.offset_ms == pytest.approx(source.offset_ms)
    assert fresh.text == "" and fresh.buffer == ""
    # A fixed-speed decoder keeps its seed but takes the measured offset.
    manual = MorseDecoder(wpm=10, adaptive=False)
    manual.adopt_timing(source)
    assert manual.dit_ms == pytest.approx(120.0)
    assert manual.offset_ms == pytest.approx(source.offset_ms)
    # The adopted history means the next letter decodes without a hold-back.
    out = "".join(fresh.feed(r) for r in make_runs("SOS", 15, offset_ms=30))
    out += fresh.idle(1_000_000)
    assert out.strip() == "SOS"


def test_reset_clears_held_runs_and_trust():
    dec = MorseDecoder()
    for r in make_runs("SO", 15):
        dec.feed(r)
    assert dec.timing_ready is True
    dec.reset()
    assert dec.timing_ready is False and dec.pending_count == 0
    dec2 = MorseDecoder()
    for r in make_runs("SO", 15):
        dec2.feed(r)
    dec2.reset(keep_timing=True)
    assert dec2.timing_ready is True and dec2.dit_ms == pytest.approx(80.0)


# ------------------------------------------------- parity vectors (web port)


@pytest.mark.parametrize(
    "runs, text, dit_ms, offset_ms",
    [
        # INTERFACES.md parity cases whose dit_ms/offset_ms depend on the
        # percentile method. Exact equality on purpose: web/js/decoder.js must
        # reproduce these numbers from vectors.json. An interpolated 10th
        # percentile (numpy default) would give the values in the comments.
        pytest.param(
            make_runs("HELLO WORLD", 12, jitter=0.2, seed=1),
            "HELLO WORLD ", 85.0, 5.0,  # interpolated: 89.5 / 0.5
            id="jitter-20pct-12wpm-seed1-hello-world",
        ),
        pytest.param(
            make_runs("PARIS PARIS PARIS", 12, jitter=0.2, seed=1),
            "PARIS PARIS PARIS ", 80.0, 0.0,  # interpolated: 89.0 / 0.0
            id="jitter-20pct-12wpm-seed1-paris-x3",
        ),
        pytest.param(
            make_runs("HELLO WORLD", 15, offset_ms=30, jitter=0.1, seed=1),
            "HELLO WORLD ", 70.0, 30.0,  # interpolated: 74.5 / 34.5
            id="reverb-30ms-15wpm-jitter-10pct",
        ),
        # Method-independent cases, pinned so the vector set stays complete.
        pytest.param(make_runs("SOS", 15), "SOS ", 80.0, 0.0, id="sos-15wpm-clean"),
        pytest.param(make_runs("HELLO WORLD", 8), "HELLO WORLD ", 150.0, 0.0, id="hello-world-8wpm"),
        pytest.param(make_runs("SOS", 15, offset_ms=30), "SOS ", 80.0, 30.0, id="reverb-30ms-15wpm"),
    ],
)
def test_parity_vector_timing_is_pinned_to_nearest_rank(runs, text, dit_ms, offset_ms):
    emitted, dec = decode_runs(runs)
    assert emitted == text
    assert dec.dit_ms == dit_ms
    assert dec.offset_ms == offset_ms


def test_speed_change_parity_text_is_pinned_to_nearest_rank():
    # While the 12 WPM dits (100 ms) drain out of the window the 6 WPM dits
    # (200 ms) sit exactly on the 2T boundary, so the transition text depends
    # on the percentile method: nearest-rank reads the last transitional
    # letters as 'TTE', an interpolated percentile as 'TI'.
    emitted, dec = decode_runs(_speed_change_runs(12, 6, "PARIS PARIS PARIS PARIS"))
    # The slowdown resync re-locks three marks after the change; before it the
    # 200 ms dits read as dahs and the transitional text depends on nearest rank.
    assert emitted == "PARIS PARIS TTNARIS PARIS PARIS PARIS "
    assert (dec.dit_ms, dec.offset_ms) == (200.0, 0.0)


# ---------------------------------------------------------- reverb correction


def test_reverb_offset_30ms_15wpm_decodes_and_estimates_true_dit():
    emitted, dec = decode_runs(make_runs("SOS", 15, offset_ms=30))
    assert emitted == "SOS "
    assert abs(dec.dit_ms - 80.0) <= 0.10 * 80.0
    assert dec.offset_ms == pytest.approx(30.0)
    assert dec.wpm == pytest.approx(15.0, rel=0.1)


def test_plan_section_4_measured_runs_decode_to_sos():
    emitted, dec = decode_runs(PLAN_SECTION_4_RUNS)
    assert emitted == "SOS "
    assert dec.dit_ms == pytest.approx(80.0)
    assert dec.offset_ms == pytest.approx(30.0)


def test_reverb_offset_with_words_and_jitter():
    emitted, dec = decode_runs(make_runs("HELLO WORLD", 15, offset_ms=30, jitter=0.1, seed=1))
    assert emitted.strip() == "HELLO WORLD"
    assert 20 <= dec.offset_ms <= 40


def test_reverb_correction_is_needed_for_plan_runs():
    """PLAN section 4: with T taken as the shortest mark (110 ms) and no offset,
    the 210 ms letter gaps fall below 2T = 220 ms and SOS collapses into one
    unknown symbol."""
    dec = MorseDecoder(wpm=1200 / 110, adaptive=False)
    dec._update_timing = lambda: None  # type: ignore[method-assign]  # freeze T=110, d=0
    assert dec.dit_ms == pytest.approx(110.0) and dec.offset_ms == 0.0
    emitted = "".join(dec.feed(r) for r in PLAN_SECTION_4_RUNS) + dec.idle(10_000)
    assert emitted == "? "
    assert dec.unknown_count == 1


# ------------------------------------------------------------- speed changes


def _speed_change_runs(first_wpm: float, second_wpm: float, second_text: str) -> list[Run]:
    slow_dit = 1200.0 / min(first_wpm, second_wpm)
    return (
        make_runs("PARIS PARIS", first_wpm)
        + [Run(False, round(7 * slow_dit / 10))]
        + make_runs(second_text, second_wpm)
    )


def test_speed_change_12_to_6_wpm_default_window_recovers():
    runs = _speed_change_runs(12, 6, "PARIS PARIS PARIS PARIS")
    emitted, dec = decode_runs(runs)
    assert emitted.startswith("PARIS PARIS "), "text before the change is intact"
    assert emitted.endswith(" PARIS PARIS "), "estimate re-locks on the new speed"
    assert dec.dit_ms == pytest.approx(200.0)
    assert dec.wpm == pytest.approx(6.0)


def test_speed_change_12_to_6_wpm_small_window_recovers_within_two_letters():
    # A halving of speed puts the old dits exactly on the new 2T boundary, so
    # the 10th percentile has to shed them; with a short window that takes
    # only a couple of letters.
    runs = _speed_change_runs(12, 6, "PARIS PARIS PARIS")
    emitted, dec = decode_runs(runs, window=6)
    assert emitted.startswith("PARIS PARIS ")
    assert emitted.endswith("RIS PARIS PARIS "), emitted
    assert dec.dit_ms == pytest.approx(200.0)


def test_speed_change_6_to_12_wpm_recovers_within_one_letter():
    runs = _speed_change_runs(6, 12, "PARIS PARIS PARIS")
    emitted, dec = decode_runs(runs)
    assert emitted.startswith("PARIS PARIS ")
    assert emitted.endswith("ARIS PARIS PARIS "), emitted
    assert dec.dit_ms == pytest.approx(100.0)


# ----------------------------------------------------------- unknown symbols


def test_unknown_symbol_emits_question_mark_and_counts():
    runs = pattern_to_runs(code_to_pattern("...... / ...", 1200 / 15))
    emitted, dec = decode_runs(runs)
    assert emitted == "? S "
    assert dec.unknown_count == 1
    assert dec.letter_count == 2
    assert "?" in dec.text


def test_unknown_symbol_flushed_by_idle_counts_too():
    runs = pattern_to_runs(code_to_pattern("...... ", 1200 / 15)[:-1])
    dec = MorseDecoder()
    for run in runs:
        dec.feed(run)
    # need a gap estimate before idle uses the measured dit; the intra gaps gave one
    assert dec.idle(10_000) == "? "
    assert dec.unknown_count == 1


# ---------------------------------------------------------------- idle flush


def test_idle_flushes_last_letter_and_adds_space_exactly_once():
    dec = MorseDecoder()
    emitted = "".join(dec.feed(run) for run in make_runs("SOS", 15))
    assert emitted == "SO"
    assert dec.buffer == "..."
    assert dec.dit_ms == pytest.approx(80.0)
    # 7 * 80 = 560 ms: the flush needs off_ms + offset_ms strictly greater
    assert dec.idle(560) == ""
    assert dec.buffer == "..."
    assert dec.idle(570) == "S "
    assert dec.text == "SOS "
    assert dec.buffer == ""
    # further idle calls emit nothing and add no more spaces
    assert dec.idle(600) == ""
    assert dec.idle(5000) == ""
    assert dec.text == "SOS "
    # when the long OFF run is eventually finalised it adds no second space
    assert dec.feed(Run(False, 500)) == ""
    assert dec.text == "SOS "
    assert dec.letter_count == 3


def test_idle_threshold_uses_offset_correction():
    dec = MorseDecoder()
    for run in make_runs("SOS", 15, offset_ms=30):
        dec.feed(run)
    assert dec.offset_ms == pytest.approx(30.0)
    assert dec.dit_ms == pytest.approx(80.0)
    # measured OFF is 30 ms short of the true gap: 530 + 30 = 560 is not > 560
    assert dec.idle(530) == ""
    assert dec.idle(540) == "S "


def test_idle_with_empty_buffer_does_nothing():
    dec = MorseDecoder()
    assert dec.idle(100_000) == ""
    assert dec.text == ""
    for run in make_runs("E", 15):
        dec.feed(run)
    assert dec.idle(10_000) == "E "
    assert dec.idle(20_000) == ""
    assert dec.text == "E "


# ------------------------------------------------------------ wpm and timing


def test_wpm_property_equals_1200_over_dit_ms():
    dec = MorseDecoder()
    assert dec.dit_ms == DEFAULT_DIT_MS == 150.0
    assert dec.wpm == pytest.approx(1200 / 150)
    dec = MorseDecoder(wpm=10)
    assert dec.dit_ms == pytest.approx(120.0)
    assert dec.wpm == pytest.approx(10.0)
    _, dec = decode_runs(make_runs("PARIS", 20))
    assert dec.dit_ms == pytest.approx(60.0)
    assert dec.wpm == pytest.approx(1200 / dec.dit_ms) == pytest.approx(20.0)


def test_fallback_timing_until_two_marks_and_one_gap():
    dec = MorseDecoder()
    assert (dec.dit_ms, dec.offset_ms) == (150.0, 0.0)
    dec.feed(Run(True, 8))  # one mark
    assert (dec.dit_ms, dec.offset_ms) == (150.0, 0.0)
    dec.feed(Run(False, 5))  # one gap, still only one mark
    assert (dec.dit_ms, dec.offset_ms) == (150.0, 0.0)
    dec.feed(Run(True, 8))  # second mark: estimate M=80, G=50
    assert dec.dit_ms == pytest.approx(65.0)
    assert dec.offset_ms == pytest.approx(15.0)

    seeded = MorseDecoder(wpm=8)
    assert seeded.dit_ms == pytest.approx(150.0)
    seeded.feed(Run(True, 30))
    assert seeded.dit_ms == pytest.approx(150.0)


def test_negative_offset_falls_back_to_t_equals_m():
    dec = MorseDecoder()
    dec.feed(Run(True, 8))
    dec.feed(Run(False, 24))  # gap longer than the marks
    dec.feed(Run(True, 8))
    assert dec.dit_ms == pytest.approx(80.0)
    assert dec.offset_ms == 0.0


def test_adaptive_false_with_wpm_keeps_dit_fixed_and_adapts_offset():
    dec = MorseDecoder(wpm=10, adaptive=False)
    assert dec.dit_ms == pytest.approx(120.0)
    emitted = ""
    for run in make_runs("PARIS", 12):  # sent at 12 WPM, decoder pinned to 10
        emitted += dec.feed(run)
        assert dec.dit_ms == pytest.approx(120.0)
    emitted += dec.idle(10_000)
    assert emitted == "PARIS "
    assert dec.wpm == pytest.approx(10.0)

    reverb = MorseDecoder(wpm=10, adaptive=False)
    emitted = ""
    for run in make_runs("PARIS", 10, offset_ms=30):
        emitted += reverb.feed(run)
        assert reverb.dit_ms == pytest.approx(120.0)
    emitted += reverb.idle(10_000)
    assert emitted == "PARIS "
    assert reverb.offset_ms == pytest.approx(30.0), "only d adapts"


def test_adaptive_false_without_wpm_still_adapts():
    _, dec = decode_runs(make_runs("PARIS", 15), adaptive=False)
    assert dec.dit_ms == pytest.approx(80.0)


def test_wpm_seed_is_overridden_when_adaptive():
    emitted, dec = decode_runs(make_runs("PARIS", 15), wpm=10)
    assert emitted == "PARIS "
    assert dec.dit_ms == pytest.approx(80.0)


def test_thresholds_at_exact_boundaries():
    dec = MorseDecoder(wpm=10, adaptive=False)  # T = 120 fixed, d stays 0 here
    assert dec.feed(Run(True, 12)) == ""  # 120 < 240: dit
    assert dec.feed(Run(False, 12)) == ""  # intra-letter
    assert dec.feed(Run(True, 24)) == ""  # exactly 2T: dah
    assert dec.buffer == ".-"
    assert dec.feed(Run(False, 24)) == "A"  # exactly 2T ends the letter
    assert dec.feed(Run(True, 12)) == ""
    assert dec.feed(Run(False, 60)) == "E "  # exactly 5T ends the word
    assert dec.offset_ms == 0.0


def test_window_limits_the_timing_history():
    dec = MorseDecoder(window=4)
    for run in make_runs("SOS", 15):
        dec.feed(run)
    assert dec.dit_ms == pytest.approx(80.0)
    # eight slow dits and seven slow gaps push the fast runs out of a 4-run window
    for run in make_runs("HH", 6):
        dec.feed(run)
    assert dec.dit_ms == pytest.approx(200.0)
    assert dec.offset_ms == 0.0


# ------------------------------------------------------- gaps, spaces, reset


def test_leading_gap_emits_nothing_and_is_ignored_for_timing():
    dec = MorseDecoder()
    assert dec.feed(Run(False, 3000)) == ""
    assert dec.feed(Run(False, 50)) == ""
    assert dec.text == ""
    assert (dec.dit_ms, dec.offset_ms) == (150.0, 0.0)
    emitted = "".join(dec.feed(r) for r in make_runs("SOS", 15)) + dec.idle(10_000)
    assert emitted == "SOS "
    assert dec.dit_ms == pytest.approx(80.0)


def test_word_gap_adds_exactly_one_space():
    emitted, _ = decode_runs(make_runs("SOS SOS", 15))
    assert emitted == "SOS SOS "
    dec = MorseDecoder(wpm=15)  # seeded: no hold-back, the lone E is classified at once
    for run in make_runs("E", 15):
        dec.feed(run)
    assert dec.feed(Run(False, 100)) == "E "
    assert dec.feed(Run(False, 100)) == ""  # a second long gap adds no space
    assert dec.text == "E "


def test_very_long_gaps_do_not_disturb_the_dit_estimate():
    runs: list[Run] = []
    for word in ("SOS", "SOS", "SOS"):
        runs += make_runs(word, 15) + [Run(False, 1000)]  # 10 s of silence
    emitted, dec = decode_runs(runs)
    assert emitted == "SOS SOS SOS "
    assert dec.dit_ms == pytest.approx(80.0)
    assert dec.offset_ms == 0.0


def test_reset_clears_text_and_optionally_timing():
    _, dec = decode_runs(make_runs("SOS", 15, offset_ms=30))
    assert dec.text == "SOS "
    dec.reset(keep_timing=True)
    assert dec.text == "" and dec.buffer == ""
    assert dec.letter_count == 0 and dec.unknown_count == 0
    assert dec.dit_ms == pytest.approx(80.0)
    assert dec.offset_ms == pytest.approx(30.0)
    dec.reset()
    assert (dec.dit_ms, dec.offset_ms) == (150.0, 0.0)
    emitted = "".join(dec.feed(r) for r in make_runs("E", 15)) + dec.idle(10_000)
    assert emitted == "E "

    seeded = MorseDecoder(wpm=10)
    seeded.feed(Run(True, 8))
    seeded.feed(Run(False, 8))
    seeded.feed(Run(True, 8))
    assert seeded.dit_ms == pytest.approx(80.0)
    seeded.reset()
    assert seeded.dit_ms == pytest.approx(120.0)


def test_constructor_validation():
    with pytest.raises(ValueError):
        MorseDecoder(wpm=0)
    with pytest.raises(ValueError):
        MorseDecoder(wpm=-5)
    with pytest.raises(ValueError):
        MorseDecoder(window=1)


# ---------------------------------------------------------------- long marks
# INTERFACES.md: a mark longer than max_mark_ms (default 5000) is not a symbol.
# It is ignored, the pending buffer is cleared, nothing is emitted and
# unknown_count is not incremented, so a held button or a detector timeout
# never produces '?'. The rule is strict ("longer than"): a mark of exactly
# max_mark_ms is still decoded as a dah.

LETTER_GAP_15WPM = Run(False, 24)  # 3 dits at 15 WPM = 240 ms
WORD_GAP_15WPM = Run(False, 56)  # 7 dits at 15 WPM = 560 ms
SIX_SECOND_MARK = Run(True, 600)


def test_long_mark_between_letters_is_ignored_and_next_letter_survives():
    runs = (
        make_runs("SO", 15)
        + [LETTER_GAP_15WPM, SIX_SECOND_MARK, LETTER_GAP_15WPM]
        + make_runs("S", 15)
    )
    dec = MorseDecoder()
    pieces = [dec.feed(run) for run in runs]
    pieces.append(dec.idle(10_000))
    emitted = "".join(pieces)
    assert emitted == "SOS "
    assert dec.text == "SOS "
    assert "?" not in dec.text
    assert dec.unknown_count == 0
    assert dec.letter_count == 3
    assert pieces[runs.index(SIX_SECOND_MARK)] == ""
    # the 6 s mark left the timing estimate exactly where SOS at 15 WPM puts it
    assert dec.dit_ms == pytest.approx(80.0)
    assert dec.offset_ms == 0.0


def test_long_mark_clears_pending_buffer_without_counting_unknown():
    dec = MorseDecoder()
    # six dits in one letter: would decode to '?' if the letter were closed
    for run in pattern_to_runs(code_to_pattern("......", 1200 / 15)):
        dec.feed(run)
    assert dec.buffer == "......"
    assert dec.feed(SIX_SECOND_MARK) == ""
    assert dec.buffer == ""
    assert dec.text == ""
    assert dec.unknown_count == 0
    assert dec.letter_count == 0
    # nothing is pending, so neither the gap nor an idle flush produces '?'
    assert dec.feed(LETTER_GAP_15WPM) == ""
    assert dec.idle(10_000) == ""
    assert dec.text == ""
    # and the message continues normally afterwards
    emitted = "".join(dec.feed(r) for r in make_runs("E", 15)) + dec.idle(10_000)
    assert emitted == "E "
    assert dec.unknown_count == 0


def test_long_mark_is_excluded_from_the_timing_window():
    # window=2: marks [80, 240] and gaps [80, 80] give T = 80. Had the 6 s mark
    # entered the window it would have displaced the dit: marks [240, 6000]
    # give M = 240 and T = 160.
    dec = MorseDecoder(window=2)
    for run in (Run(True, 8), Run(False, 8), Run(True, 24), Run(False, 8)):
        dec.feed(run)
    assert dec.dit_ms == pytest.approx(80.0)
    dec.feed(SIX_SECOND_MARK)
    assert dec.dit_ms == pytest.approx(80.0)
    assert dec.offset_ms == 0.0


def test_long_mark_after_a_word_gap_adds_no_second_space():
    runs = make_runs("SOS", 15) + [WORD_GAP_15WPM, SIX_SECOND_MARK, WORD_GAP_15WPM] + make_runs("SOS", 15)
    emitted, dec = decode_runs(runs)
    assert emitted == "SOS SOS "
    assert dec.unknown_count == 0
    assert dec.letter_count == 6


def test_long_mark_while_idle_leaves_nothing_to_flush():
    dec = MorseDecoder()
    for run in make_runs("S", 15):
        dec.feed(run)
    assert dec.buffer == "..." and dec.pending_count == 0  # trusted at the third dit
    dec.feed(SIX_SECOND_MARK)
    assert dec.buffer == ""
    assert dec.idle(float("inf")) == ""
    assert dec.text == ""


def test_leading_long_mark_emits_nothing_and_message_decodes_afterwards():
    dec = MorseDecoder()
    assert dec.feed(SIX_SECOND_MARK) == ""
    assert dec.feed(Run(False, 100)) == ""
    assert dec.text == ""
    assert (dec.dit_ms, dec.offset_ms) == (150.0, 0.0)
    emitted = "".join(dec.feed(r) for r in make_runs("SOS", 15)) + dec.idle(10_000)
    assert emitted == "SOS "
    assert dec.dit_ms == pytest.approx(80.0)


def test_beeper_fixture_marks_up_to_3700ms_are_still_symbols():
    # tests/fixtures/beeper_long_2491hz_1m.wav after the detector: ON runs of
    # 3700, 490, 570 and 2020 ms with gaps of 370, 250 and 300 ms. None is
    # longer than max_mark_ms, so all four are decoded (the 3.7 s run as a
    # dah; the others as dits and a dah against T = 430 ms, i.e. "-..-").
    # The text is not Morse; what matters is that nothing is dropped.
    runs = [
        Run(True, 370), Run(False, 37), Run(True, 49), Run(False, 25),
        Run(True, 57), Run(False, 30), Run(True, 202),
    ]
    dec = MorseDecoder()
    assert dec.feed(runs[0]) == ""
    assert dec.pending_count == 1, "3.7 s is held back, not ignored"
    emitted = "".join(dec.feed(r) for r in runs[1:]) + dec.idle(float("inf"))
    assert emitted == "X "
    assert dec.letter_count == 1
    assert dec.unknown_count == 0


@pytest.mark.parametrize(
    "blocks, expected",
    [
        pytest.param(37, "T", id="3700ms-beeper"),
        pytest.param(202, "T", id="2020ms-beeper"),
        pytest.param(499, "T", id="4990ms"),
        pytest.param(500, "", id="5000ms-exactly-max-mark-ms-is-ignored"),
        pytest.param(501, "", id="5010ms-one-block-over-is-ignored"),
        pytest.param(600, "", id="6000ms"),
        pytest.param(100_000, "", id="1000s"),
    ],
)
def test_marks_up_to_max_mark_ms_decode_and_longer_ones_are_ignored(blocks: int, expected: str):
    assert DEFAULT_MAX_MARK_MS == 5000.0
    dec = MorseDecoder(wpm=15, adaptive=False)  # T = 80 fixed, d = 0
    assert dec.max_mark_ms == 5000.0
    assert dec.feed(Run(True, blocks)) == ""
    assert dec.buffer == ("-" if expected else "")
    assert dec.feed(LETTER_GAP_15WPM) == expected
    assert dec.text == expected
    assert dec.unknown_count == 0
    assert dec.letter_count == len(expected)


def test_max_mark_ms_is_a_constructor_keyword_and_a_settable_attribute():
    dec = MorseDecoder(wpm=15, adaptive=False, max_mark_ms=1000)
    assert dec.max_mark_ms == 1000.0
    assert dec.feed(Run(True, 99)) == "" and dec.buffer == "-"  # 990 ms: a dah
    assert dec.feed(Run(True, 100)) == "" and dec.buffer == ""  # exactly 1000 ms: ignored, buffer cleared
    dec.max_mark_ms = 2000.0
    assert dec.feed(Run(True, 100)) == "" and dec.buffer == "-"
    assert dec.feed(Run(True, 200)) == "" and dec.buffer == ""
    assert dec.unknown_count == 0 and dec.letter_count == 0

    unlimited = MorseDecoder(wpm=15, adaptive=False, max_mark_ms=float("inf"))
    unlimited.feed(Run(True, 6_000_000))
    assert unlimited.buffer == "-"


def test_max_mark_ms_survives_reset():
    dec = MorseDecoder(max_mark_ms=1234.5)
    dec.reset()
    assert dec.max_mark_ms == 1234.5
    dec.reset(keep_timing=True)
    assert dec.max_mark_ms == 1234.5


@pytest.mark.parametrize("bad", [0, -1, -5000.0, float("nan")])
def test_max_mark_ms_must_be_positive(bad: float):
    with pytest.raises(ValueError):
        MorseDecoder(max_mark_ms=bad)


# ------------------------------------------------------------- glitch rejection


def test_glitch_marks_after_a_message_are_ignored():
    # Keyboard clicks after HELLO WORLD at 8 WPM (T = 150): 20 to 50 ms marks far
    # apart. They used to become E and, worse, shrink the dit estimate so that
    # everything after read as dahs and word gaps.
    noise = [
        Run(False, 200), Run(True, 3), Run(False, 200), Run(True, 4),
        Run(False, 300), Run(True, 2), Run(False, 200), Run(True, 5), Run(False, 200),
    ]
    emitted, dec = decode_runs(make_runs("HELLO WORLD", 8) + noise)
    assert emitted == "HELLO WORLD "
    assert dec.letter_count == 10 and dec.unknown_count == 0
    assert dec.dit_ms == pytest.approx(150.0) and dec.offset_ms == 0.0


def test_glitch_inside_a_letter_is_ignored_with_its_short_gaps():
    # S at 8 WPM with a 30 ms click in the middle of the second intra gap
    # (50 + 30 + 70 = 150 ms). The click and the 50 ms gap are dropped; the
    # 70 ms gap is inside the letter.
    dit = Run(True, 15)
    glitched_s = [dit, Run(False, 15), dit, Run(False, 5), Run(True, 3), Run(False, 7), dit]
    emitted, dec = decode_runs(make_runs("HELLO", 8) + [Run(False, 105)] + glitched_s)
    assert emitted == "HELLO S "
    assert dec.dit_ms == pytest.approx(150.0, rel=0.15)


def test_glitch_rule_only_applies_once_the_estimate_is_trusted():
    # Before trust a short mark is evidence like any other (it may be a fast dit).
    dec = MorseDecoder()
    assert dec.feed(Run(True, 3)) == ""
    assert dec.pending_count == 1
    emitted, dec = decode_runs(make_runs("SOS", 40))  # 30 ms dits are legitimate at 40 WPM
    assert emitted.strip() == "SOS"


def test_provisional_reading_of_held_runs():
    dec = MorseDecoder()
    o_runs = make_runs("O", 8)  # 450 ms dahs
    assert dec.feed(o_runs[0]) == ""
    assert dec.provisional == "-"  # against the 150 ms seed a 450 ms mark is a dah
    dec.reset()
    e_runs = make_runs("E", 15)
    dec.feed(e_runs[0])
    assert dec.provisional == "."
    dec2 = MorseDecoder()
    for r in make_runs("EE", 2):  # 600 ms dits, 1800 ms letter gap: two marks, still held
        dec2.feed(r)
    assert dec2.timing_ready is False
    assert dec2.provisional == ". ."  # letter gap shown as a space


def test_a_real_speed_up_of_three_times_is_accepted_through_the_valve():
    # 5 WPM (240 ms dits) then 15 WPM (80 ms dits): the new dits are under
    # 0.4 x 240 = 96 ms and would all be rejected as clicks. Four in a row with
    # Morse spacing open the valve, the window adapts and the text recovers.
    runs = make_runs("PARIS PARIS", 5) + [Run(False, 168)] + make_runs("PARIS PARIS PARIS", 15)
    emitted, dec = decode_runs(runs)
    assert emitted.startswith("PARIS PARIS ")
    assert emitted.endswith("PARIS "), emitted
    assert dec.dit_ms == pytest.approx(80.0, rel=0.1)


# ------------------------------------------------------------- slowdown resync


@pytest.mark.parametrize("first, second, tail", [
    (20, 4, "ELLO WORLD "),   # 5x slower: the first letter is lost, the rest is exact
    (40, 8, "ELLO WORLD "),
    (15, 5, "ELLO WORLD "),   # 3x: slow dits look like dahs, so five marks are needed
    (12, 4, "ELLO WORLD "),
    (12, 6, "ELLO WORLD "),   # 2x
    (20, 12, "HELLO WORLD "),  # moderate: nothing lost (letter gaps used to read as word gaps)
    (15, 10, "HELLO WORLD "),
])
def test_speed_decrease_relocks_within_three_marks(first: int, second: int, tail: str):
    slow = 1200.0 / min(first, second)
    runs = make_runs("PARIS", first) + [Run(False, round(7 * slow / 10))] + make_runs("HELLO WORLD", second)
    emitted, dec = decode_runs(runs)
    assert emitted.startswith("PARIS ")
    assert emitted.endswith(tail), emitted
    assert dec.dit_ms == pytest.approx(1200.0 / second, rel=0.1)


@pytest.mark.parametrize("text", ["MOTTO OO", "0 TO 9", "MOM OM", "OSO", "TT EEE", "SOS HELLO"])
def test_runs_of_dahs_do_not_trigger_a_false_resync(text: str):
    emitted, dec = decode_runs(make_runs(text, 15))
    assert emitted.strip() == text
    assert dec.dit_ms == pytest.approx(80.0, rel=0.06)


def test_resync_rereads_the_letter_in_progress():
    # 20 WPM, then H at 4 WPM: the third slow dit triggers the resync and the
    # letter in progress is re-read with the new estimate.
    dec = MorseDecoder()
    for r in make_runs("PARIS", 20):
        dec.feed(r)
    dec.feed(Run(False, 210))
    slow = make_runs("H", 4)  # four 300 ms dits with 300 ms gaps
    out = "".join(dec.feed(r) for r in slow[:5])  # dit gap dit gap dit: resync on the third mark
    assert dec.dit_ms == pytest.approx(300.0)
    assert dec.buffer == "."  # the mark that triggered it is a dit under the new estimate
    assert out.strip() == "T T"  # the two marks read as dahs before it were already emitted
