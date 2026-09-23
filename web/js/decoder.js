/**
 * Morse timing decoder: turns final ON/OFF runs into text.
 *
 * Line-by-line port of `morse/decoder.py`; the Python module is the reference
 * and `web/test/vectors.json` pins both to the same numbers. The decoder
 * consumes {@link Run} objects one at a time and keeps an adaptive estimate of
 * the dit length `T` (`ditMs`) and of the reverb offset `d` (`offsetMs`) that
 * lengthens every mark and shortens every gap by the same amount.
 *
 * Timing rule
 * -----------
 * Keep the last `window` raw mark lengths and gap lengths. `M` is the 10th
 * percentile of the marks and `G` the 10th percentile of the gaps after
 * capping each gap at `20 * M`. Percentiles use the nearest-rank method:
 * sort ascending and take the element at 1-based rank `ceil(q / 100 * n)`
 * (so the minimum for `n <= 10`, the 2nd smallest for `11 <= n <= 20`, the
 * 3rd smallest for `n = 30`); never interpolate between order statistics.
 *
 * Then `T = (M + G) / 2` and `d = (M - G) / 2`; if `d < 0` use `T = M` and
 * `d = 0`. When only one mark class has been seen (longest mark below twice
 * the shortest) and `M >= 2.5 G`, the marks are dahs rather than dits:
 * `T = (M + G) / 4`, `d = (M - 3G) / 4` (a second gap class, when present,
 * arbitrates: see `_marksAreDahs`). With both classes present but dits rare,
 * `M` is taken from the dit class itself. With fewer than two marks or one gap
 * the estimate falls back to `T = 1200 / wpm` when a speed was given, else
 * `T = 150 ms`, and `d = 0`. When `adaptive` is false and `wpm` was given, `T`
 * stays fixed and only `d` adapts.
 *
 * Hold-back
 * ---------
 * Without a WPM seed a mark alone cannot tell a slow dit from a fast dah, so
 * runs are held back (`timingReady` false, `pendingCount` > 0) until the
 * estimate is trusted: both mark classes seen (longest mark >= 2 x shortest)
 * or three marks (five, or a letter gap, when three marks of one class look
 * like dahs: reverberant dits with jitter look the same). Once trusted, a mark
 * shorter than `0.4 T` is a click and ignored, unless four arrive in a row
 * with Morse-like spacing, which is a faster speed and opens a valve until
 * the next long gap. Three consecutive marks of at least `1.5 T` (no dit
 * seen) mean the keying slowed down: the windows are rebuilt from those runs
 * and the letter in progress re-read (five marks when they could be genuine
 * dahs, about `3 T`). The held runs are then classified in one go, so a message
 * may open with T, M, O or a digit and may be keyed at 2 WPM. While holding,
 * `idle()` flushes after the longer of `7T` and `3.5 x` the longest held mark.
 *
 * Marks are corrected as `ms - d` and gaps as `ms + d`. A corrected mark
 * `< 2T` is a dit, otherwise a dah. A corrected gap `< 2T` is inside a
 * letter, `2T..5T` ends the letter and `>= 5T` ends the word (one space).
 * Unknown symbol sequences emit `?`. A leading gap before any mark emits
 * nothing and is not used for timing.
 *
 * A mark of `maxMarkMs` or longer (default 5000 ms) is not a symbol: it is
 * ignored, the pending buffer is cleared and nothing is emitted, so a held
 * button or a detector timeout never produces `?` or disturbs the letter that
 * follows. The comparison uses the measured length before the offset
 * correction (`ms >= maxMarkMs`). An ignored mark is not used for timing and
 * increments neither `letterCount` nor `unknownCount`.
 *
 * ES module with no DOM or Web Audio dependency; runs in Node for the tests.
 */

import { lookup } from "./table.js";

/** Dit length (ms) used before enough runs have arrived and no WPM was given. */
export const DEFAULT_DIT_MS = 150.0;

/** Measured marks this long or longer (ms) are not symbols and are ignored. */
export const DEFAULT_MAX_MARK_MS = 5000.0;

const PERCENTILE = 10.0;
const GAP_CAP_FACTOR = 20.0;
const DIT_DAH_SPLIT = 2.0; // corrected mark >= this * T is a dah
const LETTER_GAP = 2.0; // corrected gap >= this * T ends the letter
const WORD_GAP = 5.0; // corrected gap >= this * T ends the word
const IDLE_FLUSH = 7.0; // idle OFF (corrected) > this * T flushes the pending letter
const TRUST_RATIO = 2.0; // longest/shortest mark >= this: both mark classes have been seen
const TRUST_MARKS = 3; // ... or this many marks: the estimate is trusted and classification starts
const GLITCH_FRACTION = 0.4; // trusted: a measured run shorter than this * T is a glitch, not a symbol
const GLITCH_STREAK_MAX = 4; // ... unless this many arrive in a row: then the keying got faster
const GLITCH_STREAK_GAP_FACTOR = 8.0; // a short mark after a gap longer than this * its length is a click
const TRUST_MARKS_AMBIGUOUS = 5; // single-class, dah-like marks with no letter gap yet: wait for this many
const DAH_RULE_RATIO = 2.5; // single-class marks >= this * shortest gap are dahs, not dits
const PENDING_IDLE_FACTOR = 3.5; // untrusted: idle flush waits this * longest pending mark
const FRAGMENT_MS = 20.0; // marks this short never anchor the dit class (debounce leftovers)
const RESYNC_MIN_RATIO = 1.5; // a mark at least this many T long is "not a dit at the current estimate"
const RESYNC_STREAK = 3; // that many such marks in a row: the estimate is stale-low (keying slowed)
const RESYNC_STREAK_AMBIGUOUS = 5; // ... when those marks are about 3 T: slow dits and real dahs look alike
const RESYNC_DAH_BAND = [2.2, 4.5]; // a mark this many T long could genuinely be a dah (jitter, low-biased T)
const RESYNC_GRACE = 8; // marks accepted without the glitch rule right after a resync

/**
 * Return the q-th percentile of `values` by the nearest-rank method.
 *
 * This is the reference definition of "percentile" for the timing rule: sort
 * ascending and take the element at 1-based rank `ceil(q / 100 * n)`, clamped
 * to `1..n`. The result is always one of the observed values; for `q = 10` it
 * is the minimum when `n <= 10`, the 2nd smallest for `11 <= n <= 20` and the
 * 3rd smallest for `n = 30`. The arithmetic (`q / 100 * n`, left to right) is
 * the same IEEE expression as the Python reference, so the rank matches
 * bit for bit.
 *
 * @param {Iterable<number>} values non-empty
 * @param {number} q percentile in 0..100
 * @returns {number}
 * @throws {RangeError} when `values` is empty
 */
export function nearestRankPercentile(values, q) {
  const ordered = Array.from(values).sort((a, b) => a - b);
  const n = ordered.length;
  if (n === 0) throw new RangeError("percentile of an empty sequence");
  const rank = Math.ceil((q / 100.0) * n);
  const index = Math.min(Math.max(rank - 1, 0), n - 1);
  return ordered[index];
}

/**
 * Fixed-capacity sliding window (the port of `collections.deque(maxlen=n)`).
 * @template T
 */
class SlidingWindow {
  /** @param {number} maxlen */
  constructor(maxlen) {
    this.maxlen = maxlen;
    /** @type {T[]} */
    this.items = [];
  }

  /** @param {T} value */
  push(value) {
    this.items.push(value);
    if (this.items.length > this.maxlen) this.items.shift();
  }

  clear() {
    this.items.length = 0;
  }

  get length() {
    return this.items.length;
  }
}

/**
 * Timing state machine that decodes final runs into text.
 *
 * Public state, mirroring the Python attributes:
 * - `ditMs`: current dit-length estimate `T` in ms.
 * - `offsetMs`: current reverb correction `d` in ms (marks measure `d` too
 *   long, gaps `d` too short).
 * - `wpm`: `1200 / ditMs` (read-only getter).
 * - `maxMarkMs`: a mark measured this long or longer is ignored (settable).
 * - `buffer`: pending symbols of the letter in progress, e.g. `".-"`.
 * - `text`: everything emitted so far.
 * - `letterCount`: letters emitted (unknown `?` included).
 * - `unknownCount`: unknown symbol sequences emitted as `?`.
 */
export class MorseDecoder {
  /**
   * @param {object} [options]
   * @param {number | null} [options.wpm=null] optional keying speed; seeds
   *   `T = 1200 / wpm` until enough runs have arrived (and fixes it when
   *   `adaptive` is false). Must be positive when given.
   * @param {boolean} [options.adaptive=true] when false and `wpm` is given,
   *   `T` never changes and only the offset `d` adapts to the measured runs.
   * @param {number} [options.window=30] number of recent marks and of recent
   *   gaps kept for the timing estimate; at least 2.
   * @param {number} [options.maxMarkMs=5000] marks measured this long or
   *   longer are not symbols and are ignored; `Infinity` disables the rule.
   *   Must be positive.
   * @throws {RangeError} on an invalid option
   */
  constructor({ wpm = null, adaptive = true, window = 30, maxMarkMs = DEFAULT_MAX_MARK_MS } = {}) {
    if (wpm !== null && wpm !== undefined && !(wpm > 0)) {
      throw new RangeError(`wpm must be positive, got ${wpm}`);
    }
    if (!(window >= 2)) {
      throw new RangeError(`window must be at least 2, got ${window}`);
    }
    if (!(maxMarkMs > 0)) {
      throw new RangeError(`maxMarkMs must be positive, got ${maxMarkMs}`);
    }
    /** @type {number | null} */
    this._wpm = wpm === null || wpm === undefined ? null : Number(wpm);
    /** @type {boolean} */
    this._adaptive = Boolean(adaptive);
    /** @type {number} */
    this._window = Math.trunc(window);
    /** @type {number} */
    this.maxMarkMs = Number(maxMarkMs);
    /** @type {SlidingWindow<number>} */
    this._marks = new SlidingWindow(this._window);
    /** @type {SlidingWindow<number>} */
    this._gaps = new SlidingWindow(this._window);
    /** @type {boolean} */
    this._seenMark = false;
    /**
     * Runs held back until the timing estimate is trusted (Auto mode only:
     * a WPM seed makes the estimate trusted from the start).
     * @type {Array<{on: boolean, ms: number}>}
     */
    this._pending = [];
    /** @type {boolean} */
    this._trusted = this._wpm !== null;
    /** @type {number} consecutive marks rejected as glitches */
    this._glitchStreak = 0;
    /** @type {number | null} length of the most recent gap run */
    this._lastGapMs = null;
    /** @type {number[]} lengths (in T) of consecutive marks at least 1.5 T long */
    this._longRatios = [];
    /** @type {number[]} raw marks of the letter in progress, re-read after a resync */
    this._letterMarks = [];
    this._glitchGrace = 0;

    /** @type {number} */
    this.ditMs = this._seedDitMs();
    /** @type {number} */
    this.offsetMs = 0.0;
    /** @type {string} */
    this.buffer = "";
    /** @type {string} */
    this.text = "";
    /** @type {number} */
    this.letterCount = 0;
    /** @type {number} */
    this.unknownCount = 0;
  }

  // ---------------------------------------------------------------- public

  /** Current speed estimate, `1200 / ditMs`. @returns {number} */
  get wpm() {
    return 1200.0 / this.ditMs;
  }

  /**
   * Consume one final run and return the newly emitted text.
   *
   * Marks append a dit or dah to `buffer` and return `""`. Gaps return the
   * letter they close (`""` for an intra-letter gap), plus a space when they
   * close a word. Everything returned is also appended to `text`.
   *
   * A mark measured `maxMarkMs` or longer is not a symbol (a held button, or
   * the detector's stuck-ON timeout): it clears `buffer`, returns `""`,
   * leaves the timing estimate and the counters alone, and the gap after it
   * is handled as usual, so the next letter decodes cleanly.
   *
   * @param {{on: boolean, ms: number}} run a final run (a {@link Run})
   * @returns {string}
   */
  feed(run) {
    const ms = Number(run.ms);
    const settled = this._trusted && !this._pending.length;
    if (run.on) {
      if (ms >= this.maxMarkMs) {
        this.buffer = "";
        this._letterMarks.length = 0;
        this._pending.length = 0;
        return "";
      }
      // A click, far shorter than a dit: not a symbol, not timing evidence.
      // Four in a row with Morse spacing are a faster speed instead, so the
      // valve lets them through and the window adapts.
      // The valve opens and stays open until a long gap; a click after
      // silence always resets it.
      if (settled && this._glitchGrace === 0 && ms < GLITCH_FRACTION * this.ditMs) {
        if (this._lastGapMs !== null && this._lastGapMs > GLITCH_STREAK_GAP_FACTOR * ms) this._glitchStreak = 0;
        this._glitchStreak += 1;
        if (this._glitchStreak < GLITCH_STREAK_MAX) return "";
        this._glitchStreak = GLITCH_STREAK_MAX;
      }
      if (this._glitchGrace) this._glitchGrace -= 1;
      this._marks.push(ms);
      this._seenMark = true;
      this._updateTiming();
      if (settled) return this._classifyMark(ms);
      this._pending.push({ on: true, ms });
      return this._trusted ? this._replay() : "";
    }

    this._lastGapMs = ms;
    if (!this._seenMark) {
      return ""; // leading silence carries no information
    }
    // The gap around a glitch or a dropout: inside the letter, not timing evidence.
    if (settled && ms < GLITCH_FRACTION * this.ditMs) return "";
    this._gaps.push(ms);
    this._updateTiming();
    if (settled) return this._classifyGap(ms);
    if (!this._pending.length) return ""; // untrusted with nothing held: an idle flush already closed the letter
    this._pending.push({ on: false, ms });
    return this._trusted ? this._replay() : "";
  }

  /**
   * `true` once the estimate is trusted and runs are classified as they
   * arrive. Without a WPM seed the first runs are held back: a mark alone
   * cannot tell a slow dit from a fast dah. The estimate is trusted once both
   * mark classes have been seen (longest mark >= 2 x shortest) or three marks
   * have arrived; the held runs are then decoded in one go.
   * @returns {boolean}
   */
  get timingReady() {
    return this._trusted;
  }

  /** Number of final runs held back while the estimate is not yet trusted. @returns {number} */
  get pendingCount() {
    return this._pending.length;
  }

  /**
   * What the held-back runs read as under the current estimate, e.g. `"... -"`:
   * dits and dahs of the pending runs with a space at each gap that would end
   * a letter; empty when nothing is held. Shown by the UI while the speed
   * estimate settles and replaced by the final reading on replay.
   * @returns {string}
   */
  get provisional() {
    let out = "";
    for (const r of this._pending) {
      if (r.on) out += r.ms - this.offsetMs < DIT_DAH_SPLIT * this.ditMs ? "." : "-";
      else if (r.ms + this.offsetMs >= LETTER_GAP * this.ditMs && out && !out.endsWith(" ")) out += " ";
    }
    return out;
  }

  /**
   * Take over `other`'s recent-run windows and trust state, then recompute.
   *
   * Used when the UI swaps decoders (Auto/Manual speed): the new decoder
   * starts with the measured speed and reverb offset instead of re-learning
   * them. Text, pending letter, counters and held-back runs are not copied;
   * flush `other` first (`idle(Infinity)`) if a letter is in progress. With a
   * fixed `wpm` only the windows are taken, so `T` stays at the seed and `d`
   * comes from the history.
   *
   * @param {MorseDecoder} other
   */
  adoptTiming(other) {
    this._marks.clear();
    for (const v of other._marks.items) this._marks.push(v);
    this._gaps.clear();
    for (const v of other._gaps.items) this._gaps.push(v);
    this._seenMark = other._seenMark;
    if (this._wpm === null) this._trusted = other._trusted;
    this._updateTiming();
  }

  /**
   * Report the length of the current, unfinished OFF run.
   *
   * Once `offMs + offsetMs > 7 * ditMs` the pending letter is decoded and a
   * space appended, so the last letter of a message appears without waiting
   * for the next tone. Returns the newly emitted text; with an empty `buffer`
   * nothing happens, so the flush occurs exactly once. `Infinity` flushes
   * unconditionally (end of stream).
   *
   * @param {number} offMs
   * @returns {string}
   */
  idle(offMs) {
    if (this._pending.length) {
      // A held mark may be a dah whose letter gap is as long as itself, so
      // wait for the longer of 7T and 3.5 x the longest held mark, then decode
      // the held runs with the best estimate available.
      let longest = 0;
      for (const r of this._pending) if (r.on && r.ms > longest) longest = r.ms;
      const limit = Math.max(IDLE_FLUSH * this.ditMs, PENDING_IDLE_FACTOR * longest);
      if (offMs + this.offsetMs > limit) {
        return this._replay() + this._flushLetter() + this._endWord();
      }
      return "";
    }
    if (!this.buffer) return "";
    if (offMs + this.offsetMs > IDLE_FLUSH * this.ditMs) {
      return this._flushLetter() + this._endWord();
    }
    return "";
  }

  /**
   * Clear the text, the pending letter and the counters.
   *
   * With `keepTiming` the recent-run windows and the current `T` and `d`
   * survive; otherwise timing returns to the seed values. `maxMarkMs` is
   * never touched.
   *
   * @param {boolean} [keepTiming=false]
   */
  reset(keepTiming = false) {
    this.buffer = "";
    this.text = "";
    this.letterCount = 0;
    this.unknownCount = 0;
    this._seenMark = false;
    this._pending.length = 0;
    this._glitchStreak = 0;
    this._lastGapMs = null;
    this._longRatios.length = 0;
    this._letterMarks.length = 0;
    this._glitchGrace = 0;
    if (!keepTiming) {
      this._marks.clear();
      this._gaps.clear();
      this._trusted = this._wpm !== null;
      this._updateTiming();
    }
  }

  // --------------------------------------------------------------- private

  /** @returns {number} */
  _seedDitMs() {
    return this._wpm !== null ? 1200.0 / this._wpm : DEFAULT_DIT_MS;
  }

  /** @param {number} ms @returns {string} */
  _classifyMark(ms) {
    const corrected = ms - this.offsetMs;
    const ratio = corrected / this.ditMs;
    if (ratio >= RESYNC_MIN_RATIO) this._longRatios.push(ratio);
    else this._longRatios.length = 0; // a mark as short as a dit: the estimate is not stale-low
    this.buffer += corrected < DIT_DAH_SPLIT * this.ditMs ? "." : "-";
    this._letterMarks.push(ms);
    if (this._slowdownDetected()) this._resync();
    return "";
  }

  /** @param {number} ms @returns {string} */
  _classifyGap(ms) {
    const corrected = ms + this.offsetMs;
    // A gap as short as a dit: the marks around it are keyed at this speed.
    if (corrected < RESYNC_MIN_RATIO * this.ditMs) this._longRatios.length = 0;
    if (corrected < LETTER_GAP * this.ditMs) return "";
    let emitted = this._flushLetter();
    if (corrected >= WORD_GAP * this.ditMs) emitted += this._endWord();
    return emitted;
  }

  /**
   * No mark as short as the dit for several marks: the estimate is stale-low
   * (the keying slowed). Speed-ups re-lock within three marks because new
   * short dits become the 10th percentile at once; a slowdown used to wait
   * for the whole window to drain while every new dit read as a dah. Three
   * consecutive marks of at least 1.5 T settle it, except that marks about
   * 3 T long could genuinely be dahs, so when any falls in that band five are
   * required.
   * @returns {boolean}
   */
  _slowdownDetected() {
    if (!this._adaptive && this._wpm !== null) return false; // the speed is fixed by the user
    const n = this._longRatios.length;
    if (n < RESYNC_STREAK) return false;
    const [lo, hi] = RESYNC_DAH_BAND;
    const ambiguous = this._longRatios.some((r) => r >= lo && r <= hi);
    return n >= (ambiguous ? RESYNC_STREAK_AMBIGUOUS : RESYNC_STREAK);
  }

  /**
   * Rebuild the estimate from the recent runs only, so the new speed applies
   * at once; re-read the letter in progress; suspend the glitch rule briefly
   * in case the resync was mistaken (a genuine run of dahs).
   */
  _resync() {
    const k = this._longRatios.length;
    const recentMarks = this._marks.items.slice(-k);
    const recentGaps = this._gaps.items.slice(-k);
    this._marks.clear();
    for (const v of recentMarks) this._marks.push(v);
    this._gaps.clear();
    for (const v of recentGaps) this._gaps.push(v);
    this._longRatios.length = 0;
    this._glitchGrace = RESYNC_GRACE;
    this._updateTiming();
    this.buffer = this._letterMarks.map((m) => (m - this.offsetMs < DIT_DAH_SPLIT * this.ditMs ? "." : "-")).join("");
  }

  /** Classify every held-back run with the current estimate, oldest first. @returns {string} */
  _replay() {
    let emitted = "";
    for (const r of this._pending) emitted += r.on ? this._classifyMark(r.ms) : this._classifyGap(r.ms);
    this._pending.length = 0;
    return emitted;
  }

  /**
   * Recompute `ditMs` and `offsetMs` from the recent runs; refresh trust.
   *
   * `M` and `G` are the nearest-rank 10th percentiles of the recent marks and
   * (capped) gaps. Normally the shortest mark is a dit (`T + d`) and the
   * shortest gap a dit gap (`T - d`), so `T = (M + G) / 2`, `d = (M - G) / 2`.
   * When only one mark class has been seen (longest mark below twice the
   * shortest) and the marks are at least 2.5 x the shortest gap, they are dahs
   * (`3T + d`): `T = (M + G) / 4`, `d = (M - 3G) / 4`. That keeps a message
   * opening with T, M, O or a digit like 0 from being read as dits.
   */
  _updateTiming() {
    const seed = this._seedDitMs();
    if (this._marks.length < 2 || this._gaps.length < 1) {
      this.ditMs = seed;
      this.offsetMs = 0.0;
      return;
    }
    let m = nearestRankPercentile(this._marks.items, PERCENTILE);
    const cap = GAP_CAP_FACTOR * m;
    const g = nearestRankPercentile(this._gaps.items.map((gap) => Math.min(gap, cap)), PERCENTILE);
    if (!this._adaptive && this._wpm !== null) {
      this.ditMs = seed;
      this.offsetMs = Math.max((m - g) / 2.0, 0.0);
      return;
    }
    const marks = this._marks.items;
    const singleClass = Math.max(...marks) < TRUST_RATIO * Math.min(...marks);
    if (!singleClass) {
      // Both classes present but dits may be rare (digits, "MOM"): when the
      // 10th percentile lands in the dah class, take the dit class itself.
      // Marks of two blocks or less are debounce leftovers and never anchor it.
      const usable = marks.filter((x) => x > FRAGMENT_MS);
      const anchor = usable.length ? Math.min(...usable) : Math.min(...marks);
      if (m >= DAH_RULE_RATIO * g && m >= TRUST_RATIO * anchor) {
        m = nearestRankPercentile(marks.filter((x) => x >= anchor && x < TRUST_RATIO * anchor), 50.0);
      }
    }
    if (singleClass && m >= DAH_RULE_RATIO * g && this._marksAreDahs(m, g)) {
      this.ditMs = (m + g) / 4.0;
      this.offsetMs = Math.max((m - 3.0 * g) / 4.0, 0.0);
    } else {
      const d = (m - g) / 2.0;
      if (d < 0.0) {
        this.ditMs = m;
        this.offsetMs = 0.0;
      } else {
        this.ditMs = (m + g) / 2.0;
        this.offsetMs = d;
      }
    }
    if (!this._trusted) {
      // Three marks of one class are enough, unless they look like dahs
      // (>= 2.5 x the shortest gap) with no letter gap yet to arbitrate:
      // reverberant dits with jitter look the same, so wait for a letter gap
      // or five marks.
      const n = marks.length;
      const ambiguous = singleClass && m >= DAH_RULE_RATIO * g;
      if (!singleClass || n >= TRUST_MARKS_AMBIGUOUS ||
          (n >= TRUST_MARKS && (!ambiguous || this._secondGapClass(m, g) !== null))) {
        this._trusted = true;
      }
    }
  }

  /**
   * The shortest recent gap at least twice `g` (a letter or word gap), capped at `20 m`.
   * @param {number} m @param {number} g @returns {number | null}
   */
  _secondGapClass(m, g) {
    const cap = GAP_CAP_FACTOR * m;
    const longer = this._gaps.items.map((x) => Math.min(x, cap)).filter((x) => x >= TRUST_RATIO * g);
    return longer.length ? Math.min(...longer) : null;
  }

  /**
   * Single-class window with `m >= 2.5 g`: dahs (`3T + d`) or reverberant dits?
   *
   * Without a second gap class the ratio alone decides. With one (the
   * shortest gap at least twice `g`: a letter or word gap), compare it with
   * the letter and word gaps each hypothesis predicts and keep the closer in
   * log distance. Dahs: letter `(m + 3g) / 2`, word `(3m + 5g) / 2`. Dits:
   * letter `m + 2g`, word `3m + 4g`.
   *
   * @param {number} m @param {number} g @returns {boolean}
   */
  _marksAreDahs(m, g) {
    const g2 = this._secondGapClass(m, g);
    if (g2 === null) return true;
    const score = (letter, word) => Math.min(Math.abs(Math.log(g2 / letter)), Math.abs(Math.log(g2 / word)));
    return score((m + 3.0 * g) / 2.0, (3.0 * m + 5.0 * g) / 2.0) <= score(m + 2.0 * g, 3.0 * m + 4.0 * g);
  }

  /** Decode `buffer` into one character, append it and return it. @returns {string} */
  _flushLetter() {
    if (!this.buffer) return "";
    let char = lookup(this.buffer);
    if (char === null) {
      char = "?";
      this.unknownCount += 1;
    }
    this.letterCount += 1;
    this.buffer = "";
    this._letterMarks.length = 0;
    this.text += char;
    return char;
  }

  /** Append exactly one space after the last word, if not already there. @returns {string} */
  _endWord() {
    if (!this.text || this.text.endsWith(" ")) return "";
    this.text += " ";
    return " ";
  }
}
