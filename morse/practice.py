"""Practice mode: targets to key, scoring of the copy, and a rhythm report.

Pure logic shared with ``web/js/practice.js`` (same word list, same
generators, same scoring), so both apps grade the same way.

* :class:`Practice` hands out targets: common words, call signs, five-digit
  groups, or a mix.  Seeded, so tests and the two ports agree.
* :func:`score` aligns what was sent against the target with the edit
  distance and reports correct letters, errors and accuracy.
* :func:`rhythm` measures how far the operator's marks and gaps sit from the
  ideal 1 : 3 : 7 proportions, in percent of the intended element.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal, Sequence

from morse.runs import Run

__all__ = ["KINDS", "WORDS", "Practice", "Rhythm", "Score", "call_sign", "digit_group", "rhythm", "score"]

Kind = Literal["words", "calls", "digits", "mixed"]
KINDS: tuple[str, ...] = ("words", "calls", "digits", "mixed")

WORDS: tuple[str, ...] = (
    "THE", "AND", "FOR", "YOU", "ARE", "WITH", "THIS", "HAVE", "FROM", "THAT", "NOT", "BUT",
    "ALL", "CAN", "HER", "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM", "HIS", "HOW",
    "MAN", "NEW", "NOW", "OLD", "SEE", "TWO", "WAY", "WHO", "BOY", "DID", "ITS", "LET", "PUT",
    "SAY", "SHE", "TOO", "USE", "GOOD", "TIME", "YEAR", "WORK", "BACK", "CALL", "COME", "EACH",
    "FIND", "GIVE", "HAND", "HELP", "HERE", "HOME", "JUST", "KEEP", "KIND", "KNOW", "LAST",
    "LIKE", "LINE", "LIVE", "LONG", "LOOK", "MAKE", "MANY", "MORE", "MOST", "MOVE", "MUCH",
    "MUST", "NAME", "NEED", "NEXT", "ONLY", "OPEN", "OVER", "PART", "PLAY", "READ", "REAL",
    "SAME", "SEND", "SHOW", "SIDE", "SOME", "TAKE", "TELL", "THAN", "THEM", "THEN", "THEY",
    "TURN", "VERY", "WANT", "WEEK", "WELL", "WHAT", "WHEN", "WILL", "WORD", "ABOUT", "AFTER",
    "AGAIN", "COULD", "EVERY", "FIRST", "GREAT", "HOUSE", "LARGE", "NEVER", "OTHER", "PLACE",
    "RADIO", "RIGHT", "SMALL", "SOUND", "STILL", "THEIR", "THERE", "THESE", "THING", "THINK",
    "THREE", "UNDER", "WATER", "WHERE", "WHICH", "WORLD", "WOULD", "WRITE", "MORSE", "SIGNAL",
    "BEEPER", "HELLO", "PARIS", "CODEX", "ANTENNA", "REPEAT", "READY", "AGAIN", "OVER", "OUT",
)

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_PREFIXES = ("K", "W", "N", "AA", "KB", "KC", "KD", "WA", "WB", "VE", "VK", "G", "M", "DL", "JA", "F", "EA", "PY")


def call_sign(rng: random.Random) -> str:
    """A plausible amateur call sign: prefix, one digit, one to three letters."""
    prefix = rng.choice(_PREFIXES)
    suffix = "".join(rng.choice(_LETTERS) for _ in range(rng.randint(1, 3)))
    return f"{prefix}{rng.randint(0, 9)}{suffix}"


def digit_group(rng: random.Random, length: int = 5) -> str:
    """A group of digits, the classic numbers drill."""
    return "".join(str(rng.randint(0, 9)) for _ in range(length))


class Practice:
    """Hands out targets of one kind; ``seed`` makes the sequence reproducible."""

    def __init__(self, kind: Kind = "words", seed: int | None = None) -> None:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.kind: Kind = kind
        self._rng = random.Random(seed)
        self.target: str | None = None
        self.asked: int = 0
        self.correct: int = 0

    def set_kind(self, kind: Kind) -> None:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.kind = kind

    def next_target(self) -> str:
        """Draw the next target (never the same as the current one)."""
        for _ in range(20):
            kind = self.kind
            if kind == "mixed":
                kind = self._rng.choice(("words", "words", "calls", "digits"))
            if kind == "words":
                target = self._rng.choice(WORDS)
            elif kind == "calls":
                target = call_sign(self._rng)
            else:
                target = digit_group(self._rng)
            if target != self.target:
                break
        self.target = target
        self.asked += 1
        return target

    def record(self, result: Score) -> None:
        """Count a checked target toward the session tally."""
        if result.correct == result.total and result.errors == 0:
            self.correct += 1

    def reset_session(self) -> None:
        self.asked = 0
        self.correct = 0
        self.target = None


@dataclass(frozen=True)
class Score:
    """How a copy compares with its target."""

    target: str
    sent: str
    correct: int
    """Target characters reproduced in order (the edit-distance alignment's matches)."""
    errors: int
    """Edit distance: substitutions, insertions and deletions."""
    total: int
    """Characters in the target."""

    @property
    def accuracy(self) -> float:
        """1.0 for a perfect copy; errors beyond the target's length drive it to 0."""
        if self.total == 0:
            return 1.0 if not self.sent else 0.0
        return max(0.0, 1.0 - self.errors / self.total)

    @property
    def perfect(self) -> bool:
        return self.errors == 0 and self.total > 0


def _normalise(text: str) -> str:
    return " ".join(text.upper().split())


def score(target: str, sent: str) -> Score:
    """Grade ``sent`` against ``target`` (case and surplus whitespace ignored)."""
    a = _normalise(target)
    b = _normalise(sent)
    n, m = len(a), len(b)
    # Levenshtein distance with a match count recovered from the alignment.
    dist = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dist[i][0] = i
    for j in range(1, m + 1):
        dist[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dist[i][j] = min(dist[i - 1][j] + 1, dist[i][j - 1] + 1, dist[i - 1][j - 1] + cost)
    # Walk back to count matched characters.
    i, j, matches = n, m, 0
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1] and dist[i][j] == dist[i - 1][j - 1]:
            matches += 1
            i -= 1
            j -= 1
        elif dist[i][j] == dist[i - 1][j - 1] + 1:
            i -= 1
            j -= 1
        elif dist[i][j] == dist[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return Score(target=a, sent=b, correct=matches, errors=dist[n][m], total=n)


@dataclass(frozen=True)
class Rhythm:
    """Timing quality of a keyed passage, in percent of the intended element length."""

    mark_error_pct: float
    """Mean |measured − ideal| / ideal over marks, ideal being 1 or 3 dits."""
    gap_error_pct: float
    """Mean |measured − ideal| / ideal over gaps, ideal being 1, 3 or 7 dits."""
    marks: int
    gaps: int

    @property
    def error_pct(self) -> float:
        """Marks and gaps together, weighted by count."""
        n = self.marks + self.gaps
        if n == 0:
            return 0.0
        return (self.mark_error_pct * self.marks + self.gap_error_pct * self.gaps) / n


def rhythm(runs: Sequence[Run], dit_ms: float) -> Rhythm:
    """Compare each run with the nearest ideal element at ``dit_ms``.

    Marks are dits (1) or dahs (3) by the 2-dit split; gaps are element (1),
    letter (3) or word (7) gaps by the 2- and 5-dit splits. Leading and
    trailing gaps are not runs, so pass only the runs between the first and
    last mark.
    """
    if not dit_ms > 0:
        raise ValueError(f"dit_ms must be positive, got {dit_ms!r}")
    mark_errors: list[float] = []
    gap_errors: list[float] = []
    for run in runs:
        units = run.ms / dit_ms
        if run.on:
            ideal = 1.0 if units < 2.0 else 3.0
            mark_errors.append(abs(units - ideal) / ideal)
        else:
            ideal = 1.0 if units < 2.0 else 3.0 if units < 5.0 else 7.0
            gap_errors.append(abs(units - ideal) / ideal)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0  # noqa: E731
    return Rhythm(
        mark_error_pct=100.0 * mean(mark_errors),
        gap_error_pct=100.0 * mean(gap_errors),
        marks=len(mark_errors),
        gaps=len(gap_errors),
    )
