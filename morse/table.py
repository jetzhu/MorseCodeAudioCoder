"""International Morse code table and text encoder.

``MORSE_TABLE`` maps a character to its dot/dash string, ``INVERSE`` maps the
dot/dash string back to the character.  Every code is unique, so the two
dictionaries are exact inverses of each other.
"""
from __future__ import annotations

MORSE_TABLE: dict[str, str] = {
    # Letters
    "A": ".-",
    "B": "-...",
    "C": "-.-.",
    "D": "-..",
    "E": ".",
    "F": "..-.",
    "G": "--.",
    "H": "....",
    "I": "..",
    "J": ".---",
    "K": "-.-",
    "L": ".-..",
    "M": "--",
    "N": "-.",
    "O": "---",
    "P": ".--.",
    "Q": "--.-",
    "R": ".-.",
    "S": "...",
    "T": "-",
    "U": "..-",
    "V": "...-",
    "W": ".--",
    "X": "-..-",
    "Y": "-.--",
    "Z": "--..",
    # Digits
    "0": "-----",
    "1": ".----",
    "2": "..---",
    "3": "...--",
    "4": "....-",
    "5": ".....",
    "6": "-....",
    "7": "--...",
    "8": "---..",
    "9": "----.",
    # Punctuation
    ".": ".-.-.-",
    ",": "--..--",
    "?": "..--..",
    "/": "-..-.",
    "=": "-...-",
    "'": ".----.",
    "(": "-.--.",
    ")": "-.--.-",
    ":": "---...",
    ";": "-.-.-.",
    "+": ".-.-.",
    "-": "-....-",
    "_": "..--.-",
    '"': ".-..-.",
    "@": ".--.-.",
    "!": "-.-.--",
}
"""Character -> dot/dash code for A-Z, 0-9 and common punctuation."""

INVERSE: dict[str, str] = {code: char for char, code in MORSE_TABLE.items()}
"""Dot/dash code -> character; the exact inverse of ``MORSE_TABLE``."""

assert len(INVERSE) == len(MORSE_TABLE), "MORSE_TABLE contains a duplicate code"

LETTER_SEP = " "
"""Separator between letters in ``encode`` output."""

WORD_SEP = " / "
"""Separator between words in ``encode`` output."""


def lookup(symbols: str) -> str | None:
    """Return the character for a dot/dash string, or ``None`` when unknown.

    The lookup is exact: ``"..."`` gives ``"S"``, while ``""``, ``".. ."`` or
    any sequence not in the table gives ``None``.
    """
    return INVERSE.get(symbols)


def encode(text: str, unknown: str = "") -> str:
    """Encode text as Morse, letters separated by one space, words by ``' / '``.

    ``'SOS X'`` becomes ``'... --- ... / -..-'``.  Input is upper-cased and any
    run of whitespace is one word boundary.  Characters not in the table are
    replaced by ``unknown``; with the default empty string they are dropped,
    and a word that becomes empty as a result is dropped too.
    """
    words: list[str] = []
    for word in text.upper().split():
        codes = [MORSE_TABLE.get(ch, unknown) for ch in word]
        encoded = LETTER_SEP.join(code for code in codes if code)
        if encoded:
            words.append(encoded)
    return WORD_SEP.join(words)
