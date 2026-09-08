"""Tests for morse.table: table integrity, lookup, encode."""
from __future__ import annotations

import string

import pytest

from morse.table import INVERSE, MORSE_TABLE, encode, lookup

PUNCTUATION = ".,?/=\'():;+-_\"@!"


def decode(morse: str) -> str:
    """Inverse of ``encode`` for round-trip tests: ' / ' splits words, ' ' letters."""
    words = []
    for word in morse.split(" / "):
        words.append("".join(lookup(code) or "?" for code in word.split(" ") if code))
    return " ".join(words)


# --- table integrity -------------------------------------------------------

def test_table_covers_letters_digits_and_punctuation():
    for ch in string.ascii_uppercase + string.digits + PUNCTUATION:
        assert ch in MORSE_TABLE, ch
    assert len(MORSE_TABLE) == 26 + 10 + len(PUNCTUATION)


def test_codes_use_only_dots_and_dashes():
    for ch, code in MORSE_TABLE.items():
        assert code, ch
        assert set(code) <= {".", "-"}, (ch, code)


def test_codes_are_unique_and_inverse_matches():
    assert len(INVERSE) == len(MORSE_TABLE)
    for ch, code in MORSE_TABLE.items():
        assert INVERSE[code] == ch
    for code, ch in INVERSE.items():
        assert MORSE_TABLE[ch] == code


@pytest.mark.parametrize(
    "char, code",
    [
        ("E", "."), ("T", "-"), ("S", "..."), ("O", "---"), ("A", ".-"), ("N", "-."),
        ("Q", "--.-"), ("Z", "--.."),
        ("0", "-----"), ("1", ".----"), ("5", "....."), ("9", "----."),
        (".", ".-.-.-"), (",", "--..--"), ("?", "..--.."), ("/", "-..-."),
        ("=", "-...-"), ("'", ".----."), ("(", "-.--."), (")", "-.--.-"),
        (":", "---..."), (";", "-.-.-."), ("+", ".-.-."), ("-", "-....-"),
        ("_", "..--.-"), ('"', ".-..-."), ("@", ".--.-."), ("!", "-.-.--"),
    ],
)
def test_known_codes(char, code):
    assert MORSE_TABLE[char] == code


# --- lookup ---------------------------------------------------------------

def test_lookup_known():
    assert lookup("...") == "S"
    assert lookup("---") == "O"
    assert lookup(".-") == "A"
    assert lookup("-----") == "0"
    assert lookup("..--..") == "?"


def test_lookup_round_trips_every_entry():
    for ch, code in MORSE_TABLE.items():
        assert lookup(code) == ch


@pytest.mark.parametrize("symbols", ["", ".......", "--------", "abc", ". .", " ...", "...-...-"])
def test_lookup_unknown_returns_none(symbols):
    assert lookup(symbols) is None


# --- encode ---------------------------------------------------------------

def test_encode_contract_example():
    assert encode("SOS X") == "... --- ... / -..-"


def test_encode_single_word_letters_separated_by_one_space():
    assert encode("SOS") == "... --- ..."
    assert encode("E") == "."


def test_encode_is_case_insensitive():
    assert encode("sos") == encode("SOS")
    assert encode("Hello") == encode("HELLO")


def test_encode_hello_world():
    assert encode("HELLO WORLD") == ".... . .-.. .-.. --- / .-- --- .-. .-.. -.."


def test_encode_digits_and_punctuation():
    assert encode("73!") == "--... ...-- -.-.--"
    assert encode("A@B.C") == ".- .--.-. -... .-.-.- -.-."


def test_encode_collapses_whitespace_runs():
    assert encode("  SOS   X  ") == "... --- ... / -..-"
    assert encode("A\tB\nC") == ".- / -... / -.-."


def test_encode_empty_and_blank():
    assert encode("") == ""
    assert encode("   ") == ""


def test_encode_drops_unknown_characters_by_default():
    assert encode("A~B") == ".- -..."
    assert encode("A ~ B") == ".- / -..."       # an all-unknown word disappears
    assert encode("~") == ""


def test_encode_unknown_placeholder():
    assert encode("A~B", unknown="?") == ".- ? -..."
    assert encode("A ~ B", unknown="?") == ".- / ? / -..."


def test_encode_decode_round_trip_full_table():
    text = "THE QUICK BROWN FOX JUMPS OVER 13 LAZY DOGS 2490.7 HZ ?!"
    assert decode(encode(text)) == text
    every = "".join(sorted(MORSE_TABLE))
    assert decode(encode(every)) == every
