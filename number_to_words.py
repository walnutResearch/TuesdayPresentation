#!/usr/bin/env python3
"""
Convert numbers to English words (e.g. 42 -> "forty-two"). Handles integers in a typical range; used for
readable labels or reports. Can be run as a script or imported (number_to_words function).

How to run:
  python number_to_words.py 42
  python number_to_words.py --number 123
"""
import argparse
import sys


UNITS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
]

TEENS = [
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]

TENS = [
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
]

SCALES = [
    "",
    "thousand",
    "million",
    "billion",
    "trillion",
    "quadrillion",
    "quintillion",
]


def _convert_hundreds(number: int) -> str:
    words = []
    hundreds = number // 100
    remainder = number % 100

    if hundreds:
        words.append(UNITS[hundreds])
        words.append("hundred")

    if remainder:
        if remainder < 10:
            words.append(UNITS[remainder])
        elif 10 <= remainder < 20:
            words.append(TEENS[remainder - 10])
        else:
            tens_val = remainder // 10
            unit_val = remainder % 10
            if unit_val:
                words.append(f"{TENS[tens_val]}-{UNITS[unit_val]}")
            else:
                words.append(TENS[tens_val])

    return " ".join(words) if words else UNITS[0]


def number_to_words(value: str) -> str:
    """
    Convert a numeric string (int or float) into English words.

    - Supports arbitrarily large integers.
    - For decimals, pronounces the fractional part digit-by-digit after "point".
    - Handles negative numbers.
    """
    value = value.strip().replace(",", "")
    if not value:
        raise ValueError("Empty input")

    negative = value.startswith("-")
    if negative:
        value = value[1:]

    if value.count(".") > 1:
        raise ValueError("Invalid number format")

    if "." in value:
        int_part_str, frac_part_str = value.split(".", 1)
    else:
        int_part_str, frac_part_str = value, ""

    if not int_part_str or not int_part_str.isdigit():
        # Allow empty int part like .5 -> 0.5
        if int_part_str == "":
            int_part = 0
        else:
            raise ValueError("Invalid integer part")
    else:
        int_part = int(int_part_str)

    # Convert integer part
    if int_part == 0:
        int_words = UNITS[0]
    else:
        chunks = []
        n = int_part
        while n > 0:
            chunks.append(n % 1000)
            n //= 1000

        words_parts = []
        for idx, chunk in enumerate(chunks):
            if chunk == 0:
                continue
            chunk_words = _convert_hundreds(chunk)
            scale = SCALES[idx] if idx < len(SCALES) else f"10^{idx * 3}"
            if scale:
                words_parts.append(f"{chunk_words} {scale}")
            else:
                words_parts.append(chunk_words)

        int_words = " ".join(reversed(words_parts))

    # Convert fractional part (if any)
    if frac_part_str:
        if not frac_part_str.isdigit():
            raise ValueError("Invalid fractional part")
        frac_words = " ".join(UNITS[int(d)] for d in frac_part_str)
        full = f"{int_words} point {frac_words}"
    else:
        full = int_words

    return f"negative {full}" if negative and full != UNITS[0] else full


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a number to its English words representation.",
    )
    parser.add_argument(
        "number",
        help="Number to convert (e.g., 12345, -42, 3.1415). Commas allowed.",
    )
    args = parser.parse_args(argv)

    try:
        print(number_to_words(args.number))
        return 0
    except Exception as exc:  # noqa: BLE001 - user-facing CLI
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


