# SPDX-FileCopyrightText: 2024 Idiap Research Institute
# SPDX-FileContributor: Karl El Hajal
#
# SPDX-License-Identifier: MIT

import re

from num2words import num2words

PUNCTUATION_TO_REPLACE_WITH_COMMA = ["(", ")", ":"]
PUNCTUATION = ".!?,"
SPACE = " "


def replace_some_punctuation_with_comma(text):
    for punct in PUNCTUATION_TO_REPLACE_WITH_COMMA:
        text = text.replace(punct, " , ")
    return text


def split_string_on_punctuation(s):
    substrings = []
    current_substring = ""

    for char in s:
        current_substring += char
        if char in PUNCTUATION:
            substrings.append(current_substring.strip())
            current_substring = ""

    if current_substring:
        substrings.append(current_substring)

    return substrings


def remove_punctuation_only_substrings(substrings):
    new_substrings = []
    for substring in substrings:
        if not all(c in PUNCTUATION + SPACE for c in substring):
            new_substrings.append(substring)
    return new_substrings


def clean_punctuation(text):
    text = replace_some_punctuation_with_comma(text)
    substrings = split_string_on_punctuation(text)
    substrings = remove_punctuation_only_substrings(substrings)
    text = SPACE.join(substrings)
    return text


def clean_input_text(text):
    text = text.lower().strip()
    text = clean_punctuation(text)

    def replace_numbers(match):
        return num2words(int(match.group()))

    text = re.sub(r"\d+", replace_numbers, text)
    text = " ".join(text.split())  # Remove extra spaces

    if text and text[-1] not in PUNCTUATION:
        text += "."

    return text
