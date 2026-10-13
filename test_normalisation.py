"""Normalisation decides what counts as an error, so it gets the most tests."""

import pytest


@pytest.mark.parametrize("spoken,written", [
    ("fifteen", "15"),
    ("the fifteenth of june", "the 15 of june"),
    ("two thousand and twenty six", "2026"),
    ("twenty six", "26"),
])
def test_numbers_and_their_spoken_forms_match(ev, spoken, written):
    assert ev.normalize(spoken) == ev.normalize(written)


@pytest.mark.xfail(reason=(
    "KNOWN BUG: 'eight twenty' normalises to '820' while '8:20' normalises to "
    "'8 20', so a correct transcript of a time is scored as wrong. The Whisper "
    "normaliser joins the two number words before our own number handling sees "
    "them. Affects the times category."), strict=True)
def test_time_matches_its_spoken_form(ev):
    assert ev.normalize("eight twenty") == ev.normalize("8:20")
    assert ev.normalize("eight twenty") != ev.normalize("28")


@pytest.mark.xfail(reason=(
    "KNOWN BUG: consecutive number words are glued into one token - "
    "'one two three' becomes '123'. Where one side has digits and the other "
    "words, they cannot match. Affects the numbers and phone_numbers "
    "categories."), strict=True)
def test_consecutive_numbers_stay_separate(ev):
    assert ev.normalize("one two three") == ev.normalize("1 2 3")


def test_case_and_punctuation_are_ignored(ev):
    assert ev.normalize("Yes, we have CROP loans!") == ev.normalize("yes we have crop loans")


@pytest.mark.xfail(reason=(
    "KNOWN BUG: a trailing '...' survives the first pass as a bare '.' token "
    "and is scored as a word, costing one insertion or substitution whenever "
    "one side ends in an ellipsis. A second pass removes it, so the function "
    "is not idempotent."), strict=True)
def test_normalisation_is_stable(ev):
    once = ev.normalize("That's why -- she has given 15 reasons...")
    assert ev.normalize(once) == once


@pytest.mark.parametrize("raw,expected", [
    ("Speaker 1: hello how are you", "hello how are you"),
    ("Speaker 1: hello there Speaker 2: i am fine", "hello there i am fine"),
    ("Speaker 1: a\nSpeaker 2: b", "a b"),
    ("SPEAKER 2 - yes we have crop loans", "yes we have crop loans"),
    ("Speaker1:ugadi is a new year day", "ugadi is a new year day"),
])
def test_speaker_labels_are_removed(ev, raw, expected):
    assert ev.strip_speaker_labels(raw) == expected


@pytest.mark.parametrize("text", [
    "please reduce the speaker volume",
    "the speaker was very clear",
])
def test_the_word_speaker_without_a_number_survives(ev, text):
    # One of the reference sentences is "please reduce the speaker volume".
    # A rule that strips the bare word would delete a real word and inflate
    # WER on every sentence containing it.
    assert ev.strip_speaker_labels(text) == text
