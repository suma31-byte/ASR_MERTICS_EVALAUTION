"""Seeded corruptions: the counts must mean exactly what the report claims."""

import pytest


def test_a_perfect_match_scores_zero(ev):
    ref = "yes we have crop loans in assam"
    m = ev.compute_full_metrics(ref, ref)
    assert (m["WER"], m["CER"], m["SER"]) == (0.0, 0.0, 0)


def test_one_deletion_in_ten_words_is_ten_percent(ev):
    # Plain words, not number words: those get rewritten by the normaliser, so
    # using them here would test normalisation rather than the alignment.
    ref = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
    m = ev.compute_full_metrics(ref, "alpha bravo charlie delta echo foxtrot golf hotel india")
    assert m["WER"] == pytest.approx(10.0)
    assert (m["substitutions"], m["deletions"], m["insertions"]) == (0, 1, 0)


def test_one_substitution_is_counted_as_a_substitution(ev):
    m = ev.compute_full_metrics("alpha bravo charlie delta echo",
                                "alpha bravo wrong delta echo")
    assert (m["substitutions"], m["deletions"], m["insertions"]) == (1, 0, 0)
    assert m["WER"] == pytest.approx(20.0)


def test_one_insertion_is_counted_as_an_insertion(ev):
    m = ev.compute_full_metrics("alpha bravo charlie delta",
                                "alpha bravo extra charlie delta")
    assert (m["substitutions"], m["deletions"], m["insertions"]) == (0, 0, 1)
    assert m["WER"] == pytest.approx(25.0)


def test_an_empty_hypothesis_is_a_total_failure(ev):
    m = ev.compute_full_metrics("some reference words here", "")
    assert m["WER"] == 100.0 and m["SER"] == 1
    assert m["deletions"] == 4


def test_a_spoken_number_is_not_an_error(ev):
    m = ev.compute_full_metrics("the appointment is on 15 june",
                                "the appointment is on fifteenth june")
    assert m["WER"] == 0.0


@pytest.mark.parametrize("wer,label", [
    (2.0, "Excellent"), (9.0, "Good"), (20.0, "Acceptable"),
    (30.0, "Poor"), (55.0, "Very Poor"),
])
def test_quality_bands(ev, wer, label):
    assert ev.label_wer(wer) == label
