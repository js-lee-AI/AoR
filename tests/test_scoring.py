import pytest

from aor import Example, f1_score, parse_letter, score
from aor.scoring import exact_match, first_line, normalize_answer


def test_normalize_answer():
    assert normalize_answer("The  “Beatles”!") == "beatles"


def test_first_line():
    assert first_line("\n\n  Paris \nFrance") == "Paris"
    assert first_line("") == ""


def test_f1():
    assert f1_score("Neal Dahlen", ["Neal Dahlen"]) == 1.0
    assert f1_score("Dahlen", ["Neal Dahlen"]) == pytest.approx(2 / 3)
    assert f1_score("Jay Baruchel.", ["Neal Dahlen"]) == 0.0
    assert f1_score("the 111\nmore text", ["111"]) == 1.0
    assert exact_match("111.", ["111"]) == 1.0


@pytest.mark.parametrize("text,letter", [
    ("C", "C"), ("C. a mineral", "C"), ("(B)", "B"), ("The answer is D.", "D"), ("Option: a", "A"),
    ("I think (C) fits", "C"), ("E", None), ("none of them", None),
])
def test_parse_letter(text, letter):
    assert parse_letter(text, 4) == letter


def test_score():
    assert score("C. a mineral", Example("q", "p", options=("w", "x", "y", "z"), gold_letter="C")) == 1.0
    assert score("1971", Example("q", "p", golds=("1971",))) == 1.0
