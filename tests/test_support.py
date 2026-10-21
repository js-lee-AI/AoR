import pytest

from aor import Example, PassageCheck, normalize, supported
from aor.support import content_recall


def test_normalize():
    assert normalize("The Eiffel Tower!") == "eiffel tower"
    assert normalize("an apple, a pear") == "apple pear"
    assert normalize("Anglo - Irish") == "anglo irish"
    # marks outside ASCII stay as tokens of their own
    assert normalize("10–12 °C") == "10 – 12 ° c"


@pytest.mark.parametrize("answer,expected", [
    ("1971", True),
    ("Ireland in 1971.", False),  # the whole answer must occur, not a part of it
    ("the Irish Free State", True),
    ("Irish Free", True),
    ("Free Stat", False),         # whole tokens only
    ("1922", False),
    ("", False),
])
def test_supported(answer, expected):
    passage = "In 1971 , after the Irish War of Independence , most of Ireland became the Irish Free State ."
    assert supported(answer, passage) is expected


def test_first_line_only():
    assert supported("1971\nbecause the passage says so", "It happened in 1971 .")
    assert not supported("Sure.\n1971", "It happened in 1971 .")


def test_yes_no_count_as_supported():
    check = PassageCheck(Example("Were both films released in 1999?", "Film A (1999). Film B (1999)."))
    assert check.supports("Yes")
    assert check.supports("no.")
    assert not check.supports("maybe")


def mc(passage, options=("a rock", "a crystal", "a mineral", "a bone")):
    return PassageCheck(Example("Which would most likely form a fossil?", passage, options=options))


def test_multiple_choice_support_needs_a_unique_option():
    one = mc("Minerals replace the tissue; a mineral cast remains.")
    assert one.occurring == {"C"} and one.acceptable == {"C"}
    assert one.supports("C") and one.supports("C. a mineral") and not one.supports("D")
    two = mc("It lies in sedimentary rock, where a mineral cast forms.")
    assert two.occurring == {"A", "C"} and two.acceptable == frozenset()
    assert not two.supports("C")


def test_veto_free_form():
    check = PassageCheck(Example("when did ireland gain its independence", "In 1971 , Ireland seceded ."))
    assert check.vetoes("1922")
    assert not check.vetoes("1971")
    assert not check.vetoes("")          # m states no answer
    assert check.masked_letter("1922") is None


def test_veto_multiple_choice():
    # m is A and only option C's text occurs: the veto masks A
    check = mc("A mineral cast remains after millions of years.")
    assert check.vetoes("A") and check.masked_letter("A") == "A"
    # an option whose text occurs scores highest, so it is never masked
    assert not check.vetoes("C") and check.masked_letter("C") is None
    # a reply without a letter is no answer
    assert not check.vetoes("I am not sure")


def test_veto_on_a_paraphrased_passage():
    # no option text occurs; the veto holds only when another option has more of its words in the passage
    opts = ("rising sea levels", "falling temperatures", "more volcanic activity", "less rainfall")
    check = mc("The region grew cold over centuries as temperatures dropped.", opts)
    assert check.occurring == frozenset()
    assert check.option_scores()["B"] == pytest.approx(0.5)
    assert check.vetoes("A")
    assert not check.vetoes("B")


def test_content_recall_ignores_function_words():
    assert content_recall("the shape of the coast", normalize("A coast with a sharp shape.")) == 1.0
    assert content_recall("of the", normalize("anything")) == 0.0
