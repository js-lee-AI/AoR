from aor import Example, accept_or_repair
from aor.demo import CASES, headline, replay

FF = Example("when did ireland gain its independence from england",
             "In 1971 , most of Ireland seceded from the United Kingdom .", uid=1)
MC = Example("Which would most likely form a fossil?", "A mineral cast remains after millions of years.",
             options=("a rock", "a crystal", "a mineral", "a bone"), uid=2)


class Scripted:
    """A batch decoder with one fixed reply per example index, which records how it was called."""

    def __init__(self, replies, masked_replies=None):
        self.replies, self.masked_replies, self.calls = replies, masked_replies or {}, []

    def __call__(self, idx, masked=None):
        masked = masked or [None] * len(idx)
        self.calls.append((list(idx), list(masked)))
        return [self.masked_replies.get(i, self.replies[i]) if c else self.replies[i] for i, c in zip(idx, masked)]


def run(examples, sample, closed, cad, idr):
    return accept_or_repair(examples, Scripted(sample), closed, [("cad", cad), ("id", idr)])


def test_accept_draws_nothing_else():
    closed, cad, idr = Scripted(["1922"]), Scripted(["1971"]), Scripted(["1971"])
    [t] = run([FF], ["1971"], closed, cad, idr)
    assert (t.answer, t.action, t.decoder) == ("1971", "accept", "sampling")
    assert closed.calls == cad.calls == idr.calls == []


def test_cad_repair():
    cad, idr = Scripted(["1971"]), Scripted(["1971"])
    [t] = run([FF], ["1922"], Scripted(["1922"]), cad, idr)
    assert (t.answer, t.action, t.decoder, t.veto) == ("1971", "repair", "cad", True)
    assert idr.calls == []                      # the first supported repair ends the example


def test_vetoed_repair_is_skipped():
    [t] = run([FF], ["1922"], Scripted(["1922"]), Scripted(["1922."]), Scripted(["1971"]))
    assert (t.answer, t.decoder) == ("1971", "id")
    assert [d.rejected for d in t.repairs] == [True, False]


def test_fallback_and_initial():
    # no repair passes the support test: the first one that states an answer and is not vetoed is returned
    [t] = run([FF], ["1922"], Scripted(["1922"]), Scripted(["1922"]), Scripted(["1916"]))
    assert (t.answer, t.action, t.decoder) == ("1916", "fallback", "id")
    # every repair repeats the vetoed m or is empty: the initial answer stays
    [t] = run([FF], ["1922"], Scripted(["1922"]), Scripted(["1922"]), Scripted([""]))
    assert (t.answer, t.action, t.decoder) == ("1922", "initial", "sampling")


def test_no_veto_when_closed_book_answer_is_supported():
    [t] = run([FF], ["the UK"], Scripted(["1971"]), Scripted(["1971"]), Scripted(["1971"]))
    assert t.veto is False and t.answer == "1971"


def test_multiple_choice_masks_the_vetoed_letter():
    cad = Scripted(["A"], masked_replies={0: "C"})
    [t] = run([MC], ["A"], Scripted(["A"]), cad, Scripted(["A"]))
    assert cad.calls == [([0], ["A"])]
    assert (t.answer, t.action, t.masked_letter) == ("C", "repair", "A")


def test_multiple_choice_stops_when_no_option_can_pass():
    two = Example("q", "It lies in sedimentary rock, where a mineral cast forms.",
                  options=("a rock", "a crystal", "a mineral", "a bone"))
    idr = Scripted(["C"])
    [t] = run([two], ["D"], Scripted(["D"]), Scripted(["D"], masked_replies={0: "B"}), idr)
    assert (t.answer, t.action, t.decoder) == ("B", "fallback", "cad")
    assert idr.calls == []                      # no later repair could pass the support test


def test_batches_only_pending_examples():
    sample = ["1971", "B"]
    closed = Scripted(["1922", "A"])
    cad = Scripted(["1971", "A"], masked_replies={1: "C"})
    traces = accept_or_repair([FF, MC], Scripted(sample), closed, [("cad", cad), ("id", Scripted(["x", "x"]))])
    assert closed.calls == [([1], [None])]
    assert cad.calls == [([1], ["A"])]
    assert [t.action for t in traces] == ["accept", "repair"]


def test_demo_replays_the_paper_examples():
    got = [(t.answer, t.action, t.decoder) for t in replay()]
    assert got == [("111", "accept", "sampling"), ("1971", "repair", "cad"), ("Neal Dahlen", "repair", "id"),
                   ("C. a mineral", "fallback", "cad")]
    assert [c["example"].uid for c in CASES] == [308, 912, 3674, 948]
    assert [headline(t) for t in replay()] == ["accepted", "CAD repair", "ID repair", "D vetoed, CAD fallback"]
