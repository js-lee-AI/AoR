"""Table 1 and the per-model Means of the paper, rebuilt from the per-example scores in records/."""

import pytest

np = pytest.importorskip("numpy")

from aor import table  # noqa: E402
from aor.records import load_splits  # noqa: E402

MODELS = ["qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b", "llama3.1-8b", "mistral-7b"]

# NQ-Swap, FaithEval, counterfactual avg., NQ-Orig., SQuAD, HotpotQA, ordinary avg., Mean, Cost
TABLE1 = {
    "sampling": (56.9, 63.6, 60.3, 76.0, 84.5, 63.5, 74.7, 67.5, 1.00),
    "cad": (62.4, 73.7, 68.1, 72.5, 84.6, 60.8, 72.6, 70.3, 1.18),
    "adacad": (61.4, 65.5, 63.4, 75.6, 85.5, 63.2, 74.8, 69.1, 1.18),
    "coiecd": (62.2, 73.7, 68.0, 72.7, 84.5, 61.1, 72.8, 70.4, 1.18),
    "nwcad": (57.3, 63.8, 60.6, 76.0, 84.5, 63.7, 74.8, 67.7, 1.18),
    "rpcd": (61.0, 71.1, 66.1, 75.4, 85.3, 62.5, 74.4, 70.2, 2.00),
    "id": (64.3, 67.8, 66.0, 74.8, 85.6, 63.6, 74.7, 70.4, 2.01),
    "strict": (60.3, 64.8, 62.5, 76.0, 84.9, 63.1, 74.6, 68.6, 1.03),
    "strict_cad": (64.7, 75.1, 69.9, 70.9, 84.1, 58.3, 71.1, 70.5, 1.21),
    "aor": (65.2, 80.4, 72.8, 78.2, 87.5, 65.2, 77.0, 74.9, 1.71),
}
# difference from sampling, and whether its 95% interval excludes zero (colored in the paper)
DIFFS = {
    "cad": ((7.8, True), (-2.1, True)), "adacad": ((3.2, True), (0.1, False)),
    "coiecd": ((7.7, True), (-1.9, True)), "nwcad": ((0.3, True), (0.1, False)),
    "rpcd": ((5.8, True), (-0.3, False)), "id": ((5.8, True), (0.0, False)),
    "strict": ((2.2, True), (0.0, False)), "strict_cad": ((9.6, True), (-3.6, True)),
    "aor": ((12.5, True), (2.3, True)),
}
PER_MODEL = {
    "sampling": (65.1, 68.1, 72.2, 72.2, 69.5, 57.7), "cad": (65.3, 70.6, 76.2, 75.5, 72.3, 62.3),
    "adacad": (65.9, 69.4, 73.3, 73.2, 71.9, 60.9), "coiecd": (65.5, 70.6, 76.1, 75.5, 72.4, 62.2),
    "nwcad": (65.4, 68.3, 72.3, 72.2, 69.8, 57.9), "rpcd": (65.8, 70.4, 74.3, 74.6, 71.2, 65.1),
    "id": (62.9, 71.9, 75.5, 75.4, 71.4, 65.1), "strict": (64.1, 68.5, 73.0, 73.0, 73.3, 59.6),
    "strict_cad": (64.1, 70.5, 76.1, 75.8, 72.7, 63.9), "aor": (69.0, 74.7, 79.0, 80.3, 77.7, 68.6),
}


@pytest.fixture(scope="module")
def rows():
    return table.from_records(table.RECORDS, MODELS)


def test_splits():
    splits = load_splits(table.RECORDS / "splits.json")
    assert {t: len(s["scored"]) for t, s in splits.items()} == \
        {"nqswap": 724, "faitheval": 800, "nqorig": 724, "squad": 724, "hotpotqa": 800}
    for s in splits.values():
        assert len(s["dev"]) == 200 and len(s["test"]) == 800 and len(s["clusters"]) == len(s["scored"])
        assert set(s["scored"]) <= set(s["test"]) and not set(s["dev"]) & set(s["test"])


@pytest.mark.parametrize("method", list(TABLE1))
def test_table1(rows, method):
    r = rows[method]
    got = (r["tasks"]["nqswap"], r["tasks"]["faitheval"], r["groups"]["counterfactual"], r["tasks"]["nqorig"],
           r["tasks"]["squad"], r["tasks"]["hotpotqa"], r["groups"]["ordinary"], r["mean"])
    assert [round(x, 1) for x in got] == list(TABLE1[method][:8])
    assert round(r["cost"], 2) == TABLE1[method][8]
    assert [round(r["per_model"][m], 1) for m in MODELS] == list(PER_MODEL[method])


@pytest.mark.parametrize("method", list(DIFFS))
def test_differences_from_sampling(rows, method):
    vs = rows[method]["vs_base"]
    for group, (delta, significant) in zip(("counterfactual", "ordinary"), DIFFS[method]):
        assert round(vs[group]["delta"], 1) == delta
        assert vs[group]["significant"] is significant


def test_aor_beats_every_baseline_on_both_groups():
    from aor.records import load_paper_records
    from aor.stats import paired_bootstrap

    scores, clusters, _ = load_paper_records(table.RECORDS, MODELS, table.TASKS, list(TABLE1))
    for base in TABLE1:
        if base == "aor":
            continue
        ci = paired_bootstrap(scores, "aor", base, MODELS, clusters)
        assert ci["counterfactual"]["lo"] > 0 and ci["ordinary"]["lo"] > 0 and ci["mean"]["lo"] > 0, base
