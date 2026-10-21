import importlib.util
import json
import re
from pathlib import Path

from aor.demo import CASES, replay

DEMO = Path(__file__).resolve().parents[1] / "demo"


def page_data():
    page = (DEMO / "replay.html").read_text(encoding="utf-8")
    return json.loads(re.search(r'id="replay-data">\n(.*?)\n</script>', page, re.S).group(1))


def test_page_holds_the_current_replay():
    spec = importlib.util.spec_from_file_location("make_replay", DEMO / "make_replay.py")
    make_replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_replay)
    assert page_data() == json.loads(make_replay.render())     # rerun demo/make_replay.py after changing aor.demo


def test_page_shows_the_recorded_answers_and_what_aor_did():
    for case, trace, shown in zip(CASES, replay(), page_data()["cases"], strict=True):
        drawn = [trace.initial]
        if trace.action != "accept":
            drawn += [trace.closed_book] + [d.text for d in trace.repairs]
        assert shown["aor"]["reply"] == trace.answer
        assert [s["reply"] for s in shown["aor"]["steps"]] == drawn
        assert {b["name"]: b["reply"] for b in shown["baselines"]} == {"CAD": case["answers"]["cad"],
                                                                       "Sampling": case["answers"]["sampling"]}
