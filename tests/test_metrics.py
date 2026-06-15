"""Regression tests: the metrics must recover injected biases.

Run with:  PYTHONPATH=. python3 tests/test_metrics.py
(kept dependency-free so it runs without pytest)
"""
from babeljudge import build_dataset, evaluate, MockJudge

SOURCES = [
    {"id": f"q{i}", "language": lang,
     "prompt": f"prompt {i}",
     "reference": "Alpha beta gamma delta. Epsilon zeta eta theta. "
                  "Iota kappa lambda mu. Nu xi omicron pi."}
    for i, lang in enumerate(["en", "en", "es", "sw"])
]


def almost(x, lo, hi, msg):
    assert lo <= x <= hi, f"{msg}: expected [{lo},{hi}], got {x}"


def test_biases_recovered():
    items = build_dataset(SOURCES, severities=(0.25, 0.4, 0.55, 0.7), seed=7)
    judges = [
        MockJudge("fair", skill=0.9, position_bias=0.0, verbosity_bias=0.0, seed=1),
        MockJudge("slotA", skill=0.9, position_bias=0.4, verbosity_bias=0.0, seed=2),
        MockJudge("verbose", skill=0.9, position_bias=0.0, verbosity_bias=0.5, seed=3),
    ]
    res = evaluate(judges, items)
    cards = {(c["judge"], c["language"]): c for c in res["cards"]}

    # position bias: slotA must show a much larger |delta| than fair
    fair_pos = max(abs(cards[("fair", l)]["position_bias_delta"]) for l in ["en", "es", "sw"])
    slota_pos = min(abs(cards[("slotA", l)]["position_bias_delta"]) for l in ["en", "es", "sw"])
    assert slota_pos > fair_pos, (slota_pos, fair_pos)

    # verbosity: verbose must be fooled more than fair on repeat_pad items
    fair_v = max(cards[("fair", l)]["verbosity_susceptibility"] for l in ["en", "es", "sw"])
    verbose_v = min(cards[("verbose", l)]["verbosity_susceptibility"] for l in ["en", "es", "sw"])
    assert verbose_v > fair_v, (verbose_v, fair_v)

    # fair judge should top the reliability leaderboard
    assert res["leaderboard"][0]["judge"] == "fair", res["leaderboard"]
    print("OK: position bias, verbosity bias, and leaderboard ranking all recovered.")


if __name__ == "__main__":
    test_biases_recovered()
