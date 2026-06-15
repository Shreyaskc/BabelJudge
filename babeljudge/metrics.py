"""Reliability metrics for LLM-as-a-judge evaluation.

All metrics are pure-Python (no numpy dependency) so the harness runs anywhere.
They are computed from a list of `Judgment` objects produced by running each
item under both presentation orders.

Definitions
-----------
accuracy
    Order-balanced rate at which the judge picks the reference (the gold-better
    response). Averaged over both orders so raw position bias cannot inflate it.

position_bias_delta
    ref-win-rate when reference is shown first MINUS ref-win-rate when shown
    second. ~0 is good; large magnitude means the judge favors a slot.

verbosity_susceptibility
    On `repeat_pad` items only (where the perturbed text is longer-but-worse):
    the rate at which the judge prefers the longer perturbed response. High =
    the judge is fooled by length.

order_consistency
    Fraction of items for which the judge makes the SAME underlying choice under
    both orders. A judge that flips when you swap slots is internally
    inconsistent regardless of accuracy.

cohen_kappa / fleiss_kappa
    Chance-corrected agreement, for judge-vs-gold and multi-judge jury analysis.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .schema import Judgment


def _by_item(judgments: Iterable[Judgment]) -> dict[str, dict[str, Judgment]]:
    """Group judgments by item_id, keyed by presentation order."""
    grouped: dict[str, dict[str, Judgment]] = defaultdict(dict)
    for j in judgments:
        grouped[j.item_id][j.order] = j
    return grouped


def accuracy(judgments: list[Judgment]) -> float:
    wins = sum(1 for j in judgments if j.choice == "reference")
    scored = sum(1 for j in judgments if j.choice in ("reference", "perturbed"))
    return wins / scored if scored else float("nan")


def position_bias_delta(judgments: list[Judgment]) -> float:
    def rate(order: str) -> float:
        js = [j for j in judgments if j.order == order and j.choice in ("reference", "perturbed")]
        if not js:
            return float("nan")
        return sum(1 for j in js if j.choice == "reference") / len(js)
    r_first = rate("ref_first")
    p_first = rate("pert_first")
    if r_first != r_first or p_first != p_first:  # NaN guard
        return float("nan")
    return r_first - p_first


def verbosity_susceptibility(judgments: list[Judgment], items_by_id: dict) -> float:
    """Rate of preferring the longer-but-worse response, on repeat_pad items."""
    rel = [j for j in judgments
           if items_by_id.get(j.item_id) and items_by_id[j.item_id].longer_is_perturbed
           and j.choice in ("reference", "perturbed")]
    if not rel:
        return float("nan")
    fooled = sum(1 for j in rel if j.choice == "perturbed")
    return fooled / len(rel)


def order_consistency(judgments: list[Judgment]) -> float:
    grouped = _by_item(judgments)
    paired = [d for d in grouped.values() if "ref_first" in d and "pert_first" in d]
    if not paired:
        return float("nan")
    same = sum(1 for d in paired if d["ref_first"].choice == d["pert_first"].choice)
    return same / len(paired)


def cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    """Cohen's kappa between two raters over a list of (a, b) category pairs."""
    if not pairs:
        return float("nan")
    n = len(pairs)
    cats = sorted({c for ab in pairs for c in ab})
    po = sum(1 for a, b in pairs if a == b) / n
    pa = {c: sum(1 for a, _ in pairs if a == c) / n for c in cats}
    pb = {c: sum(1 for _, b in pairs if b == c) / n for c in cats}
    pe = sum(pa[c] * pb[c] for c in cats)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def fleiss_kappa(rating_table: list[dict[str, int]]) -> float:
    """Fleiss' kappa. `rating_table[i][cat]` = # raters assigning item i to cat.

    Used for jury/ensemble analysis when multiple judges score the same items.
    """
    if not rating_table:
        return float("nan")
    cats = sorted({c for row in rating_table for c in row})
    n_items = len(rating_table)
    n_raters = sum(rating_table[0].get(c, 0) for c in cats)
    if n_raters <= 1:
        return float("nan")
    p_cat = {c: 0.0 for c in cats}
    for row in rating_table:
        for c in cats:
            p_cat[c] += row.get(c, 0)
    total = n_items * n_raters
    p_cat = {c: v / total for c, v in p_cat.items()}
    p_bar = 0.0
    for row in rating_table:
        s = sum(row.get(c, 0) ** 2 for c in cats)
        p_bar += (s - n_raters) / (n_raters * (n_raters - 1))
    p_bar /= n_items
    p_e = sum(v ** 2 for v in p_cat.values())
    return (p_bar - p_e) / (1 - p_e) if p_e != 1 else 1.0


def _nan_to(x: float, default: float) -> float:
    return default if x != x else x


def reliability_score(accuracy_: float, pos_delta: float,
                      verb_susc: float, consistency: float) -> float:
    """A single rankable reliability number in roughly [0, 1].

    Competence base = accuracy and order-consistency. Bias terms GATE that base
    multiplicatively, so a judge that is accurate-but-biased is correctly
    penalized -- which raw accuracy fails to do, since on most perturbations the
    better response is also the longer one. This is the core argument for
    reporting bias probes alongside accuracy rather than a single score.
    """
    acc = _nan_to(accuracy_, 0.0)
    cons = _nan_to(consistency, 0.0)
    pos_pen = min(1.0, abs(_nan_to(pos_delta, 0.0)))           # 0 good, 1 worst
    verb_pen = max(0.0, (_nan_to(verb_susc, 0.5) - 0.5) * 2)   # >chance only
    base = 0.6 * acc + 0.4 * cons
    return base * (1 - 0.8 * pos_pen) * (1 - 0.7 * verb_pen)


def reliability_card(
    judge: str,
    language: str,
    judgments: list[Judgment],
    items_by_id: dict,
) -> dict:
    """Assemble a per-(judge, language) reliability card."""
    n_items = len({j.item_id for j in judgments})
    acc = accuracy(judgments)
    pos = position_bias_delta(judgments)
    verb = verbosity_susceptibility(judgments, items_by_id)
    cons = order_consistency(judgments)
    return {
        "judge": judge,
        "language": language,
        "n_items": n_items,
        "n_judgments": len(judgments),
        "accuracy": round(acc, 4),
        "position_bias_delta": round(pos, 4),
        "verbosity_susceptibility": round(verb, 4),
        "order_consistency": round(cons, 4),
        "reliability_score": round(reliability_score(acc, pos, verb, cons), 4),
    }
