"""Reliability metrics for agentic evaluation.

Mirrors babeljudge.metrics but operates on AgentJudgment objects and adds
agentic-specific metrics:

tool_accuracy
    Order-balanced rate at which the judge correctly prefers the reference
    trajectory. Equivalent to accuracy in the text-based benchmark.

argument_accuracy
    Same as tool_accuracy but restricted to `argument_corrupt` and
    `missing_required_arg` items. Measures how well the judge catches
    argument-level errors specifically.

trajectory_length_bias
    On `step_pad` items (where the perturbed trace is longer-but-worse):
    the rate the judge prefers the longer trace. Analogous to verbosity_susceptibility.

hallucination_detection
    On `hallucinated_tool` items: the rate the judge correctly prefers the
    reference (non-hallucinated) trace. High = good hallucination detection.

order_consistency
    Fraction of items with matching verdicts under both presentation orders.

reliability_score
    Single rankable number: competence gated by bias penalties.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .schema import AgentJudgment


def _by_item(judgments: Iterable[AgentJudgment]) -> dict[str, dict[str, AgentJudgment]]:
    grouped: dict[str, dict[str, AgentJudgment]] = defaultdict(dict)
    for j in judgments:
        grouped[j.item_id][j.order] = j
    return grouped


def tool_accuracy(judgments: list[AgentJudgment]) -> float:
    wins = sum(1 for j in judgments if j.choice == "reference")
    scored = sum(1 for j in judgments if j.choice in ("reference", "perturbed"))
    return wins / scored if scored else float("nan")


def argument_accuracy(judgments: list[AgentJudgment], items_by_id: dict) -> float:
    arg_perturbations = {"argument_corrupt", "missing_required_arg"}
    rel = [j for j in judgments
           if items_by_id.get(j.item_id)
           and items_by_id[j.item_id].perturbation in arg_perturbations
           and j.choice in ("reference", "perturbed")]
    if not rel:
        return float("nan")
    return sum(1 for j in rel if j.choice == "reference") / len(rel)


def trajectory_length_bias(judgments: list[AgentJudgment], items_by_id: dict) -> float:
    """Rate of preferring the longer-but-worse trajectory (step_pad items)."""
    rel = [j for j in judgments
           if items_by_id.get(j.item_id)
           and items_by_id[j.item_id].longer_is_perturbed
           and j.choice in ("reference", "perturbed")]
    if not rel:
        return float("nan")
    return sum(1 for j in rel if j.choice == "perturbed") / len(rel)


def hallucination_detection(judgments: list[AgentJudgment], items_by_id: dict) -> float:
    """Rate of correctly preferring non-hallucinated traces."""
    rel = [j for j in judgments
           if items_by_id.get(j.item_id)
           and items_by_id[j.item_id].perturbation == "hallucinated_tool"
           and j.choice in ("reference", "perturbed")]
    if not rel:
        return float("nan")
    return sum(1 for j in rel if j.choice == "reference") / len(rel)


def position_bias_delta(judgments: list[AgentJudgment]) -> float:
    def rate(order: str) -> float:
        js = [j for j in judgments
              if j.order == order and j.choice in ("reference", "perturbed")]
        return sum(1 for j in js if j.choice == "reference") / len(js) if js else float("nan")
    r_first = rate("ref_first")
    p_first = rate("pert_first")
    if r_first != r_first or p_first != p_first:
        return float("nan")
    return r_first - p_first


def order_consistency(judgments: list[AgentJudgment]) -> float:
    grouped = _by_item(judgments)
    paired = [d for d in grouped.values() if "ref_first" in d and "pert_first" in d]
    if not paired:
        return float("nan")
    return sum(1 for d in paired if d["ref_first"].choice == d["pert_first"].choice) / len(paired)


def _nan_to(x: float, default: float) -> float:
    return default if x != x else x


def reliability_score(
    tool_acc: float,
    pos_delta: float,
    traj_bias: float,
    consistency: float,
    halluc_det: float,
) -> float:
    """Agentic reliability score in roughly [0, 1].

    Competence = tool accuracy + order consistency.
    Bias penalties: position bias, trajectory-length bias, hallucination misses.
    """
    acc = _nan_to(tool_acc, 0.0)
    cons = _nan_to(consistency, 0.0)
    pos_pen = min(1.0, abs(_nan_to(pos_delta, 0.0)))
    len_pen = max(0.0, (_nan_to(traj_bias, 0.5) - 0.5) * 2)
    # hallucination: 1 = perfect detection, 0 = never catches it
    halluc_miss = 1.0 - _nan_to(halluc_det, 0.5)
    base = 0.6 * acc + 0.4 * cons
    return base * (1 - 0.7 * pos_pen) * (1 - 0.6 * len_pen) * (1 - 0.5 * halluc_miss)


def reliability_card(
    judge: str,
    focus: str,
    judgments: list[AgentJudgment],
    items_by_id: dict,
) -> dict:
    """Assemble a per-judge agentic reliability card."""
    n_items = len({j.item_id for j in judgments})
    t_acc = tool_accuracy(judgments)
    arg_acc = argument_accuracy(judgments, items_by_id)
    pos = position_bias_delta(judgments)
    traj = trajectory_length_bias(judgments, items_by_id)
    halluc = hallucination_detection(judgments, items_by_id)
    cons = order_consistency(judgments)
    rscore = reliability_score(t_acc, pos, traj, cons, halluc)
    return {
        "judge": judge,
        "focus": focus,
        "n_items": n_items,
        "n_judgments": len(judgments),
        "tool_accuracy": round(t_acc, 4),
        "argument_accuracy": round(arg_acc, 4) if arg_acc == arg_acc else "nan",
        "hallucination_detection": round(halluc, 4) if halluc == halluc else "nan",
        "position_bias_delta": round(pos, 4),
        "trajectory_length_bias": round(traj, 4) if traj == traj else "nan",
        "order_consistency": round(cons, 4),
        "reliability_score": round(rscore, 4),
    }
