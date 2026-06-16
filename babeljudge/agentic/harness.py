"""Agentic evaluation harness.

Mirrors babeljudge.harness: runs AgentJudge over gold-labeled AgentJudgeItems
under both presentation orders, collects AgentJudgments, and computes
per-judge reliability cards.
"""
from __future__ import annotations

import json
from collections import defaultdict

from .judges import AgentJudge
from .metrics import reliability_card
from .schema import AgentJudgeItem, AgentJudgment

ORDERS = ("ref_first", "pert_first")


def _present(item: AgentJudgeItem, order: str):
    if order == "ref_first":
        return item.reference_trace, item.perturbed_trace
    return item.perturbed_trace, item.reference_trace


def _normalize(order: str, raw_choice: str) -> str:
    if raw_choice == "tie":
        return "tie"
    if order == "ref_first":
        return "reference" if raw_choice == "A" else "perturbed"
    else:
        return "perturbed" if raw_choice == "A" else "reference"


def run_agent_judge(
    judge: AgentJudge,
    items: list[AgentJudgeItem],
) -> list[AgentJudgment]:
    """Run judge over items × orders; return list of AgentJudgments."""
    judgments: list[AgentJudgment] = []
    for item in items:
        for order in ORDERS:
            a_trace, b_trace = _present(item, order)
            try:
                raw = judge.compare(a_trace, b_trace)
                choice = _normalize(order, raw)
            except Exception as e:
                raw, choice = f"ERROR: {e}", "error"
            judgments.append(AgentJudgment(
                item_id=item.item_id,
                judge=judge.name,
                order=order,
                choice=choice,
                raw=str(raw),
                focus=judge.focus,
            ))
    return judgments


def evaluate_agent(judges: list[AgentJudge], items: list[AgentJudgeItem]) -> dict:
    """Run all judges over all items; return cards and leaderboard."""
    items_by_id = {it.item_id: it for it in items}
    results = {"cards": [], "leaderboard": []}
    for judge in judges:
        judgments = run_agent_judge(judge, items)
        card = reliability_card(judge.name, judge.focus, judgments, items_by_id)
        results["cards"].append(card)
    results["leaderboard"] = sorted(
        results["cards"], key=lambda c: c["reliability_score"], reverse=True
    )
    return results


def agent_cards_to_markdown(results: dict) -> str:
    lines = [
        "# BabelJudge Agentic Reliability Report\n",
        "Evaluates LLM-as-a-judge reliability on agent trajectories. "
        "Gold labels are free: the reference trajectory is always the correct one; "
        "the perturbed trajectory has a controlled defect (wrong args, swapped tool, "
        "hallucinated call, missing step, etc.).\n",
        "| Rank | Judge | Focus | Tool Acc | Halluc Det | Order Consist | Reliability |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, c in enumerate(results["leaderboard"], 1):
        halluc = f"{c['hallucination_detection']:.3f}" if c["hallucination_detection"] != "nan" else "—"
        lines.append(
            f"| {i} | {c['judge']} | {c['focus']} | {c['tool_accuracy']:.3f} | "
            f"{halluc} | {c['order_consistency']:.3f} | {c['reliability_score']:.3f} |"
        )
    lines.append("\n## Per-judge detail\n")
    lines.append("| Judge | Focus | N | Tool Acc | Arg Acc | Halluc Det | "
                 "PosBias | TrajBias | Order Consist | Reliability |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in results["cards"]:
        arg_acc = f"{c['argument_accuracy']:.3f}" if c["argument_accuracy"] != "nan" else "—"
        halluc = f"{c['hallucination_detection']:.3f}" if c["hallucination_detection"] != "nan" else "—"
        traj = f"{c['trajectory_length_bias']:.3f}" if c["trajectory_length_bias"] != "nan" else "—"
        lines.append(
            f"| {c['judge']} | {c['focus']} | {c['n_items']} | "
            f"{c['tool_accuracy']:.3f} | {arg_acc} | {halluc} | "
            f"{c['position_bias_delta']:+.3f} | {traj} | "
            f"{c['order_consistency']:.3f} | {c['reliability_score']:.3f} |"
        )
    return "\n".join(lines)


def save_agent_results(results: dict, json_path: str, md_path: str) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(agent_cards_to_markdown(results))
