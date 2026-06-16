"""Agentic evaluation demo — zero network, zero GPU required.

Demonstrates BabelJudge's agentic extension using MockJudge and synthetic
tool-call tasks. Run from the repository root:

    python examples/run_agentic_demo.py

What this shows:
  - 8 synthetic tool-call tasks (weather, stocks, scheduling, DB queries, ...)
  - 9 agentic perturbation types (wrong args, swapped tools, hallucinated calls, ...)
  - Dual-order presentation to expose position bias
  - Per-judge agentic reliability card with new metrics:
      tool_accuracy, argument_accuracy, hallucination_detection,
      trajectory_length_bias, order_consistency, reliability_score
"""
import json
import sys
from pathlib import Path

# allow running from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import babeljudge as bj
from babeljudge.agentic import (
    synthetic_tool_tasks,
    build_agent_dataset,
    AgentJudge,
    evaluate_agent,
    agent_cards_to_markdown,
    AGENT_PERTURBATIONS,
)


def main():
    print("=" * 60)
    print("BabelJudge — Agentic Evaluation Demo")
    print("=" * 60)

    # ── 1. Load reference traces ──────────────────────────────────────────────
    traces = synthetic_tool_tasks()
    print(f"\n[1] Loaded {len(traces)} synthetic tool-call tasks")
    for t in traces[:3]:
        n_calls = len(t.tool_calls)
        print(f"    {t.task_id}: {n_calls} tool call(s) — {t.task_description[:60]}...")

    # ── 2. Build gold-labeled dataset ─────────────────────────────────────────
    perturbations = list(AGENT_PERTURBATIONS.keys())
    items = build_agent_dataset(
        traces,
        perturbations=perturbations,
        severities=(0.4, 0.7),
    )
    print(f"\n[2] Built {len(items)} gold-labeled items")
    print(f"    Perturbations: {', '.join(perturbations)}")
    print(f"    Severities: 0.4, 0.7")

    # show one item as an example
    ex = items[0]
    print(f"\n    Example item: {ex.item_id}")
    print(f"      Task: {ex.task_description[:70]}")
    print(f"      Perturbation: {ex.perturbation} (severity={ex.severity})")
    ref_calls = ex.reference_trace.tool_calls
    pert_calls = ex.perturbed_trace.tool_calls
    print(f"      Reference: {len(ref_calls)} tool call(s)")
    print(f"      Perturbed: {len(pert_calls)} tool call(s)")
    if ref_calls and pert_calls:
        rc, pc = ref_calls[0], pert_calls[0]
        if rc.tool_name != pc.tool_name:
            print(f"      Tool name: {rc.tool_name!r} -> {pc.tool_name!r}")
        elif rc.arguments != pc.arguments:
            print(f"      Args changed: {rc.arguments} -> {pc.arguments}")

    # ── 3. Set up judges ──────────────────────────────────────────────────────
    # MockJudge randomly picks A or B — it shows what a random-baseline judge
    # looks like in the reliability framework (expect ~0.5 accuracy, high bias).
    mock_judge = bj.create_judge("mock", name="RandomBaseline")

    # Two modes: tool_calls (fast, token-efficient) vs full_trace (full context)
    judge_tool = AgentJudge(mock_judge, focus="tool_calls",
                            name="RandomBaseline[tool_calls]")
    judge_trace = AgentJudge(mock_judge, focus="full_trace",
                             name="RandomBaseline[full_trace]")

    print("\n[3] Judges configured:")
    print(f"    {judge_tool.name}")
    print(f"    {judge_trace.name}")

    # ── 4. Run evaluation ─────────────────────────────────────────────────────
    print(f"\n[4] Running evaluation ({len(items)} items × 2 orders × 2 judges = "
          f"{len(items) * 2 * 2} judge calls)...")
    results = evaluate_agent([judge_tool, judge_trace], items)

    # ── 5. Print results ──────────────────────────────────────────────────────
    print("\n[5] Results\n")
    print(agent_cards_to_markdown(results))

    # ── 6. Save outputs ───────────────────────────────────────────────────────
    out_json = "agentic_demo_results.json"
    out_md = "agentic_demo_leaderboard.md"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(out_md, "w") as f:
        f.write(agent_cards_to_markdown(results))

    print(f"\n[6] Saved: {out_json}, {out_md}")
    print("\nDone! Replace MockJudge with any real judge to get meaningful results:")
    print("  from babeljudge import create_judge")
    print('  judge = AgentJudge(create_judge("anthropic"), focus="tool_calls")')
    print('  results = evaluate_agent([judge], items)')


if __name__ == "__main__":
    main()
