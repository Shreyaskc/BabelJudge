"""babeljudge.agentic — reliability benchmarking for LLM judges on agent trajectories.

The same gold-labeling-by-degradation principle as the core text benchmark:
start from a correct reference trajectory, apply a controlled perturbation
(wrong arguments, swapped tool, hallucinated call, missing step), and the
reference is by construction the better trajectory — no annotation needed.

Quick start (no network required):
    from babeljudge import create_judge
    from babeljudge.agentic import (
        synthetic_tool_tasks, build_agent_dataset, AgentJudge, evaluate_agent
    )

    traces = synthetic_tool_tasks()
    items  = build_agent_dataset(traces)
    judge  = AgentJudge(create_judge("mock"))
    results = evaluate_agent([judge], items)
"""
from .schema import ToolCall, TraceStep, AgentTrace, AgentJudgeItem, AgentJudgment
from .perturbations import (
    AGENT_PERTURBATIONS, make_agent_item, build_agent_dataset,
    argument_corrupt, tool_name_swap, hallucinated_tool,
    missing_required_arg, extra_spurious_arg, drop_intermediate_step,
    corrupt_tool_result, early_termination, step_pad,
)
from .sources import synthetic_tool_tasks, from_bfcl
from .judges import AgentJudge, serialize_trace
from .harness import run_agent_judge, evaluate_agent, agent_cards_to_markdown, save_agent_results
from .metrics import (
    tool_accuracy, argument_accuracy, trajectory_length_bias,
    hallucination_detection, position_bias_delta, order_consistency,
    reliability_score, reliability_card,
)

__all__ = [
    # schema
    "ToolCall", "TraceStep", "AgentTrace", "AgentJudgeItem", "AgentJudgment",
    # perturbations
    "AGENT_PERTURBATIONS", "make_agent_item", "build_agent_dataset",
    "argument_corrupt", "tool_name_swap", "hallucinated_tool",
    "missing_required_arg", "extra_spurious_arg", "drop_intermediate_step",
    "corrupt_tool_result", "early_termination", "step_pad",
    # sources
    "synthetic_tool_tasks", "from_bfcl",
    # judges
    "AgentJudge", "serialize_trace",
    # harness
    "run_agent_judge", "evaluate_agent", "agent_cards_to_markdown", "save_agent_results",
    # metrics
    "tool_accuracy", "argument_accuracy", "trajectory_length_bias",
    "hallucination_detection", "position_bias_delta", "order_consistency",
    "reliability_score", "reliability_card",
]
