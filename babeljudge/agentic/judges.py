"""Judge wrappers for agentic evaluation.

AgentJudge wraps any babeljudge.Judge to compare agent trajectories rather
than plain text responses. It serializes traces to a structured text format
that highlights tool calls, arguments, and results, then delegates the
comparison to the underlying text judge.

Two evaluation modes are supported:
  "tool_calls"   -- present only the tool-call sequence (name + args + result).
                    Fast, token-efficient, focuses purely on tool-use quality.
  "full_trace"   -- present each step including model thoughts and tool calls.
                    More context; better for multi-turn reasoning evaluation.
"""
from __future__ import annotations

import textwrap
from typing import Optional

from .schema import AgentJudgeItem, AgentJudgment, AgentTrace


# ── serialization ─────────────────────────────────────────────────────────────

def _format_tool_calls(trace: AgentTrace) -> str:
    """Render a trace as a compact tool-call sequence."""
    lines = []
    tc_index = 0
    for step in trace.steps:
        if step.tool_call is not None:
            tc = step.tool_call
            args_str = ", ".join(f"{k}={v!r}" for k, v in tc.arguments.items())
            result_str = (f" -> {tc.result[:120]}..." if tc.result and len(tc.result) > 120
                          else f" -> {tc.result}" if tc.result else "")
            lines.append(f"[{tc_index + 1}] {tc.tool_name}({args_str}){result_str}")
            tc_index += 1
        if step.is_final and not step.tool_call:
            break
    return "\n".join(lines) if lines else "(no tool calls)"


def _format_full_trace(trace: AgentTrace) -> str:
    """Render a trace including model thoughts."""
    lines = []
    for i, step in enumerate(trace.steps, 1):
        if step.thought:
            lines.append(f"Thought {i}: {step.thought}")
        if step.tool_call is not None:
            tc = step.tool_call
            args_str = ", ".join(f"{k}={v!r}" for k, v in tc.arguments.items())
            result_str = (f"\n   Result: {tc.result[:200]}..." if tc.result and len(tc.result) > 200
                          else f"\n   Result: {tc.result}" if tc.result else "")
            lines.append(f"Action {i}: {tc.tool_name}({args_str}){result_str}")
        if step.is_final:
            lines.append("(trace complete)")
            break
    return "\n".join(lines) if lines else "(empty trace)"


def serialize_trace(trace: AgentTrace, mode: str = "tool_calls") -> str:
    """Convert a trace to a string for pairwise comparison."""
    header = f"Task: {trace.task_description}\n\n"
    if mode == "tool_calls":
        body = "Tool call sequence:\n" + _format_tool_calls(trace)
    else:
        body = "Agent trace:\n" + _format_full_trace(trace)
    return header + body


# ── prompt templates ──────────────────────────────────────────────────────────

_TOOL_CALLS_INSTRUCTION = textwrap.dedent("""\
    You are evaluating two agent trajectories for the same task.
    Each trajectory shows the sequence of tool calls the agent made to solve the task.
    The BETTER trajectory is the one that:
    - Uses the correct tools with correct arguments
    - Retrieves the right information without hallucinated or spurious calls
    - Completes the task without unnecessary steps
    - Produces a consistent, correct result
    Respond with exactly "A", "B", or "tie".""")

_FULL_TRACE_INSTRUCTION = textwrap.dedent("""\
    You are evaluating two agent trajectories for the same task.
    Each trajectory shows the agent's reasoning steps and tool calls.
    The BETTER trajectory is the one that:
    - Reasons correctly about what tools to use and why
    - Makes correct, complete tool calls in the right order
    - Arrives at a correct and complete final answer
    - Does not include hallucinated tools, redundant steps, or wrong arguments
    Respond with exactly "A", "B", or "tie".""")

_PAIRWISE_TEMPLATE = textwrap.dedent("""\
    {instruction}

    === Trajectory A ===
    {a}

    === Trajectory B ===
    {b}

    Which trajectory is better? Respond with exactly "A", "B", or "tie".""")


def _parse_ab(text: str) -> str:
    text = text.strip().upper()
    for tok in text.split():
        if tok in ("A", "B"):
            return tok
    if "TIE" in text or "EQUAL" in text or "BOTH" in text:
        return "tie"
    return "tie"


# ── AgentJudge ────────────────────────────────────────────────────────────────

class AgentJudge:
    """Wraps a text Judge to evaluate agent trajectories.

    Usage:
        from babeljudge import create_judge
        from babeljudge.agentic import AgentJudge

        text_judge = create_judge("mock")
        judge = AgentJudge(text_judge, focus="tool_calls")
        judgment = judge.compare_item(item)
    """

    def __init__(
        self,
        judge,                          # babeljudge.Judge
        focus: str = "tool_calls",      # "tool_calls" or "full_trace"
        name: Optional[str] = None,
    ):
        self.judge = judge
        self.focus = focus
        self.name = name or f"{judge.name}[agentic-{focus}]"
        if focus not in ("tool_calls", "full_trace"):
            raise ValueError(f"focus must be 'tool_calls' or 'full_trace', got {focus!r}")

    def compare(self, ref_trace: AgentTrace, pert_trace: AgentTrace,
                task_description: Optional[str] = None) -> str:
        """Compare two traces; return 'A', 'B', or 'tie' where A = first arg."""
        a_text = serialize_trace(ref_trace, self.focus)
        b_text = serialize_trace(pert_trace, self.focus)
        instruction = (_TOOL_CALLS_INSTRUCTION if self.focus == "tool_calls"
                       else _FULL_TRACE_INSTRUCTION)
        prompt = task_description or ref_trace.task_description
        full_prompt = _PAIRWISE_TEMPLATE.format(
            instruction=instruction, a=a_text, b=b_text)
        # the underlying text judge sees the formatted prompt as the user query
        raw = self.judge.compare(full_prompt, "", "", instruction="")
        # since we embedded everything in the prompt, A/B already maps correctly
        return _parse_ab(raw)
