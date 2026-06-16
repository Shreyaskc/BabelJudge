"""Data structures for agentic evaluation.

An *agentic item* represents a task that requires tool use. The "reference
trajectory" is a correct sequence of tool calls that solves the task. The
"perturbed trajectory" is a controlled degradation: wrong arguments, swapped
tools, missing steps, hallucinated calls. The reference trajectory is by
construction the better one -- no human annotation required.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class ToolCall:
    """A single tool invocation: name + arguments dict + the tool's response."""
    tool_name: str
    arguments: dict[str, Any]
    result: Optional[str] = None   # tool response; None if the call was never executed
    step_index: int = 0


@dataclass
class TraceStep:
    """One step in an agent trace: a model thought plus an optional tool call."""
    thought: Optional[str]           # model reasoning / CoT, if any
    tool_call: Optional[ToolCall]    # None for the final answer step
    is_final: bool = False           # True on the last step (answer emitted)


@dataclass
class AgentTrace:
    """A full agent trajectory: a sequence of steps that solves a task."""
    task_id: str
    task_description: str            # the user-facing task
    steps: list[TraceStep]
    final_answer: Optional[str] = None
    meta: dict = field(default_factory=dict)

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [s.tool_call for s in self.steps if s.tool_call is not None]


@dataclass
class AgentJudgeItem:
    """A pairwise judging item for agentic evaluation.

    reference_trace is the correct trajectory; perturbed_trace is a controlled
    degradation. The gold answer is always the reference trajectory.
    """
    item_id: str
    task_description: str
    reference_trace: AgentTrace
    perturbed_trace: AgentTrace
    perturbation: str               # which perturbation produced perturbed_trace
    severity: float
    longer_is_perturbed: bool = False   # True for step-padding perturbations
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentJudgment:
    """One judge decision on one presentation of an AgentJudgeItem."""
    item_id: str
    judge: str
    order: str          # "ref_first" or "pert_first"
    choice: str         # "reference", "perturbed", "tie", or "error"
    raw: Optional[str] = None
    focus: str = "tool_calls"   # "tool_calls" or "full_trace"
