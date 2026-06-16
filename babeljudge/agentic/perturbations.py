"""Controlled-perturbation generator for agent trajectories.

Mirrors babeljudge.perturbations but operates on AgentTrace objects rather
than plain text. Each perturbation degrades a reference trajectory in a
targeted way so the gold label (reference > perturbed) is known for free.

Perturbation catalogue
----------------------
argument_corrupt      Corrupt one argument value in the most important tool call.
                      Probes whether the judge catches factual tool-call errors.

tool_name_swap        Replace a tool name with a plausible-but-wrong alternative.
                      Probes whether the judge notices the wrong tool was chosen.

hallucinated_tool      Insert a made-up tool call between real steps.
                      Probes hallucination detection.

missing_required_arg  Remove a required argument from a tool call.
                      Probes whether the judge catches incomplete calls.

extra_spurious_arg    Add a nonsense key/value to a tool call's arguments.
                      A subtler perturbation -- the call may still "work".

drop_intermediate_step Remove one non-final step from the trajectory.
                      Probes whether the judge notices a reasoning gap.

corrupt_tool_result   Mutate the result string returned by a tool.
                      Probes whether the judge catches downstream inconsistency.

early_termination     Drop all steps after the midpoint, cutting the answer short.
                      Probes whether the judge notices an incomplete trajectory.

step_pad              Append redundant repeated steps after the answer.
                      The VERBOSITY-BIAS probe: the perturbed trace is LONGER
                      but strictly worse (redundant / circular tool calls).
"""
from __future__ import annotations

import copy
import random
import re
from typing import Callable, Optional

from .schema import AgentTrace, AgentJudgeItem, ToolCall, TraceStep


# ── helpers ──────────────────────────────────────────────────────────────────

def _deep(trace: AgentTrace) -> AgentTrace:
    return copy.deepcopy(trace)


def _tool_steps(trace: AgentTrace) -> list[int]:
    """Indices of steps that contain a tool call."""
    return [i for i, s in enumerate(trace.steps) if s.tool_call is not None]


def _corrupt_str(value: str, rng: random.Random) -> str:
    """Slightly corrupt a string value."""
    if not value:
        return value
    ops = [
        lambda v: v[::-1],                                     # reverse
        lambda v: v.upper() if v == v.lower() else v.lower(),  # case flip
        lambda v: re.sub(r"\d+", lambda m: str(int(m.group()) + rng.randint(1, 9)), v)
        if re.search(r"\d", v) else v + "_INVALID",            # numeric bump
        lambda v: v.replace(v.split()[0], "WRONG", 1) if " " in v else "WRONG",
    ]
    return rng.choice(ops)(value)


# ── perturbations ─────────────────────────────────────────────────────────────

def argument_corrupt(trace: AgentTrace, severity: float, rng: random.Random) -> AgentTrace:
    """Corrupt one argument value in a tool call."""
    t = _deep(trace)
    idxs = _tool_steps(t)
    if not idxs:
        return t
    step = t.steps[rng.choice(idxs)]
    tc = step.tool_call
    if not tc.arguments:
        return t
    key = rng.choice(list(tc.arguments.keys()))
    old = tc.arguments[key]
    if isinstance(old, str):
        tc.arguments[key] = _corrupt_str(old, rng)
    elif isinstance(old, (int, float)):
        tc.arguments[key] = old + rng.choice([-100, -10, 7, 42, 99])
    elif isinstance(old, bool):
        tc.arguments[key] = not old
    elif isinstance(old, list) and old:
        tc.arguments[key] = old[::-1]
    else:
        tc.arguments[key] = "CORRUPTED"
    return t


def tool_name_swap(trace: AgentTrace, severity: float, rng: random.Random) -> AgentTrace:
    """Replace a tool name with a plausible-but-wrong alternative."""
    t = _deep(trace)
    idxs = _tool_steps(t)
    if not idxs:
        return t
    step = t.steps[rng.choice(idxs)]
    tc = step.tool_call
    all_names = [t_.steps[i].tool_call.tool_name for i in _tool_steps(t)
                 for t_ in [t] if t_.steps[i].tool_call is not None]
    alternatives = [n for n in all_names if n != tc.tool_name]
    if alternatives:
        tc.tool_name = rng.choice(alternatives)
    else:
        # no other real tool name available, invent a plausible one
        tc.tool_name = tc.tool_name + "_v2"
    return t


def hallucinated_tool(trace: AgentTrace, severity: float, rng: random.Random) -> AgentTrace:
    """Insert a made-up tool call between real steps."""
    t = _deep(trace)
    fake_names = ["verify_credentials", "cache_lookup", "validate_schema",
                  "log_event", "preprocess_input", "fetch_metadata"]
    fake_tc = ToolCall(
        tool_name=rng.choice(fake_names),
        arguments={"input": "HALLUCINATED", "mode": "fast"},
        result="OK",
        step_index=-1,
    )
    fake_step = TraceStep(thought="Let me verify this first.", tool_call=fake_tc)
    # insert after a random non-final step
    non_final = [i for i, s in enumerate(t.steps) if not s.is_final]
    if non_final:
        pos = rng.choice(non_final) + 1
    else:
        pos = 0
    t.steps.insert(pos, fake_step)
    return t


def missing_required_arg(trace: AgentTrace, severity: float, rng: random.Random) -> AgentTrace:
    """Remove a required argument from a tool call."""
    t = _deep(trace)
    idxs = _tool_steps(t)
    if not idxs:
        return t
    step = t.steps[rng.choice(idxs)]
    tc = step.tool_call
    if len(tc.arguments) < 2:
        return argument_corrupt(t, severity, rng)   # fall back
    key = rng.choice(list(tc.arguments.keys()))
    del tc.arguments[key]
    return t


def extra_spurious_arg(trace: AgentTrace, severity: float, rng: random.Random) -> AgentTrace:
    """Add a nonsense key/value pair to a tool call's arguments."""
    t = _deep(trace)
    idxs = _tool_steps(t)
    if not idxs:
        return t
    step = t.steps[rng.choice(idxs)]
    spurious_keys = ["debug_mode", "legacy_compat", "trace_id", "verbose",
                     "retry_count", "format_override"]
    step.tool_call.arguments[rng.choice(spurious_keys)] = rng.choice([True, 99, "auto", "v1"])
    return t


def drop_intermediate_step(trace: AgentTrace, severity: float, rng: random.Random) -> AgentTrace:
    """Remove one non-final intermediate step from the trajectory."""
    t = _deep(trace)
    non_final = [i for i, s in enumerate(t.steps) if not s.is_final]
    if len(non_final) < 2:
        return t
    drop_idx = rng.choice(non_final[:-1])   # keep the last non-final step
    t.steps.pop(drop_idx)
    return t


def corrupt_tool_result(trace: AgentTrace, severity: float, rng: random.Random) -> AgentTrace:
    """Mutate the result string returned by a tool."""
    t = _deep(trace)
    idxs = [i for i in _tool_steps(t) if t.steps[i].tool_call.result]
    if not idxs:
        return t
    tc = t.steps[rng.choice(idxs)].tool_call
    tc.result = _corrupt_str(tc.result, rng)
    return t


def early_termination(trace: AgentTrace, severity: float, rng: random.Random) -> AgentTrace:
    """Drop all steps after the midpoint, simulating an incomplete trajectory."""
    t = _deep(trace)
    cut = max(1, int(len(t.steps) * (1 - severity)))
    t.steps = t.steps[:cut]
    if t.steps:
        t.steps[-1].is_final = True
    t.final_answer = None
    return t


def step_pad(trace: AgentTrace, severity: float, rng: random.Random) -> AgentTrace:
    """Append redundant repeated steps (VERBOSITY-BIAS probe).

    The returned trace is LONGER but strictly worse: it loops back and repeats
    tool calls that were already resolved, adding no value.
    """
    t = _deep(trace)
    idxs = _tool_steps(t)
    if not idxs:
        return t
    n_extra = max(1, int(len(idxs) * (0.5 + severity)))
    # mark current final step as non-final before padding
    for s in t.steps:
        s.is_final = False
    for _ in range(n_extra):
        repeat_step = copy.deepcopy(t.steps[rng.choice(idxs)])
        repeat_step.thought = "Let me double-check this."
        repeat_step.is_final = False
        t.steps.append(repeat_step)
    # re-add a terminal step
    t.steps.append(TraceStep(thought="Confirmed.", tool_call=None, is_final=True))
    return t


# ── registry ─────────────────────────────────────────────────────────────────

AGENT_PERTURBATIONS: dict[str, Callable[[AgentTrace, float, random.Random], AgentTrace]] = {
    "argument_corrupt":      argument_corrupt,
    "tool_name_swap":        tool_name_swap,
    "hallucinated_tool":     hallucinated_tool,
    "missing_required_arg":  missing_required_arg,
    "extra_spurious_arg":    extra_spurious_arg,
    "drop_intermediate_step": drop_intermediate_step,
    "corrupt_tool_result":   corrupt_tool_result,
    "early_termination":     early_termination,
    "step_pad":              step_pad,
}

# Perturbations where the perturbed trace is the LONGER one.
_LONGER_PERTURBATIONS = {"step_pad", "hallucinated_tool", "extra_spurious_arg"}


# ── item builder ─────────────────────────────────────────────────────────────

def make_agent_item(
    item_id: str,
    reference_trace: AgentTrace,
    perturbation: str,
    severity: float = 0.5,
    seed: Optional[int] = None,
) -> AgentJudgeItem:
    """Build a single gold-labeled AgentJudgeItem from a reference trace."""
    if perturbation not in AGENT_PERTURBATIONS:
        raise ValueError(f"unknown perturbation {perturbation!r}; "
                         f"choose from {sorted(AGENT_PERTURBATIONS)}")
    rng = random.Random(seed if seed is not None
                        else hash((item_id, perturbation)) & 0xFFFFFFFF)
    perturbed = AGENT_PERTURBATIONS[perturbation](reference_trace, severity, rng)
    return AgentJudgeItem(
        item_id=item_id,
        task_description=reference_trace.task_description,
        reference_trace=reference_trace,
        perturbed_trace=perturbed,
        perturbation=perturbation,
        severity=severity,
        longer_is_perturbed=perturbation in _LONGER_PERTURBATIONS,
    )


def build_agent_dataset(
    traces: list[AgentTrace],
    perturbations: Optional[list[str]] = None,
    severities: tuple[float, ...] = (0.3, 0.6),
    seed: int = 0,
) -> list[AgentJudgeItem]:
    """Expand a list of reference AgentTraces into gold-labeled AgentJudgeItems.

    Each trace is crossed with each perturbation × severity, producing a
    dataset where every item has a known-correct gold answer.
    """
    perturbations = perturbations or list(AGENT_PERTURBATIONS)
    items: list[AgentJudgeItem] = []
    for trace in traces:
        for pert in perturbations:
            for sev in severities:
                items.append(make_agent_item(
                    item_id=f"{trace.task_id}::{pert}::{sev}",
                    reference_trace=trace,
                    perturbation=pert,
                    severity=sev,
                    seed=seed,
                ))
    return items
