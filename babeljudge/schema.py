"""Core data structures for MultiJudge.

An *item* is a single evaluation unit: a prompt plus two candidate responses,
one of which is the known-better "reference" and one of which is a controlled
degradation. Because we construct the degradation ourselves, we know the gold
preference for free -- no human annotation required.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class JudgeItem:
    """A single pairwise judging item with a known gold preference."""
    item_id: str
    language: str
    prompt: str
    reference: str          # the known-better response
    perturbed: str          # the controlled degradation
    perturbation: str       # which perturbation produced `perturbed`
    severity: float         # 0..1, how aggressive the degradation was
    # Gold answer is always "reference": the reference response should win.
    # The exception is recorded explicitly for probes where length is a confound.
    longer_is_perturbed: bool = False   # True when `perturbed` is the LONGER text
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Judgment:
    """One judge decision on one presentation of an item.

    `order` records how the two responses were shown to the judge:
      - "ref_first": A = reference, B = perturbed
      - "pert_first": A = perturbed, B = reference
    `choice` is normalized back to which underlying response won:
      - "reference" or "perturbed" (or "tie"/"error")
    """
    item_id: str
    language: str
    judge: str
    order: str
    choice: str
    raw: Optional[str] = None     # raw judge output, for auditing
    prompt_language: str = "en"   # language the judge *instructions* were in
