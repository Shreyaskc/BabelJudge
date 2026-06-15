"""Controlled-perturbation generator.

This is the cost-saving core of MultiJudge. Instead of paying annotators to
label which of two responses is better, we *start* from a high-quality
reference response and apply a controlled degradation. The reference is then,
by construction, the better response -- giving us a gold label for free.

Each perturbation is designed to probe a specific failure mode:

  truncate       -> information loss            (judge should still prefer ref)
  shuffle        -> coherence loss              (judge should still prefer ref)
  number_corrupt -> factual error               (judge should still prefer ref)
  drop_entities  -> information loss            (judge should still prefer ref)
  repeat_pad     -> *length* added, no value    (VERBOSITY-BIAS probe: the
                    perturbed text is LONGER but worse; a judge that prefers it
                    is exhibiting verbosity bias)

Most perturbations are deliberately language-agnostic (operate on whitespace
tokens, sentence delimiters, digits) so the same generator works across the
~12-15 languages in the benchmark. Language-specific hooks (e.g. negation
injection) are pluggable via `LANG_CONFIG` without touching the core.
"""
from __future__ import annotations

import random
import re
from typing import Callable, Optional

from .schema import JudgeItem

# Sentence splitting that tolerates Latin, CJK, and Arabic/Devanagari punctuation.
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?\u3002\uFF01\uFF1F\u06D4\u0964])\s+")
_DIGITS = re.compile(r"\d+")
# A loose "entity-ish" heuristic: capitalized tokens (Latin scripts). For
# non-cased scripts this simply yields fewer hits and we fall back to token drop.
_CAPWORD = re.compile(r"\b[A-Z][\w-]+\b")


def _sentences(text: str) -> list[str]:
    parts = [s for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    return parts if parts else [text.strip()]


def truncate(text: str, severity: float, rng: random.Random) -> str:
    """Drop the trailing `severity` fraction of sentences (information loss)."""
    sents = _sentences(text)
    keep = max(1, int(round(len(sents) * (1.0 - severity))))
    return " ".join(sents[:keep])


def shuffle(text: str, severity: float, rng: random.Random) -> str:
    """Reorder sentences to damage coherence. Severity sets shuffle probability."""
    sents = _sentences(text)
    if len(sents) < 2:
        return text
    idx = list(range(len(sents)))
    for i in range(len(idx) - 1, 0, -1):
        if rng.random() < severity:
            j = rng.randint(0, i)
            idx[i], idx[j] = idx[j], idx[i]
    return " ".join(sents[i] for i in idx)


def number_corrupt(text: str, severity: float, rng: random.Random) -> str:
    """Corrupt numeric tokens to introduce factual errors."""
    def repl(m: re.Match) -> str:
        if rng.random() >= severity:
            return m.group(0)
        n = m.group(0)
        # perturb by a random nonzero delta, preserving digit length where possible
        delta = rng.choice([-3, -2, -1, 1, 2, 3, 7])
        try:
            return str(max(0, int(n) + delta))
        except ValueError:
            return n
    corrupted = _DIGITS.sub(repl, text)
    # if the text had no digits, fall back to entity drop so the item is still valid
    return corrupted if corrupted != text else drop_entities(text, severity, rng)


def drop_entities(text: str, severity: float, rng: random.Random) -> str:
    """Remove capitalized/entity-like tokens, or random tokens for non-cased scripts."""
    hits = list(_CAPWORD.finditer(text))
    if hits:
        out, last = [], 0
        for m in hits:
            if rng.random() < severity:
                out.append(text[last:m.start()])
                last = m.end()
        out.append(text[last:])
        result = re.sub(r"\s{2,}", " ", "".join(out)).strip()
        return result if result else text
    # non-cased fallback: drop random whitespace tokens
    toks = text.split()
    kept = [t for t in toks if rng.random() >= severity * 0.5]
    return " ".join(kept) if kept else text


def repeat_pad(text: str, severity: float, rng: random.Random) -> str:
    """Pad with repeated/duplicated content: LONGER but no added value.

    This is the verbosity-bias probe. The returned text is intentionally longer
    than the reference while being strictly worse (redundant).
    """
    sents = _sentences(text)
    extra = max(1, int(round(len(sents) * (0.5 + severity))))
    filler = [rng.choice(sents) for _ in range(extra)]
    return " ".join(sents + filler)


PERTURBATIONS: dict[str, Callable[[str, float, random.Random], str]] = {
    "truncate": truncate,
    "shuffle": shuffle,
    "number_corrupt": number_corrupt,
    "drop_entities": drop_entities,
    "repeat_pad": repeat_pad,
}

# Perturbations where the resulting `perturbed` text is the LONGER one.
_LONGER_PERTURBATIONS = {"repeat_pad"}

# Hook for language-specific perturbations; left minimal by design.
LANG_CONFIG: dict[str, dict] = {}


def make_item(
    item_id: str,
    language: str,
    prompt: str,
    reference: str,
    perturbation: str,
    severity: float = 0.5,
    seed: Optional[int] = None,
) -> JudgeItem:
    """Build a single gold-labeled JudgeItem from a reference response."""
    if perturbation not in PERTURBATIONS:
        raise ValueError(f"unknown perturbation {perturbation!r}; "
                         f"choose from {sorted(PERTURBATIONS)}")
    rng = random.Random(seed if seed is not None else hash((item_id, perturbation)) & 0xFFFFFFFF)
    perturbed = PERTURBATIONS[perturbation](reference, severity, rng)
    # guarantee the perturbation actually changed the text
    if perturbed.strip() == reference.strip():
        perturbed = truncate(reference, max(0.5, severity), rng)
    return JudgeItem(
        item_id=item_id,
        language=language,
        prompt=prompt,
        reference=reference,
        perturbed=perturbed,
        perturbation=perturbation,
        severity=severity,
        longer_is_perturbed=perturbation in _LONGER_PERTURBATIONS,
    )


def build_dataset(
    sources: list[dict],
    perturbations: Optional[list[str]] = None,
    severities: tuple[float, ...] = (0.3, 0.6),
    seed: int = 0,
) -> list[JudgeItem]:
    """Expand a list of {id, language, prompt, reference} sources into items.

    Each source is crossed with each requested perturbation and severity, so a
    few hundred reference responses become a few thousand gold-labeled items.
    """
    perturbations = perturbations or list(PERTURBATIONS)
    items: list[JudgeItem] = []
    for src in sources:
        for pert in perturbations:
            for sev in severities:
                items.append(
                    make_item(
                        item_id=f"{src['id']}::{pert}::{sev}",
                        language=src["language"],
                        prompt=src["prompt"],
                        reference=src["reference"],
                        perturbation=pert,
                        severity=sev,
                        seed=seed,
                    )
                )
    return items
