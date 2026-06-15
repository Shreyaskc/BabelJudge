"""The MultiJudge harness.

Runs a set of judges over a set of gold-labeled items, presenting every item
under BOTH presentation orders (reference-first and perturbed-first) so that
position bias and order-consistency are measurable. Produces per-(judge,
language) reliability cards.
"""
from __future__ import annotations

import json
from collections import defaultdict

from .judges import Judge, DEFAULT_INSTRUCTION
from .metrics import reliability_card
from .schema import JudgeItem, Judgment

ORDERS = ("ref_first", "pert_first")


def _present(item: JudgeItem, order: str) -> tuple[str, str]:
    """Return (A, B) text for the given presentation order."""
    if order == "ref_first":
        return item.reference, item.perturbed
    return item.perturbed, item.reference


def _normalize(order: str, raw_choice: str) -> str:
    """Map the judge's A/B/tie answer back to reference/perturbed."""
    if raw_choice == "tie":
        return "tie"
    if order == "ref_first":
        return "reference" if raw_choice == "A" else "perturbed"
    else:
        return "perturbed" if raw_choice == "A" else "reference"


def run_judge(judge: Judge, items: list[JudgeItem],
              instruction: str = DEFAULT_INSTRUCTION,
              prompt_language: str = "en") -> list[Judgment]:
    judgments: list[Judgment] = []
    for item in items:
        for order in ORDERS:
            a, b = _present(item, order)
            try:
                raw = judge.compare(item.prompt, a, b, instruction=instruction)
                choice = _normalize(order, raw)
            except Exception as e:  # a judge that errors is recorded, not crashed on
                raw, choice = f"ERROR: {e}", "error"
            judgments.append(Judgment(
                item_id=item.item_id, language=item.language, judge=judge.name,
                order=order, choice=choice, raw=str(raw), prompt_language=prompt_language,
            ))
    return judgments


def evaluate(judges: list[Judge], items: list[JudgeItem]) -> dict:
    """Run all judges over all items; return cards grouped by judge and language."""
    items_by_id = {it.item_id: it for it in items}
    by_lang: dict[str, list[JudgeItem]] = defaultdict(list)
    for it in items:
        by_lang[it.language].append(it)

    results = {"cards": [], "leaderboard": []}
    agg: dict[str, dict] = {}
    for judge in judges:
        accs, scores = [], []
        for lang, lang_items in sorted(by_lang.items()):
            js = run_judge(judge, lang_items)
            card = reliability_card(judge.name, lang, js, items_by_id)
            results["cards"].append(card)
            accs.append(card["accuracy"])
            scores.append(card["reliability_score"])
        agg[judge.name] = {
            "judge": judge.name,
            "reliability": round(sum(scores) / len(scores), 4),
            "macro_accuracy": round(sum(accs) / len(accs), 4),
            "n_languages": len(scores),
        }
    results["leaderboard"] = sorted(
        agg.values(), key=lambda r: r["reliability"], reverse=True
    )
    return results


def cards_to_markdown(results: dict) -> str:
    lines = ["# MultiJudge Reliability Leaderboard\n",
             "Reliability = competence (accuracy + order-consistency) gated by "
             "bias severity (position & verbosity). Higher is better. Raw accuracy "
             "is shown alongside to make the gap visible.\n",
             "| Rank | Judge | Reliability | Macro-Acc | #Langs |",
             "|---|---|---|---|---|"]
    for i, row in enumerate(results["leaderboard"], 1):
        lines.append(f"| {i} | {row['judge']} | {row['reliability']:.3f} | "
                     f"{row['macro_accuracy']:.3f} | {row['n_languages']} |")
    lines.append("\n## Per-language reliability cards\n")
    lines.append("| Judge | Lang | Acc | PosBiasΔ | VerbositySusc | OrderConsist | Reliability | N |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for c in results["cards"]:
        lines.append(
            f"| {c['judge']} | {c['language']} | {c['accuracy']:.3f} | "
            f"{c['position_bias_delta']:+.3f} | {c['verbosity_susceptibility']:.3f} | "
            f"{c['order_consistency']:.3f} | {c['reliability_score']:.3f} | {c['n_items']} |"
        )
    return "\n".join(lines)


def save_results(results: dict, json_path: str, md_path: str) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(cards_to_markdown(results))
