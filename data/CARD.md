# Dataset Card: BabelJudge-Bench

## Summary

BabelJudge-Bench is a multilingual benchmark for measuring the **reliability of LLM-as-a-judge** systems. Each item is a pairwise judging task with a **known gold preference**, produced by applying a *controlled degradation* to a high-quality reference response. Because the degradation is constructed, the reference is better by construction — yielding gold labels without human annotation.

## Why it exists

LLM judges are now embedded in most evaluation pipelines, but their reliability degrades in non-English and low-resource settings, and existing studies are fragmented one-off analyses rather than a reusable, comparable standard. BabelJudge provides a standardized harness and living leaderboard so any judge can be scored on the same protocol across languages.

## Construction

- **Sources:** permissively-licensed multilingual references. Each contributes a (prompt, reference) pair.
  - Aya (CohereForAI/aya_dataset, Apache-2.0) — instruction/response pairs, 65+ languages. **Default.**
  - FLORES-200 (facebook/flores, CC BY-SA 4.0) — parallel sentences, 200 languages, used as translation QA.
  - XL-Sum (csebuetnlp/xlsum, CC BY-NC-SA 4.0) — summarization, 44 languages. Requires `datasets<4.0`.
- **Perturbations:** `truncate`, `shuffle`, `number_corrupt`, `drop_entities` (information/coherence/factuality degradations where the reference should win), plus `repeat_pad` (longer-but-worse — the verbosity-bias probe).
- **Severities:** two levels (0.3, 0.6) per source to vary difficulty.
- **Presentation:** every item is judged under both orders (reference-first and perturbed-first) to expose position bias and order-(in)consistency.

## Fields

`item_id, language, prompt, reference, perturbed, perturbation, severity, longer_is_perturbed, meta`

See `babeljudge/schema.py`.

## Metrics

| Metric | Description |
|---|---|
| `accuracy` | Order-balanced reference-win-rate |
| `position_bias_delta` | Win-rate Δ between slot-A and slot-B presentation |
| `verbosity_susceptibility` | Rate of preferring longer-but-redundant (`repeat_pad`) responses |
| `order_consistency` | Fraction of items decided the same way under both orders |
| `reliability_score` | Competence gated by measured bias — the headline number |

## Languages

Default run (Aya): `en`, `sw`, `hi`, `ar`  
Extended set (Aya covers 65+): `es fr zh yo bn id tr am ha so ...`  
FLORES-200: 200 language codes (e.g. `eng_Latn`, `swh_Latn`, `hin_Deva`)

## First benchmark results (2026-06-14)

Model: `mlx-community/Qwen2.5-7B-Instruct-4bit` · Data: Aya · Languages: en, hi, ar, sw · N=10/lang · 800 judge calls

| Lang | Accuracy | Pos-BiasΔ | VerbositySusc | OrderConsist | Reliability |
|---|---|---|---|---|---|
| hi | 0.835 | +0.090 | 0.250 | 0.670 | 0.714 |
| en | 0.815 | +0.070 | 0.075 | 0.630 | 0.700 |
| ar | 0.770 | -0.040 | 0.175 | 0.640 | 0.695 |
| sw | 0.660 | -0.080 | 0.175 | 0.480 | 0.550 |

## Intended use & limitations

For diagnosing judge reliability, not for training reward models. Controlled degradations approximate but do not fully capture naturally-occurring quality differences; pair with a small human-validated slice to report concordance.

## Licensing

Code: Apache-2.0.  
Data slices inherit the license of their upstream source (see table above). Aya is Apache-2.0; FLORES-200 is CC BY-SA 4.0; XL-Sum is CC BY-NC-SA 4.0.
