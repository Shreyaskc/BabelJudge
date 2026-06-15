"""Adapters from public multilingual datasets to BabelJudge `sources`.

A `source` is a dict: {id, language, prompt, reference}. The reference is a
high-quality human-written response that the perturbation generator then
degrades to create gold-labeled judging items.

Three loaders are provided:
  - XL-Sum (csebuetnlp/xlsum): CC BY-NC-SA 4.0, 45 languages, summarization.
  - Aya (CohereForAI/aya_dataset): Apache-2.0, 65+ languages, instruction/response pairs.
  - FLORES-200 (facebook/flores): CC BY-SA 4.0, 200 languages, translation-based QA.

The network-dependent loads are isolated; `records_to_sources` is pure and unit-testable.
"""
from __future__ import annotations

from typing import Iterable, Optional

# XL-Sum config names are lowercase language names.
XLSUM_DEFAULT_LANGUAGES = [
    "english", "spanish", "french", "chinese_simplified", "arabic",
    "hindi", "swahili", "yoruba", "bengali", "indonesian", "turkish",
]

DEFAULT_LANGUAGES = XLSUM_DEFAULT_LANGUAGES  # backwards compat

# Aya uses full English language names in its `language` column.
# Map from short BCP-47 tags (used in BabelJudge items) to Aya's language strings.
_AYA_LANG_MAP = {
    "en": "English", "es": "Spanish", "fr": "French",
    "zh": "Simplified Chinese", "ar": "Standard Arabic",
    "hi": "Hindi", "sw": "Swahili", "yo": "Yoruba",
    "bn": "Bengali", "id": "Indonesian", "tr": "Turkish",
}
AYA_DEFAULT_LANGUAGES = list(_AYA_LANG_MAP.keys())

# FLORES-200 language codes (flores_path column name suffix).
FLORES_DEFAULT_LANGUAGES = ["eng_Latn", "spa_Latn", "fra_Latn", "zho_Hans",
                             "ara_Arab", "hin_Deva", "swh_Latn", "yor_Latn",
                             "ben_Beng", "ind_Latn", "tur_Latn"]

# Map FLORES lang codes to short BCP-47 tags for the `language` field.
_FLORES_TO_SHORT = {
    "eng_Latn": "en", "spa_Latn": "es", "fra_Latn": "fr", "zho_Hans": "zh",
    "ara_Arab": "ar", "hin_Deva": "hi", "swh_Latn": "sw", "yor_Latn": "yo",
    "ben_Beng": "bn", "ind_Latn": "id", "tur_Latn": "tr",
}

SUMMARIZE_INSTRUCTION = {
    # localized instruction stems for the (optional) prompt-language probe
    # XL-Sum keys are full English names; Aya/FLORES keys are BCP-47 short tags.
    "english": "Summarize the following article in a few sentences.",
    "spanish": "Resume el siguiente articulo en pocas frases.",
    "french": "Resumez l'article suivant en quelques phrases.",
    "swahili": "Fupisha makala ifuatayo kwa sentensi chache.",
    "hindi": "Is lekh ka saaransh kuch vaakyon mein dijiye.",
    # BCP-47 short tags (used by Aya and FLORES loaders)
    "en": "Summarize the following article in a few sentences.",
    "es": "Resume el siguiente articulo en pocas frases.",
    "fr": "Resumez l'article suivant en quelques phrases.",
    "zh": "请用几句话概括以下文章。",
    "ar": "لخّص المقال التالي في جمل قليلة.",
    "hi": "Is lekh ka saaransh kuch vaakyon mein dijiye.",
    "sw": "Fupisha makala ifuatayo kwa sentensi chache.",
    "yo": "Ṣe akopọ nkan yii ni awọn gbolohun diẹ.",
    "bn": "নিচের নিবন্ধটি কয়েকটি বাক্যে সংক্ষেপ করুন।",
    "id": "Ringkaslah artikel berikut dalam beberapa kalimat.",
    "tr": "Aşağıdaki makaleyi birkaç cümleyle özetleyin.",
}


def records_to_sources(records: Iterable[dict], language: str,
                       max_article_chars: int = 1500,
                       article_key: str = "text",
                       summary_key: str = "summary",
                       min_ref_sentences: int = 1,
                       id_prefix: Optional[str] = None) -> list[dict]:
    """Pure transform: XL-Sum-style records -> MultiJudge sources. Testable offline.

    `min_ref_sentences` drops references too short for perturbations to be
    meaningful (e.g. a one-sentence summary, where `truncate`/`drop_entities`
    would barely differ from the original).
    """
    from .perturbations import _sentences  # reuse the shared splitter
    instr = SUMMARIZE_INSTRUCTION.get(language, SUMMARIZE_INSTRUCTION["english"])
    prefix = id_prefix or language
    out = []
    for i, rec in enumerate(records):
        article = (rec.get(article_key) or "").strip()
        summary = (rec.get(summary_key) or "").strip()
        if not article or not summary:
            continue  # skip incomplete rows
        if len(_sentences(summary)) < min_ref_sentences:
            continue  # reference too short to degrade meaningfully
        if len(article) > max_article_chars:
            article = article[:max_article_chars].rsplit(" ", 1)[0] + " ..."
        out.append({
            "id": f"{prefix}-{i}",
            "language": language,
            "prompt": f"{instr}\n\n{article}",
            "reference": summary,
        })
    return out


def _load_xlsum(language: str, n: int, split: str):
    from datasets import load_dataset
    # csebuetnlp/xlsum uses a loading script incompatible with datasets>=4.0.
    # trust_remote_code=True is required and only works on datasets<4.0.
    # For datasets>=4.0 use from_aya() or from_flores() instead.
    return load_dataset("csebuetnlp/xlsum", language, split=f"{split}[:{n}]",
                        trust_remote_code=True)


def from_xlsum(languages: Optional[list[str]] = None, n_per_lang: int = 25,
               split: str = "test", max_article_chars: int = 1500,
               min_ref_sentences: int = 3) -> list[dict]:
    """Load XL-Sum slices for the given languages and return BabelJudge sources.

    Note: pulls more rows than requested per language to offset rows dropped by
    the `min_ref_sentences` filter.
    """
    languages = languages or XLSUM_DEFAULT_LANGUAGES
    sources: list[dict] = []
    for lang in languages:
        try:
            ds = _load_xlsum(lang, n_per_lang * 3, split)  # overfetch for filtering
        except Exception as e:
            print(f"[warn] skipping {lang}: {e}")
            continue
        s = records_to_sources(ds, lang, max_article_chars,
                               min_ref_sentences=min_ref_sentences)
        sources.extend(s[:n_per_lang])
        print(f"[{lang}] kept {min(len(s), n_per_lang)} sources")
    return sources


# ---------------------------------------------------------------------------
# Aya loader (CohereForAI/aya_dataset, Apache-2.0)
# ---------------------------------------------------------------------------

def from_aya(languages: Optional[list[str]] = None, n_per_lang: int = 25,
             split: str = "train", min_ref_sentences: int = 2) -> list[dict]:
    """Load Aya instruction/response pairs and return BabelJudge sources.

    Aya is Apache-2.0 licensed and covers 65+ languages. Each row has
    `inputs` (instruction) and `targets` (response). We use the response as
    the reference and the instruction as the prompt directly.

    `languages` are BCP-47 short tags (e.g. "en", "hi"). The loader maps them
    to Aya's full English language names internally. Defaults to `train` split
    since the test split only covers 7 languages.
    """
    from datasets import load_dataset
    from .perturbations import _sentences

    languages = languages or AYA_DEFAULT_LANGUAGES
    sources: list[dict] = []

    try:
        ds_full = load_dataset("CohereForAI/aya_dataset", split=split)
    except Exception as e:
        print(f"[warn] could not load Aya dataset: {e}")
        return sources

    for lang in languages:
        aya_name = _AYA_LANG_MAP.get(lang, lang)
        subset = [r for r in ds_full if r.get("language") == aya_name]
        if not subset:
            print(f"[warn] aya: no rows for language '{aya_name}' (tag: {lang})")
            continue
        kept = []
        for i, rec in enumerate(subset):
            prompt = (rec.get("inputs") or "").strip()
            reference = (rec.get("targets") or "").strip()
            if not prompt or not reference:
                continue
            if len(_sentences(reference)) < min_ref_sentences:
                continue
            kept.append({
                "id": f"aya-{lang}-{i}",
                "language": lang,
                "prompt": prompt,
                "reference": reference,
            })
            if len(kept) >= n_per_lang:
                break
        sources.extend(kept)
        print(f"[aya/{lang}] kept {len(kept)} sources")

    return sources


# ---------------------------------------------------------------------------
# FLORES-200 loader (facebook/flores, CC BY-SA 4.0)
# ---------------------------------------------------------------------------

def from_flores(languages: Optional[list[str]] = None, n_per_lang: int = 25,
                split: str = "devtest", min_ref_sentences: int = 1,
                pivot_lang: str = "eng_Latn") -> list[dict]:
    """Load FLORES-200 and return BabelJudge sources as translation QA pairs.

    FLORES-200 has parallel sentences in 200 languages. We construct a
    translation task: prompt = English sentence with a "Translate to <lang>"
    instruction; reference = the human translation in the target language.

    `languages` are FLORES-200 language codes (e.g. "spa_Latn"). The pivot
    (source) language is `pivot_lang` (default: English).
    """
    from datasets import load_dataset
    from .perturbations import _sentences

    languages = languages or FLORES_DEFAULT_LANGUAGES
    # Remove pivot from targets to avoid trivial identity items.
    target_langs = [l for l in languages if l != pivot_lang]

    sources: list[dict] = []

    try:
        ds = load_dataset("facebook/flores", "all", split=split)
    except Exception as e:
        print(f"[warn] could not load FLORES-200: {e}")
        return sources

    pivot_col = f"sentence_{pivot_lang}"
    for lang in target_langs:
        short = _FLORES_TO_SHORT.get(lang, lang)
        target_col = f"sentence_{lang}"
        if target_col not in ds.column_names:
            print(f"[warn] FLORES column '{target_col}' not found, skipping {lang}")
            continue

        translate_prompt = f"Translate the following sentence into {lang}.\n\n"
        kept = []
        for i, row in enumerate(ds):
            src = (row.get(pivot_col) or "").strip()
            tgt = (row.get(target_col) or "").strip()
            if not src or not tgt:
                continue
            if len(_sentences(tgt)) < min_ref_sentences:
                continue
            kept.append({
                "id": f"flores-{lang}-{i}",
                "language": short,
                "prompt": translate_prompt + src,
                "reference": tgt,
            })
            if len(kept) >= n_per_lang:
                break
        sources.extend(kept)
        print(f"[flores/{lang}] kept {len(kept)} sources")

    return sources
