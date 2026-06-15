"""One-command real run for Apple Silicon (Mac Mini / MacBook / Studio).

Usage
-----
    pip install mlx-lm datasets          # MLX path (fastest, native)
    python examples/run_mac.py

If mlx-lm is not installed, falls back to a local Ollama server:
    brew install ollama && ollama pull qwen2.5:7b
    pip install datasets openai
    python examples/run_mac.py

Data source: Aya (CohereForAI/aya_dataset, Apache-2.0) — works with datasets>=4.0.
Edit the CONFIG block to change languages, sample size, or model.
"""
import sys

# ---------------- CONFIG ----------------
# BCP-47 language tags used by the Aya dataset
LANGUAGES = ["en", "sw", "hi", "ar"]   # mix of resource levels
N_PER_LANG = 10
SEVERITIES = (0.3, 0.6)
MLX_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434/v1"
# ----------------------------------------

from babeljudge import (from_aya, build_dataset, evaluate,
                        cards_to_markdown, save_results)


def pick_judge():
    """Prefer native MLX; fall back to a local Ollama server."""
    try:
        from babeljudge import MLXJudge
        print(f"Backend: MLX ({MLX_MODEL})")
        return MLXJudge(MLX_MODEL)
    except ImportError:
        from babeljudge import OpenAICompatLocalJudge
        print(f"Backend: Ollama ({OLLAMA_MODEL} @ {OLLAMA_URL})")
        print("  (install mlx-lm for the faster native path)")
        return OpenAICompatLocalJudge(OLLAMA_MODEL, base_url=OLLAMA_URL,
                                      name=OLLAMA_MODEL.replace(":", "-"))


def main():
    print("Loading multilingual data (Aya)...")
    sources = from_aya(LANGUAGES, n_per_lang=N_PER_LANG, min_ref_sentences=2)
    if not sources:
        sys.exit("No sources loaded — check your internet connection for the first Aya download.")
    items = build_dataset(sources, severities=SEVERITIES, seed=7)
    print(f"{len(sources)} references -> {len(items)} gold-labeled items "
          f"across {len({i.language for i in items})} languages\n")

    judge = pick_judge()
    print(f"\nRunning {len(items) * 2} judge calls (both presentation orders)... "
          "this is the slow part.\n")
    results = evaluate([judge], items)

    save_results(results, "mac_results.json", "mac_leaderboard.md")
    print(cards_to_markdown(results))
    print("\nSaved mac_results.json and mac_leaderboard.md")


if __name__ == "__main__":
    main()
