"""BabelJudge: a reproducible benchmark and living leaderboard for
LLM-as-a-judge reliability across languages."""
from .schema import JudgeItem, Judgment
from .perturbations import make_item, build_dataset, PERTURBATIONS
from .judges import (Judge, MockJudge, TransformersJudge, MLXJudge,
                     OpenAICompatLocalJudge, OllamaJudge,
                     AnthropicJudge, RemoteOpenAIJudge, GeminiJudge,
                     create_judge)
from .harness import evaluate, run_judge, cards_to_markdown, save_results
from .sources import from_xlsum, from_aya, from_flores, records_to_sources

__version__ = "0.1.0"
__all__ = [
    "JudgeItem", "Judgment", "make_item", "build_dataset", "PERTURBATIONS",
    "Judge", "MockJudge", "TransformersJudge", "MLXJudge",
    "OpenAICompatLocalJudge", "OllamaJudge",
    "AnthropicJudge", "RemoteOpenAIJudge", "GeminiJudge",
    "create_judge",
    "evaluate", "run_judge", "cards_to_markdown", "save_results",
    "from_xlsum", "from_aya", "from_flores", "records_to_sources",
]
