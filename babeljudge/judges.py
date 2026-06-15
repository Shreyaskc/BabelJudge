"""Judge adapters.

A judge takes a prompt and two responses (A, B) and returns "A", "B", or "tie".
The harness handles order randomization and normalization back to
reference/perturbed, so adapters stay simple.

`MockJudge` lets the entire pipeline run with zero network and zero models --
useful for tests and for demonstrating the end-to-end flow. It can be given a
configurable position bias and verbosity bias so you can validate that the
metrics actually detect those biases.

Real backends are all free:
  * HuggingFace `transformers` running a quantized open-weight model
    (Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Aya-Expanse-8B, Gemma-2-9B)
    on a free Colab/Kaggle T4 GPU.
  * Any OpenAI-compatible *local* server (vLLM / llama.cpp / LM Studio /
    Ollama) via a base_url -- no paid API key.
Wiring stubs are included below, commented, so nothing here requires network.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod

# Standard pairwise prompt template. Real runs localize the INSTRUCTION into the
# target language for the prompt-language probe; kept English here for clarity.
PAIRWISE_TEMPLATE = (
    "{instruction}\n\n"
    "Question:\n{prompt}\n\n"
    "Response A:\n{a}\n\n"
    "Response B:\n{b}\n\n"
    "Which response is better? Answer with exactly 'A' or 'B'."
)
DEFAULT_INSTRUCTION = (
    "You are evaluating two responses to the same question. "
    "Judge which response is more accurate, complete, and coherent."
)


class Judge(ABC):
    name: str = "abstract"

    @abstractmethod
    def compare(self, prompt: str, a: str, b: str, instruction: str = DEFAULT_INSTRUCTION) -> str:
        """Return 'A', 'B', or 'tie'."""
        raise NotImplementedError


class MockJudge(Judge):
    """Deterministic mock judge with injectable, controllable biases.

    Parameters
    ----------
    skill : float in [0,1]
        Probability the judge correctly recognizes the better (longer-content,
        non-corrupted) response when no bias interferes.
    position_bias : float
        Added probability of picking slot A regardless of content.
    verbosity_bias : float
        Added probability of picking whichever response has more characters.
    """

    def __init__(self, name: str, skill: float = 0.75,
                 position_bias: float = 0.0, verbosity_bias: float = 0.0, seed: int = 0):
        self.name = name
        self.skill = skill
        self.position_bias = position_bias
        self.verbosity_bias = verbosity_bias
        self._rng = random.Random(seed)

    def compare(self, prompt: str, a: str, b: str, instruction: str = DEFAULT_INSTRUCTION) -> str:
        # Content-quality proxy = information density: unique tokens weighted by
        # how non-repetitive the text is. Pure padding (repeat_pad) adds length
        # but lowers density, so a competent judge is NOT fooled by it -- unless
        # it has a verbosity bias keyed to raw length.
        def density(t: str) -> float:
            toks = t.split()
            uniq = len(set(toks))
            return uniq * (uniq / max(1, len(toks)))
        qa, qb = density(a), density(b)
        score_a = 0.0
        if qa > qb:
            score_a += self.skill
        elif qb > qa:
            score_a += (1 - self.skill)
        else:
            score_a += 0.5
        score_a += self.position_bias                       # slot-A pull
        if len(a) > len(b):
            score_a += self.verbosity_bias                  # fooled by raw length
        elif len(b) > len(a):
            score_a -= self.verbosity_bias
        return "A" if self._rng.random() < max(0.0, min(1.0, score_a)) else "B"


def _parse_ab(text: str) -> str:
    """Normalize a judge's free-text answer to 'A', 'B', or 'tie'."""
    t = (text or "").strip().upper()
    # look at the first informative character/token
    for ch in t:
        if ch == "A":
            return "A"
        if ch == "B":
            return "B"
    return "tie"


# ---------------------------------------------------------------------------
# FREE real backends. Deps are imported lazily so the package imports fine
# without them; install only when you actually run a real judge.
# ---------------------------------------------------------------------------
class TransformersJudge(Judge):
    """Open-weight judge via HuggingFace transformers, 4-bit by default so a
    7-8B model fits a free Colab/Kaggle T4 (16 GB).

    Good multilingual choices: Qwen/Qwen2.5-7B-Instruct,
    CohereForAI/aya-expanse-8b, google/gemma-2-9b-it. For a tighter memory
    budget use Qwen/Qwen2.5-3B-Instruct.
    """

    def __init__(self, model_id: str, name: str | None = None,
                 load_in_4bit: bool = True, max_new_tokens: int = 4):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.name = name or model_id.split("/")[-1]
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model_id)
        kwargs = {"device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        else:
            kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    def compare(self, prompt: str, a: str, b: str, instruction: str = DEFAULT_INSTRUCTION) -> str:
        text = PAIRWISE_TEMPLATE.format(instruction=instruction, prompt=prompt, a=a, b=b)
        msgs = [{"role": "user", "content": text}]
        ids = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        out = self.model.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        ans = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        return _parse_ab(ans)


class MLXJudge(Judge):
    """Native Apple Silicon judge via mlx-lm — the fastest local path on a Mac.

    4-bit MLX community models are tiny and fast on an M-series chip, e.g.:
      mlx-community/Qwen2.5-7B-Instruct-4bit     (~4.3 GB)
      mlx-community/Qwen2.5-14B-Instruct-4bit    (~8 GB, fine on 24 GB)
      mlx-community/aya-expanse-8b-4bit          (stronger multilingual)
    The model downloads from HuggingFace on first use, then runs offline.
    """

    def __init__(self, model_id: str = "mlx-community/Qwen2.5-7B-Instruct-4bit",
                 name: str | None = None, max_new_tokens: int = 4):
        from mlx_lm import load, generate
        self.name = name or model_id.split("/")[-1]
        self.max_new_tokens = max_new_tokens
        self.model, self.tok = load(model_id)
        self._generate = generate

    def compare(self, prompt: str, a: str, b: str, instruction: str = DEFAULT_INSTRUCTION) -> str:
        text = PAIRWISE_TEMPLATE.format(instruction=instruction, prompt=prompt, a=a, b=b)
        if getattr(self.tok, "chat_template", None):
            p = self.tok.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True, tokenize=False)
        else:
            p = text
        out = self._generate(self.model, self.tok, p,
                             max_tokens=self.max_new_tokens, verbose=False)
        return _parse_ab(out)


class OpenAICompatLocalJudge(Judge):
    """Judge served by a LOCAL OpenAI-compatible server (vLLM / llama.cpp /
    LM Studio / Ollama). The api_key is a dummy; nothing is paid.

    Also works with any hosted OpenAI-compatible API (Groq, Together, Mistral,
    Anyscale, etc.) — just pass the real base_url and api_key:

        OpenAICompatLocalJudge("llama-3.1-70b-versatile",
                               base_url="https://api.groq.com/openai/v1",
                               api_key="gsk_...")
    """

    def __init__(self, model_id: str, base_url: str = "http://localhost:8000/v1",
                 name: str | None = None, api_key: str = "local"):
        from openai import OpenAI
        self.name = name or model_id
        self.model_id = model_id
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def compare(self, prompt: str, a: str, b: str, instruction: str = DEFAULT_INSTRUCTION) -> str:
        text = PAIRWISE_TEMPLATE.format(instruction=instruction, prompt=prompt, a=a, b=b)
        r = self.client.chat.completions.create(
            model=self.model_id, messages=[{"role": "user", "content": text}],
            max_tokens=4, temperature=0.0)
        return _parse_ab(r.choices[0].message.content)


class OllamaJudge(OpenAICompatLocalJudge):
    """Convenience wrapper for a local Ollama server.

    Requires `brew install ollama && ollama pull <model>`.
    The Ollama OpenAI-compat endpoint lives at localhost:11434/v1.

    Example
    -------
        judge = OllamaJudge("qwen2.5:7b")
        judge = OllamaJudge("llama3.1:8b", host="http://192.168.1.5:11434")
    """

    def __init__(self, model_id: str = "qwen2.5:7b", name: str | None = None,
                 host: str = "http://localhost:11434"):
        super().__init__(model_id, base_url=f"{host}/v1", name=name, api_key="ollama")


class AnthropicJudge(Judge):
    """Judge powered by a Claude model via the Anthropic API.

    Requires `pip install anthropic` and the ANTHROPIC_API_KEY env var (or pass
    api_key directly). Defaults to claude-opus-4-8.

    Example
    -------
        from babeljudge import AnthropicJudge, evaluate
        judge = AnthropicJudge()                      # claude-opus-4-8
        judge = AnthropicJudge("claude-haiku-4-5")    # cheaper / faster
        results = evaluate([judge], items)
    """

    def __init__(self, model_id: str = "claude-opus-4-8", name: str | None = None,
                 api_key: str | None = None):
        import anthropic
        self.name = name or model_id
        self.model_id = model_id
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def compare(self, prompt: str, a: str, b: str, instruction: str = DEFAULT_INSTRUCTION) -> str:
        text = PAIRWISE_TEMPLATE.format(instruction=instruction, prompt=prompt, a=a, b=b)
        msg = self.client.messages.create(
            model=self.model_id,
            max_tokens=16,
            messages=[{"role": "user", "content": text}],
        )
        return _parse_ab(msg.content[0].text)


class RemoteOpenAIJudge(Judge):
    """Judge powered by a real OpenAI model (GPT-4o, GPT-4-turbo, etc.).

    Requires `pip install openai` and the OPENAI_API_KEY env var (or pass
    api_key directly).

    Example
    -------
        from babeljudge import RemoteOpenAIJudge, evaluate
        judge = RemoteOpenAIJudge()              # gpt-4o
        judge = RemoteOpenAIJudge("gpt-4-turbo") # specific version
        results = evaluate([judge], items)
    """

    def __init__(self, model_id: str = "gpt-4o", name: str | None = None,
                 api_key: str | None = None):
        from openai import OpenAI
        self.name = name or model_id
        self.model_id = model_id
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def compare(self, prompt: str, a: str, b: str, instruction: str = DEFAULT_INSTRUCTION) -> str:
        text = PAIRWISE_TEMPLATE.format(instruction=instruction, prompt=prompt, a=a, b=b)
        r = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": text}],
            max_tokens=4, temperature=0.0)
        return _parse_ab(r.choices[0].message.content)


class GeminiJudge(Judge):
    """Judge powered by Google Gemini via the google-generativeai SDK.

    Requires `pip install google-generativeai` and the GOOGLE_API_KEY env var
    (or pass api_key directly).

    Example
    -------
        judge = GeminiJudge()                          # gemini-1.5-pro
        judge = GeminiJudge("gemini-1.5-flash")        # faster / cheaper
        judge = GeminiJudge(api_key="AIza...")
    """

    def __init__(self, model_id: str = "gemini-1.5-pro", name: str | None = None,
                 api_key: str | None = None):
        import google.generativeai as genai
        import os
        genai.configure(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
        self.name = name or model_id
        self._model = genai.GenerativeModel(model_id)

    def compare(self, prompt: str, a: str, b: str, instruction: str = DEFAULT_INSTRUCTION) -> str:
        text = PAIRWISE_TEMPLATE.format(instruction=instruction, prompt=prompt, a=a, b=b)
        r = self._model.generate_content(text)
        return _parse_ab(r.text)


# ---------------------------------------------------------------------------
# Factory — one call covers every backend
# ---------------------------------------------------------------------------

_BACKEND_ALIASES: dict[str, str] = {
    # local native
    "mlx": "mlx", "apple": "mlx",
    "transformers": "transformers", "hf": "transformers", "huggingface": "transformers",
    # local server
    "ollama": "ollama",
    "vllm": "compat", "llamacpp": "compat", "lmstudio": "compat", "compat": "compat",
    # cloud proprietary
    "anthropic": "anthropic", "claude": "anthropic",
    "openai": "openai", "gpt": "openai",
    "gemini": "gemini", "google": "gemini",
    # cloud OpenAI-compat
    "groq": "groq",
    "together": "together",
    "mistral": "mistral",
    # testing
    "mock": "mock",
}

_GROQ_URL = "https://api.groq.com/openai/v1"
_TOGETHER_URL = "https://api.together.xyz/v1"
_MISTRAL_URL = "https://api.mistral.ai/v1"


def create_judge(backend: str, model: str | None = None, **kwargs) -> Judge:
    """Factory — create any judge by name without importing the class directly.

    Parameters
    ----------
    backend : str
        One of: mlx, transformers, ollama, compat, anthropic/claude,
        openai/gpt, gemini/google, groq, together, mistral, mock.
        Aliases: hf, huggingface, apple, vllm, llamacpp, lmstudio, gpt.
    model : str, optional
        Model ID; each backend has a sensible default.
    **kwargs
        Forwarded verbatim to the chosen Judge constructor.
        Common extras: api_key, name, base_url (for compat), host (for ollama).

    Examples
    --------
        # Local native (Apple Silicon)
        create_judge("mlx")
        create_judge("mlx", "mlx-community/Qwen2.5-14B-Instruct-4bit")

        # Local server
        create_judge("ollama", "llama3.1:8b")
        create_judge("compat", "my-model", base_url="http://localhost:8080/v1")

        # Cloud
        create_judge("anthropic")
        create_judge("openai", "gpt-4-turbo")
        create_judge("gemini", "gemini-1.5-flash")
        create_judge("groq", "llama-3.1-70b-versatile", api_key="gsk_...")
        create_judge("together", api_key="...")
        create_judge("mistral", "mistral-large-latest", api_key="...")

        # Testing
        create_judge("mock", name="always-A", skill=1.0)
    """
    key = _BACKEND_ALIASES.get(backend.lower())
    if key is None:
        raise ValueError(
            f"Unknown backend {backend!r}. "
            f"Valid choices: {', '.join(sorted(set(_BACKEND_ALIASES)))}"
        )

    if key == "mlx":
        return MLXJudge(model or "mlx-community/Qwen2.5-7B-Instruct-4bit", **kwargs)
    if key == "transformers":
        return TransformersJudge(model or "Qwen/Qwen2.5-7B-Instruct", **kwargs)
    if key == "ollama":
        return OllamaJudge(model or "qwen2.5:7b", **kwargs)
    if key == "compat":
        if model is None:
            raise ValueError("create_judge('compat') requires a model name")
        return OpenAICompatLocalJudge(model, **kwargs)
    if key == "anthropic":
        return AnthropicJudge(model or "claude-opus-4-8", **kwargs)
    if key == "openai":
        return RemoteOpenAIJudge(model or "gpt-4o", **kwargs)
    if key == "gemini":
        return GeminiJudge(model or "gemini-1.5-pro", **kwargs)
    if key == "groq":
        return OpenAICompatLocalJudge(
            model or "llama-3.1-70b-versatile",
            base_url=_GROQ_URL, **kwargs)
    if key == "together":
        return OpenAICompatLocalJudge(
            model or "meta-llama/Llama-3-70b-chat-hf",
            base_url=_TOGETHER_URL, **kwargs)
    if key == "mistral":
        return OpenAICompatLocalJudge(
            model or "mistral-large-latest",
            base_url=_MISTRAL_URL, **kwargs)
    if key == "mock":
        return MockJudge(kwargs.pop("name", model or "mock"), **kwargs)

    raise AssertionError(f"unhandled key {key!r}")  # should never reach here
