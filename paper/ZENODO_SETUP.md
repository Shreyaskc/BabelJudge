# Zenodo DOI Setup (one-time, ~5 minutes)

Zenodo mints a persistent DOI for each GitHub release. Once connected, every
`git tag` + GitHub Release automatically gets a citable DOI. Academic tooling
(Zotero, Google Scholar, Semantic Scholar) picks it up within 48 hours.

## Steps

1. **Log in to Zenodo**
   Go to https://zenodo.org and sign in with your GitHub account.

2. **Connect your GitHub repo**
   - Click your name (top right) → Settings → GitHub
   - Find `BabelJudge` in the repository list
   - Flip the toggle to ON

3. **Create a GitHub Release (triggers the DOI)**
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
   Then go to github.com/Shreyaskc/BabelJudge → Releases → "Draft a new release"
   - Tag: `v0.1.0`
   - Title: `BabelJudge v0.1.0 — Initial release`
   - Description: paste from RELEASE_NOTES below
   - Click "Publish release"

4. **Get the DOI from Zenodo**
   - Go to https://zenodo.org/account/settings/github/
   - BabelJudge should now show a DOI badge
   - Copy the DOI (format: `10.5281/zenodo.XXXXXXX`)

5. **Update CITATION.cff with the DOI**
   Edit `CITATION.cff`, add under `preferred-citation`:
   ```yaml
   doi: "10.5281/zenodo.XXXXXXX"
   ```

6. **Add the Zenodo badge to README.md**
   Zenodo gives you a badge URL. Add it next to the license badge at the top
   of README.md:
   ```markdown
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
   ```

## RELEASE_NOTES for v0.1.0

```
Initial release of BabelJudge — a reproducible benchmark for LLM-as-a-judge
reliability across languages.

**What's included:**
- Gold-labeling-by-degradation methodology (5 perturbation types)
- 4-metric reliability framework: accuracy, position bias, verbosity susceptibility, order consistency
- 11 judge backends: MLX, Transformers, Ollama, vLLM, Anthropic, OpenAI, Gemini, Groq, Together, Mistral, Mock
- Benchmark results: Qwen2.5-7B-Instruct-4bit on en/hi/ar/sw (Aya dataset)
- Agentic extension: 9 trajectory-level perturbations + tool_accuracy, hallucination_detection, trajectory_length_bias metrics
- Zero-network demo: `python examples/run_agentic_demo.py`

**Cite as:**
Shreyas KC. (2026). BabelJudge: Measuring LLM-as-a-Judge Reliability Across Languages (v0.1.0). Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX
```
