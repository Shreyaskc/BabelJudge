"""End-to-end demo on synthetic multilingual data, zero network / zero models.

Builds gold-labeled items, runs three judges with *known* injected biases, and
prints the leaderboard + reliability cards. The point is to show the pipeline
works and that the metrics recover the biases we put in:

  fair_judge      -> high accuracy, ~0 position bias, low verbosity susceptibility
  slotA_judge     -> large positive position-bias delta, low order-consistency
  verbose_judge   -> high verbosity susceptibility (fooled by longer responses)
"""
from babeljudge import build_dataset, evaluate, MockJudge, cards_to_markdown, save_results

# A handful of (id, language, prompt, reference) sources standing in for the
# real permissively-licensed multilingual sources (FLORES / WikiLingua / Aya).
SOURCES = [
    {"id": "q1", "language": "en", "prompt": "Explain why the sky appears blue.",
     "reference": "Sunlight contains many colors. Air molecules scatter shorter blue "
                  "wavelengths more strongly than longer red ones. This Rayleigh "
                  "scattering sends blue light in all directions. So the daytime sky "
                  "looks blue to observers on the ground."},
    {"id": "q2", "language": "en", "prompt": "Summarize the water cycle.",
     "reference": "Water evaporates from oceans and lakes into vapor. The vapor rises "
                  "and condenses into clouds. Precipitation returns water to the "
                  "surface as rain or snow. Rivers and groundwater carry it back to "
                  "the sea, completing the cycle."},
    {"id": "q3", "language": "es", "prompt": "Describe la fotosintesis brevemente.",
     "reference": "Las plantas absorben luz solar mediante la clorofila. Convierten el "
                  "dioxido de carbono y el agua en glucosa. Liberan oxigeno como "
                  "subproducto. Este proceso ocurre principalmente en las hojas."},
    {"id": "q4", "language": "sw", "prompt": "Eleza kwa ufupi mzunguko wa maji.",
     "reference": "Maji huvukiza kutoka baharini na kuwa mvuke. Mvuke hupanda juu na "
                  "kuganda kuwa mawingu. Mvua hurudisha maji ardhini. Mito hubeba maji "
                  "kurudi baharini na mzunguko huendelea."},
    {"id": "q5", "language": "hi", "prompt": "Prakash sanshleshan samjhaiye.",
     "reference": "Paudhe suryaprakash ko sokhte hain. Ve carbon dioxide aur jal ko "
                  "glucose mein badalte hain. Is prakriya mein oxygen nikalti hai. "
                  "Yah mukhya roop se pattiyon mein hoti hai."},
]


def main():
    items = build_dataset(
        SOURCES,
        perturbations=["truncate", "shuffle", "number_corrupt", "drop_entities", "repeat_pad"],
        severities=(0.25, 0.4, 0.55, 0.7),
        seed=7,
    )
    print(f"Built {len(items)} gold-labeled items "
          f"across {len({i.language for i in items})} languages "
          f"from {len(SOURCES)} reference responses.\n")

    judges = [
        MockJudge("fair-7b",    skill=0.85, position_bias=0.0,  verbosity_bias=0.0,  seed=1),
        MockJudge("slotA-7b",   skill=0.85, position_bias=0.35, verbosity_bias=0.0,  seed=2),
        MockJudge("verbose-7b", skill=0.85, position_bias=0.0,  verbosity_bias=0.45, seed=3),
    ]

    results = evaluate(judges, items)
    print(cards_to_markdown(results))
    save_results(results, "demo_results.json", "demo_leaderboard.md")
    print("\n\nWrote demo_results.json and demo_leaderboard.md")


if __name__ == "__main__":
    main()
