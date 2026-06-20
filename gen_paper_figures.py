"""Generate paper-quality figures for the arXiv submission.

Outputs to paper/figures/ with white backgrounds, academic typography,
and vector-friendly 300 DPI PNGs suitable for inclusion in LaTeX.

Run from repo root:
    python gen_paper_figures.py
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

OUT = "paper/figures"

# ── Academic colour palette (WCAG-accessible, prints well in greyscale) ──────
BLACK  = "#1A1A1A"
DKGREY = "#444444"
GREY   = "#666666"
LGREY  = "#999999"
VLTGRY = "#F2F2F2"
WHITE  = "#FFFFFF"
BORDER = "#CCCCCC"

BLUE   = "#2166AC"   # accessible blue  (ColorBrewer 2-class diverging)
RED    = "#D6604D"   # accessible red
GREEN  = "#4DAC26"   # accessible green
ORANGE = "#E08214"   # accessible orange
PURPLE = "#762A83"   # accessible purple

LANG_COLORS = {"en": BLUE, "hi": GREEN, "ar": ORANGE, "sw": RED}
LANG_LABELS = {"en": "English", "hi": "Hindi", "ar": "Arabic", "sw": "Swahili"}
LANGS = ["en", "hi", "ar", "sw"]

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.labelsize":     10,
    "axes.titlesize":     11,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "text.color":         BLACK,
    "axes.labelcolor":    DKGREY,
    "xtick.color":        DKGREY,
    "ytick.color":        DKGREY,
    "axes.edgecolor":     BORDER,
    "axes.facecolor":     WHITE,
    "figure.facecolor":   WHITE,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "grid.color":         "#E8E8E8",
    "grid.linewidth":     0.7,
    "axes.grid":          True,
    "axes.grid.axis":     "y",
    "legend.frameon":     False,
    "legend.fontsize":    9,
})


DATA  = json.load(open("mac_results.json"))
CARDS = {c["language"]: c for c in DATA["cards"]}


def save(fig, name):
    path = f"{OUT}/{name}"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"  saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Methodology pipeline
# ─────────────────────────────────────────────────────────────────────────────
def fig_methodology():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    # helper: draw a box
    def box(cx, cy, w, h, label, sublabel=None, color=BLUE, alpha=0.12, fontsize=9.5):
        rect = FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle="round,pad=0.08",
            facecolor=color, alpha=alpha,
            edgecolor=color, linewidth=1.5,
        )
        ax.add_patch(rect)
        if sublabel:
            ax.text(cx, cy + 0.10, label, ha="center", va="center",
                    fontsize=fontsize, fontweight="bold", color=color)
            ax.text(cx, cy - 0.22, sublabel, ha="center", va="center",
                    fontsize=7.5, color=GREY, style="italic")
        else:
            ax.text(cx, cy, label, ha="center", va="center",
                    fontsize=fontsize, fontweight="bold", color=color)

    # helper: draw a horizontal arrow
    def arrow(x0, x1, y, color=DKGREY):
        ax.annotate("",
            xy=(x1, y), xytext=(x0, y),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4,
                            mutation_scale=14))

    # helper: draw an angled arrow
    def arrow_angled(x0, y0, x1, y1, color=DKGREY):
        ax.annotate("",
            xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4,
                            mutation_scale=14,
                            connectionstyle="arc3,rad=0.0"))

    # ── Row 1: main pipeline ────────────────────────────────────────────────
    Y_TOP = 3.3

    # Box 1: corpus
    box(1.05, Y_TOP, 1.7, 0.90,
        "Multilingual\nCorpus",
        "(Aya, FLORES-200)", color=PURPLE)

    arrow(1.90, 2.45, Y_TOP, color=DKGREY)

    # Box 2: perturbation
    box(3.10, Y_TOP, 1.30, 0.90,
        "Perturbation\nEngine",
        "5 types × 2 sev.", color=BLUE)

    arrow(3.75, 4.35, Y_TOP, color=DKGREY)

    # Box 3: gold-labeled pair
    box(5.10, Y_TOP, 1.50, 0.90,
        "Gold-Labeled\nItem",
        "gold = reference", color=GREEN)

    # fork down-left
    arrow_angled(5.10, Y_TOP - 0.45, 5.10, 2.40, color=DKGREY)

    # ── Row 2: dual-order presentation ─────────────────────────────────────
    Y_MID = 2.00

    # left arm: ref-first
    ax.annotate("",
        xy=(3.80, Y_MID + 0.25), xytext=(5.10, 2.40),
        arrowprops=dict(arrowstyle="-|>", color=DKGREY, lw=1.2,
                        mutation_scale=12,
                        connectionstyle="arc3,rad=0.15"))

    box(3.20, Y_MID, 1.70, 0.60,
        "Order A: (ref, pert)",
        None, color=ORANGE, fontsize=8.5)

    # right arm: pert-first
    ax.annotate("",
        xy=(6.40, Y_MID + 0.25), xytext=(5.10, 2.40),
        arrowprops=dict(arrowstyle="-|>", color=DKGREY, lw=1.2,
                        mutation_scale=12,
                        connectionstyle="arc3,rad=-0.15"))

    box(7.00, Y_MID, 1.70, 0.60,
        "Order B: (pert, ref)",
        None, color=ORANGE, fontsize=8.5)

    # label between orders
    ax.text(5.10, Y_MID, "×2 orders\n(position\nbias probe)",
            ha="center", va="center", fontsize=7.5, color=GREY, style="italic")

    # ── Row 3: judge ────────────────────────────────────────────────────────
    Y_BOT = 0.95

    # merge arrows
    ax.annotate("",
        xy=(5.10, Y_BOT + 0.55), xytext=(3.20, Y_MID - 0.30),
        arrowprops=dict(arrowstyle="-|>", color=DKGREY, lw=1.2,
                        mutation_scale=12,
                        connectionstyle="arc3,rad=-0.15"))
    ax.annotate("",
        xy=(5.10, Y_BOT + 0.55), xytext=(7.00, Y_MID - 0.30),
        arrowprops=dict(arrowstyle="-|>", color=DKGREY, lw=1.2,
                        mutation_scale=12,
                        connectionstyle="arc3,rad=0.15"))

    box(5.10, Y_BOT, 1.60, 0.70,
        "LLM Judge",
        "returns A / B / tie", color=BLUE)

    arrow(5.90, 6.80, Y_BOT, color=DKGREY)

    # Box 5: reliability card
    box(8.00, Y_BOT, 2.10, 0.90,
        "Reliability Card",
        "acc · pos-bias · verbosity · consistency", color=GREEN)

    # ── Title & footnote ────────────────────────────────────────────────────
    ax.text(5.10, 4.05,
            "BabelJudge: Gold-Labelling-by-Degradation Pipeline",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color=BLACK)

    ax.text(0.05, 0.02,
            "Gold label = reference (the unperturbed response is by construction "
            "the better one — no human annotation required).",
            ha="left", va="bottom", fontsize=7.5, color=GREY, style="italic",
            transform=ax.transAxes)

    fig.tight_layout()
    save(fig, "methodology.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Reliability vs Raw Accuracy
# ─────────────────────────────────────────────────────────────────────────────
def fig_reliability_vs_accuracy():
    fig, ax = plt.subplots(figsize=(7.5, 3.8))

    langs    = LANGS
    accuracy = [CARDS[l]["accuracy"]         for l in langs]
    reliab   = [CARDS[l]["reliability_score"] for l in langs]
    labels   = [f"{LANG_LABELS[l]}\n({l})"   for l in langs]
    colors   = [LANG_COLORS[l]               for l in langs]

    x = np.arange(len(langs))
    w = 0.32

    acc_bars = ax.bar(x - w/2, accuracy, w, color=LGREY, alpha=0.55,
                      label="Raw Accuracy", zorder=3, edgecolor=GREY, linewidth=0.5)
    rel_bars = ax.bar(x + w/2, reliab,   w, color=colors, alpha=0.90,
                      label="Reliability Score", zorder=3, edgecolor=[c for c in colors],
                      linewidth=0.8)

    # value labels
    for bar, val, col in zip(rel_bars, reliab, colors):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.012,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=col)

    for bar, val in zip(acc_bars, accuracy):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.012,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=8, color=GREY)

    # threshold line
    ax.axhline(0.65, color=RED, lw=1.2, ls="--", alpha=0.8, zorder=2)
    ax.text(3.62, 0.655, "0.65 threshold", fontsize=8, color=RED, va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, 1.00)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(
        "Reliability Score vs Raw Accuracy — Qwen2.5-7B-Instruct-4bit",
        fontsize=11, fontweight="bold", color=BLACK, pad=10)

    legend_patches = [
        mpatches.Patch(color=LGREY, alpha=0.6, label="Raw Accuracy"),
        mpatches.Patch(color=BLUE,  alpha=0.9, label="Reliability Score"),
    ]
    ax.legend(handles=legend_patches, loc="lower left", fontsize=9)

    ax.text(0.99, 0.02,
            "Reliability = competence × (1 − bias_penalty)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color=LGREY, style="italic")

    # annotate the Swahili gap
    sw_idx = LANGS.index("sw")
    ax.annotate(
        "Reliability drops\n10 pts below accuracy\ndue to low order\nconsistency (0.48)",
        xy=(sw_idx + w/2, CARDS["sw"]["reliability_score"]),
        xytext=(sw_idx + 0.85, 0.72),
        fontsize=7.5, color=DKGREY,
        arrowprops=dict(arrowstyle="-|>", color=DKGREY, lw=0.9, mutation_scale=10),
        ha="left",
    )

    fig.tight_layout()
    save(fig, "reliability_vs_accuracy.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Four-panel bias breakdown
# ─────────────────────────────────────────────────────────────────────────────
def fig_bias_breakdown():
    fig = plt.figure(figsize=(11, 5.0))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.58, wspace=0.35,
                           left=0.07, right=0.97, top=0.88, bottom=0.12)

    x   = np.arange(len(LANGS))
    bw  = 0.52

    panels = [
        ("accuracy",                "Accuracy",               None,   BLUE,   (0, 0),
         "Fraction of items where the judge correctly preferred the reference response."),
        ("position_bias_delta",     "Position Bias Δ",  0.0,    ORANGE, (0, 1),
         "Win-rate difference between slot-A and slot-B. Ideal: near 0."),
        ("verbosity_susceptibility","Verbosity Susceptibility", None, PURPLE, (1, 0),
         "Rate of preferring the longer-but-redundant (repeat_pad) response. Ideal: < 0.20."),
        ("order_consistency",       "Order Consistency",      0.65,   GREEN,  (1, 1),
         "Fraction of items with matching verdicts under both presentation orders."),
    ]

    for key, title, hline, base_color, pos, caption in panels:
        ax = fig.add_subplot(gs[pos])
        vals = [CARDS[l][key] for l in LANGS]

        # per-bar conditional colouring
        bar_colors = []
        for l, v in zip(LANGS, vals):
            if key == "position_bias_delta":
                bar_colors.append(RED if abs(v) > 0.10 else GREEN)
            elif key == "order_consistency":
                bar_colors.append(RED if v < 0.60 else GREEN)
            elif key == "verbosity_susceptibility":
                bar_colors.append(RED if v > 0.25 else (ORANGE if v > 0.15 else GREEN))
            else:
                bar_colors.append(base_color)

        bars = ax.bar(x, vals, bw, color=bar_colors, alpha=0.80, zorder=3,
                      edgecolor=bar_colors, linewidth=0.6)

        for bar, val, col in zip(bars, vals, bar_colors):
            fmt = f"{val:+.3f}" if key == "position_bias_delta" else f"{val:.3f}"
            va  = "bottom" if val >= 0 else "top"
            off = 0.012   if val >= 0 else -0.012
            ax.text(bar.get_x() + bar.get_width()/2, val + off, fmt,
                    ha="center", va=va, fontsize=8, fontweight="bold", color=col)

        if hline is not None:
            ax.axhline(hline, color=LGREY, lw=1.0, ls="--", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{LANG_LABELS[l]}\n({l})" for l in LANGS], fontsize=8.5)
        ax.set_title(title, fontsize=10.5, fontweight="bold", color=BLACK, pad=6)

        if key == "position_bias_delta":
            yabs = max(abs(v) for v in vals) + 0.07
            ax.set_ylim(-yabs, yabs)

        ax.text(0.5, -0.30, caption, transform=ax.transAxes,
                ha="center", fontsize=7.5, color=GREY, style="italic",
                wrap=True)

    fig.suptitle(
        "Per-Language Bias Breakdown — Qwen2.5-7B-Instruct-4bit  (Aya dataset, 100 items/lang)",
        fontsize=11, fontweight="bold", color=BLACK, y=0.96)

    save(fig, "bias_breakdown.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Reliability radar (one subplot per language)
# ─────────────────────────────────────────────────────────────────────────────
def fig_radar():
    metrics = ["Accuracy", "Order\nConsistency", "Pos-Bias\n(inverted)",
               "Verbosity\n(inverted)", "Reliability"]
    N      = len(metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4),
                             subplot_kw=dict(projection="polar"))
    fig.patch.set_facecolor(WHITE)
    fig.suptitle("Reliability Radar — per language  (Qwen2.5-7B-Instruct-4bit)",
                 fontsize=11, fontweight="bold", color=BLACK, y=1.03)

    for ax, lang in zip(axes, LANGS):
        c   = CARDS[lang]
        col = LANG_COLORS[lang]
        vals = [
            c["accuracy"],
            c["order_consistency"],
            1 - min(abs(c["position_bias_delta"]) * 3, 1),
            1 - min(c["verbosity_susceptibility"] * 2, 1),
            c["reliability_score"],
        ]
        vals += vals[:1]

        ax.plot(angles, vals, color=col, linewidth=2.0)
        ax.fill(angles, vals, color=col, alpha=0.15)

        ax.set_facecolor("#FAFAFA")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=7.5, color=DKGREY)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"],
                           fontsize=6, color=LGREY)
        ax.tick_params(colors=LGREY)
        ax.spines["polar"].set_color(BORDER)
        ax.yaxis.grid(True, color=BORDER,  linewidth=0.5)
        ax.xaxis.grid(True, color="#DEDEDE", linewidth=0.5)

        ax.set_title(f"{LANG_LABELS[lang]} ({lang})\nR = {c['reliability_score']:.3f}",
                     fontsize=9.5, fontweight="bold", color=col, pad=14)

    fig.tight_layout()
    save(fig, "radar_per_language.png")


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    print("Generating paper figures...")
    fig_methodology()
    fig_reliability_vs_accuracy()
    fig_bias_breakdown()
    fig_radar()
    print(f"Done — 4 figures in ./{OUT}/")
