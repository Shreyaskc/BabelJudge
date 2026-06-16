"""Generate benchmark visualizations for the GitHub README."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

DATA = json.load(open("mac_results.json"))
CARDS = {c["language"]: c for c in DATA["cards"]}

OUT = "docs/img"

# ── Style ─────────────────────────────────────────────────────────────────────
BG    = "#0D1117"
CARD  = "#161B22"
BORDER= "#30363D"
WHITE = "#E6EDF3"
GREY  = "#8B949E"
LGREY = "#484F58"

BLUE   = "#58A6FF"
GREEN  = "#3FB950"
ORANGE = "#F0883E"
RED    = "#F85149"
YELLOW = "#E3B341"
PURPLE = "#BC8CFF"

LANG_COLORS = {"en": BLUE, "hi": GREEN, "ar": ORANGE, "sw": RED}
LANG_LABELS = {"en": "English", "hi": "Hindi", "ar": "Arabic", "sw": "Swahili"}
LANGS = ["en", "hi", "ar", "sw"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color":  WHITE,
    "axes.labelcolor": GREY,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "axes.facecolor": CARD,
    "figure.facecolor": BG,
    "axes.edgecolor": BORDER,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": BORDER,
    "grid.linewidth": 0.7,
    "axes.grid": True,
    "axes.grid.axis": "y",
})


def save(fig, name):
    path = f"{OUT}/{name}"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1 — Reliability vs Raw Accuracy (the headline chart)
# ─────────────────────────────────────────────────────────────────────────────
def chart_reliability_vs_accuracy():
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)

    langs    = LANGS
    accuracy = [CARDS[l]["accuracy"] for l in langs]
    reliab   = [CARDS[l]["reliability_score"] for l in langs]
    labels   = [LANG_LABELS[l] for l in langs]
    colors   = [LANG_COLORS[l] for l in langs]

    x = np.arange(len(langs))
    w = 0.34

    ax.bar(x - w / 2, accuracy, w, color=GREY, alpha=0.40, label="Raw Accuracy", zorder=3)
    bars = ax.bar(x + w / 2, reliab, w, color=colors, alpha=0.90, label="Reliability Score", zorder=3)

    # value labels above reliability bars
    for bar, val, col in zip(bars, reliab, colors):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.008,
                f"{val:.3f}", ha="center", va="bottom",
                color=col, fontsize=10.5, fontweight="bold")

    # value labels above accuracy bars
    for xi, val in zip(x - w / 2, accuracy):
        ax.text(xi + w / 2, val + 0.008, f"{val:.2f}",
                ha="center", va="bottom", color=GREY, fontsize=9)

    ax.axhline(0.65, color=YELLOW, lw=1.2, ls="--", alpha=0.7, zorder=2)
    ax.text(3.82, 0.652, "0.65 threshold", fontsize=8.5, color=YELLOW, va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{LANG_LABELS[l]}\n({l})" for l in langs],
        fontsize=12)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "Reliability Score vs Raw Accuracy — Qwen2.5-7B-Instruct-4bit",
        fontsize=13, fontweight="bold", color=WHITE, pad=14)

    legend_patches = [
        mpatches.Patch(color=GREY, alpha=0.45, label="Raw Accuracy"),
        mpatches.Patch(color=BLUE, alpha=0.90, label="Reliability Score"),
    ]
    ax.legend(handles=legend_patches, frameon=False, labelcolor=GREY,
              fontsize=10, loc="lower left")

    ax.text(0.99, 0.01,
            "Reliability = competence × (1 − bias_penalty)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color=LGREY, fontstyle="italic")

    fig.tight_layout()
    save(fig, "reliability_vs_accuracy.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 2 — Four-panel bias breakdown
# ─────────────────────────────────────────────────────────────────────────────
def chart_bias_breakdown():
    fig = plt.figure(figsize=(12, 6))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.32,
                           left=0.07, right=0.97, top=0.90, bottom=0.10)

    labels = [LANG_LABELS[l] for l in LANGS]
    colors = [LANG_COLORS[l] for l in LANGS]
    x = np.arange(len(LANGS))
    bw = 0.55

    panels = [
        ("accuracy",               "Accuracy",               None,   GREEN,  (0, 0),
         "Fraction of items where the judge correctly preferred the reference response."),
        ("position_bias_delta",    "Position Bias Δ",   0.0,    ORANGE, (0, 1),
         "Win-rate difference between slot-A and slot-B presentation. Ideal: near 0."),
        ("verbosity_susceptibility","Verbosity Susceptibility", None, YELLOW, (1, 0),
         "Rate of preferring longer-but-redundant (repeat_pad) responses. Ideal: < 0.20."),
        ("order_consistency",      "Order Consistency",      0.65,   BLUE,   (1, 1),
         "Fraction of items with matching verdicts under both presentation orders."),
    ]

    for key, title, hline, color, pos, caption in panels:
        ax = fig.add_subplot(gs[pos])
        vals = [CARDS[l][key] for l in LANGS]

        bar_colors = []
        for l, v in zip(LANGS, vals):
            if key == "position_bias_delta":
                bar_colors.append(ORANGE if abs(v) > 0.10 else GREEN)
            elif key == "order_consistency":
                bar_colors.append(RED if v < 0.60 else GREEN)
            elif key == "verbosity_susceptibility":
                bar_colors.append(RED if v > 0.30 else (YELLOW if v > 0.15 else GREEN))
            else:
                bar_colors.append(color)

        bars = ax.bar(x, vals, bw, color=bar_colors, alpha=0.85, zorder=3)

        for bar, val in zip(bars, vals):
            va = "bottom" if val >= 0 else "top"
            offset = 0.012 if val >= 0 else -0.012
            ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                    f"{val:+.3f}" if key == "position_bias_delta" else f"{val:.3f}",
                    ha="center", va=va, fontsize=9, fontweight="bold",
                    color=bar_colors[list(x).index(bar.get_x() + bar.get_width() / 2)
                                     if False else bars.patches.index(bar)])

        if hline is not None:
            ax.axhline(hline, color=LGREY, lw=1.0, ls="--", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{LANG_LABELS[l]}\n({l})" for l in LANGS], fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold", color=WHITE, pad=8)
        ax.tick_params(labelsize=8.5)

        # caption below subplot
        ax.text(0.5, -0.28, caption, transform=ax.transAxes,
                ha="center", fontsize=7.5, color=LGREY, fontstyle="italic",
                wrap=True)

        if key == "position_bias_delta":
            yabs = max(abs(v) for v in vals) + 0.06
            ax.set_ylim(-yabs, yabs)

    fig.suptitle(
        "Per-Language Bias Breakdown — Qwen2.5-7B-Instruct-4bit  (Aya dataset, 100 items/lang)",
        fontsize=12.5, fontweight="bold", color=WHITE, y=0.97)

    save(fig, "bias_breakdown.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 3 — Reliability radar / spider chart per language
# ─────────────────────────────────────────────────────────────────────────────
def chart_radar():
    metrics = ["Accuracy", "Order\nConsistency", "Pos-Bias\n(inverted)", "Verbosity\n(inverted)", "Reliability"]
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, axes = plt.subplots(1, 4, figsize=(14, 4),
                             subplot_kw=dict(projection="polar"))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Reliability Radar — per language",
                 fontsize=13, fontweight="bold", color=WHITE, y=1.02)

    for ax, lang in zip(axes, LANGS):
        c = CARDS[lang]
        # normalise all to 0-1, higher always = better
        vals = [
            c["accuracy"],
            c["order_consistency"],
            1 - min(abs(c["position_bias_delta"]) * 3, 1),   # penalise abs(delta)
            1 - min(c["verbosity_susceptibility"] * 2, 1),   # penalise susceptibility
            c["reliability_score"],
        ]
        vals += vals[:1]

        col = LANG_COLORS[lang]
        ax.plot(angles, vals, color=col, linewidth=2)
        ax.fill(angles, vals, color=col, alpha=0.18)

        ax.set_facecolor(CARD)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=7.5, color=WHITE)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=6, color=LGREY)
        ax.tick_params(colors=LGREY)
        ax.spines["polar"].set_color(BORDER)
        ax.yaxis.grid(True, color=BORDER, linewidth=0.5)
        ax.xaxis.grid(True, color=BORDER, linewidth=0.5)

        ax.set_title(f"{LANG_LABELS[lang]} ({lang})\nReliability: {c['reliability_score']:.3f}",
                     fontsize=10, fontweight="bold", color=col, pad=14)

    fig.tight_layout()
    save(fig, "radar_per_language.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 4 — Metric summary table as a visual heatmap-style card
# ─────────────────────────────────────────────────────────────────────────────
def chart_summary_table():
    fig, ax = plt.subplots(figsize=(10, 3.8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    col_labels = ["Language", "Accuracy", "Pos-Bias Δ", "Verbosity\nSusc.", "Order\nConsist.", "Reliability\nScore"]
    rows = []
    for lang in LANGS:
        c = CARDS[lang]
        rows.append([
            f"{LANG_LABELS[lang]} ({lang})",
            f"{c['accuracy']:.3f}",
            f"{c['position_bias_delta']:+.3f}",
            f"{c['verbosity_susceptibility']:.3f}",
            f"{c['order_consistency']:.3f}",
            f"{c['reliability_score']:.3f}",
        ])

    n_rows = len(rows)
    n_cols = len(col_labels)
    col_w = [0.20, 0.14, 0.14, 0.14, 0.14, 0.16]
    row_h = 0.16
    header_h = 0.18
    start_y = 0.88

    # Header
    cx = 0.01
    for j, (lbl, cw) in enumerate(zip(col_labels, col_w)):
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx, start_y), cw - 0.01, header_h,
            boxstyle="round,pad=0.005", facecolor=BORDER, edgecolor=BORDER))
        ax.text(cx + (cw - 0.01) / 2, start_y + header_h / 2, lbl,
                ha="center", va="center", fontsize=9.5, fontweight="bold",
                color=WHITE, transform=ax.transAxes,
                multialignment="center")
        cx += cw

    # Data rows
    def cell_color(key, val_str, lang):
        val = float(val_str.replace("+", ""))
        if key == "reliability":
            return (GREEN if val >= 0.70 else YELLOW if val >= 0.60 else RED)
        if key == "accuracy":
            return (GREEN if val >= 0.80 else YELLOW if val >= 0.70 else RED)
        if key == "posbias":
            return (GREEN if abs(val) < 0.05 else YELLOW if abs(val) < 0.12 else RED)
        if key == "verbosity":
            return (GREEN if val < 0.15 else YELLOW if val < 0.25 else RED)
        if key == "consist":
            return (GREEN if val >= 0.65 else YELLOW if val >= 0.55 else RED)
        return WHITE

    keys = [None, "accuracy", "posbias", "verbosity", "consist", "reliability"]

    for i, (row, lang) in enumerate(zip(rows, LANGS)):
        ry = start_y - (i + 1) * (row_h + 0.01)
        bg = CARD if i % 2 == 0 else "#1A2030"
        cx = 0.01
        for j, (cell, cw, key) in enumerate(zip(row, col_w, keys)):
            ax.add_patch(mpatches.FancyBboxPatch(
                (cx, ry), cw - 0.01, row_h,
                boxstyle="round,pad=0.003", facecolor=bg, edgecolor=BORDER,
                linewidth=0.5, transform=ax.transAxes))
            color = WHITE if j == 0 else cell_color(key, cell, lang)
            fw = "bold" if j == 0 or j == n_cols - 1 else "normal"
            ax.text(cx + (cw - 0.01) / 2, ry + row_h / 2, cell,
                    ha="center", va="center", fontsize=10,
                    color=color, fontweight=fw, transform=ax.transAxes)
            cx += cw

    # Legend (use Rectangle, not Patch, to avoid abstract base error)
    legend_patches = [
        mpatches.Rectangle((0, 0), 1, 1, facecolor=GREEN,  alpha=0.85, label="Good"),
        mpatches.Rectangle((0, 0), 1, 1, facecolor=YELLOW, alpha=0.85, label="Marginal"),
        mpatches.Rectangle((0, 0), 1, 1, facecolor=RED,    alpha=0.85, label="Needs attention"),
    ]
    ax.legend(handles=legend_patches, frameon=False, labelcolor=GREY, fontsize=8.5,
              loc="lower right", bbox_to_anchor=(1.0, -0.02))

    ax.set_title("Benchmark Results — Qwen2.5-7B-Instruct-4bit  |  Aya dataset  |  2026-06-14",
                 fontsize=11, fontweight="bold", color=WHITE, pad=12, loc="left")

    fig.tight_layout()
    save(fig, "results_table.png")


if __name__ == "__main__":
    print("Generating benchmark charts for GitHub...")
    chart_reliability_vs_accuracy()
    chart_bias_breakdown()
    chart_radar()
    chart_summary_table()
    print(f"Done — 4 charts in ./{OUT}/")
