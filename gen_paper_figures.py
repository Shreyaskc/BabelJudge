"""Generate paper-quality figures for the arXiv submission.

Outputs to paper/figures/ — white backgrounds, academic palette, 300 DPI.
Run from repo root:  python gen_paper_figures.py
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np

OUT = "paper/figures"
os.makedirs(OUT, exist_ok=True)

# ── Academic colour palette ───────────────────────────────────────────────────
BLACK  = "#1A1A1A"
DKGREY = "#444444"
GREY   = "#666666"
LGREY  = "#AAAAAA"
VLTGRY = "#F5F5F5"
WHITE  = "#FFFFFF"
BORDER = "#CCCCCC"

BLUE   = "#2166AC"
RED    = "#D6604D"
GREEN  = "#4DAC26"
ORANGE = "#E08214"
PURPLE = "#762A83"

LANG_COLORS = {"en": BLUE, "hi": GREEN, "ar": ORANGE, "sw": RED}
LANG_LABELS = {"en": "English", "hi": "Hindi", "ar": "Arabic", "sw": "Swahili"}
LANGS = ["en", "hi", "ar", "sw"]

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.labelsize":   10,
    "axes.titlesize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "text.color":       BLACK,
    "axes.labelcolor":  DKGREY,
    "xtick.color":      DKGREY,
    "ytick.color":      DKGREY,
    "axes.edgecolor":   BORDER,
    "axes.facecolor":   WHITE,
    "figure.facecolor": WHITE,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "grid.color":       "#E8E8E8",
    "grid.linewidth":   0.7,
    "axes.grid":        True,
    "axes.grid.axis":   "y",
    "legend.frameon":   False,
    "legend.fontsize":  9,
})

DATA  = json.load(open("mac_results.json"))
CARDS = {c["language"]: c for c in DATA["cards"]}


def save(fig, name):
    path = f"{OUT}/{name}"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"  saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Methodology pipeline  (orthogonal arrows, no crossing text)
# ─────────────────────────────────────────────────────────────────────────────
def fig_methodology():
    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.0)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    def rbox(ax, cx, cy, w, h, title, subtitle, fc, ec, fs=9.5):
        """Draw a rounded box with title + italic subtitle inside."""
        rect = FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle="round,pad=0.12",
            facecolor=fc, edgecolor=ec, linewidth=1.8,
        )
        ax.add_patch(rect)
        if subtitle:
            ax.text(cx, cy + 0.18, title, ha="center", va="center",
                    fontsize=fs, fontweight="bold", color=ec)
            ax.text(cx, cy - 0.22, subtitle, ha="center", va="center",
                    fontsize=7.5, color=GREY, style="italic")
        else:
            ax.text(cx, cy, title, ha="center", va="center",
                    fontsize=fs, fontweight="bold", color=ec)

    def harrow(ax, x0, x1, y, color=DKGREY):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=1.6, mutation_scale=14))

    def varrow(ax, x, y0, y1, color=DKGREY):
        ax.annotate("", xy=(x, y1), xytext=(x, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=1.6, mutation_scale=14))

    # ── Row 1: corpus → perturbation → gold item ─────────────────────────────
    Y1 = 3.9
    BH = 1.0   # box height
    BW = 2.3   # box width

    rbox(ax, 1.5,  Y1, BW, BH,
         "Multilingual Corpus", "Aya · FLORES-200\n65+ languages",
         "#EDE7F6", PURPLE)

    harrow(ax, 2.65, 3.45, Y1, DKGREY)

    rbox(ax, 4.55, Y1, BW+0.4, BH,
         "Perturbation Engine", "5 types × 2 severities\ntruncate · shuffle · number_corrupt · …",
         "#E3F2FD", BLUE)

    harrow(ax, 5.80, 6.60, Y1, DKGREY)

    rbox(ax, 7.75, Y1, BW, BH,
         "Gold-Labeled Item", "gold = reference\n(by construction — no annotation)",
         "#E8F5E9", GREEN)

    # ── Vertical fork down from gold item ────────────────────────────────────
    FORK_Y  = Y1 - BH/2        # bottom of gold item box
    MERGE_Y = 1.90             # where the two order arms meet the judge box top

    # vertical line down from gold item to fork level
    varrow(ax, 7.75, FORK_Y, 2.80, DKGREY)

    # horizontal line at fork level to two arms
    ax.plot([4.55, 7.75], [2.80, 2.80], color=DKGREY, lw=1.6)

    # arm arrows pointing down to order boxes
    varrow(ax, 4.55, 2.80, 2.35, DKGREY)
    varrow(ax, 7.75, 2.80, 2.35, DKGREY)

    # fork label — placed to the right of the vertical line, no overlap
    ax.text(8.15, 2.80, "×2 presentation orders\n(position-bias probe)",
            ha="left", va="center", fontsize=8, color=GREY, style="italic")

    # ── Row 2: order A and order B ───────────────────────────────────────────
    Y2 = 1.85
    BW2 = 2.5
    BH2 = 0.75

    rbox(ax, 4.55, Y2, BW2, BH2,
         "Order A", "(ref, pert)", "#FFF3E0", ORANGE, fs=9)
    rbox(ax, 7.75, Y2, BW2, BH2,
         "Order B", "(pert, ref)", "#FFF3E0", ORANGE, fs=9)

    # arms merge down to judge
    JUDGE_X = 6.15
    JOIN_Y  = 1.10

    # vertical lines from order boxes down to join level
    ax.plot([4.55, 4.55], [Y2 - BH2/2, JOIN_Y], color=DKGREY, lw=1.6)
    ax.plot([7.75, 7.75], [Y2 - BH2/2, JOIN_Y], color=DKGREY, lw=1.6)
    # horizontal line at join level
    ax.plot([4.55, 7.75], [JOIN_Y, JOIN_Y], color=DKGREY, lw=1.6)

    varrow(ax, JUDGE_X, JOIN_Y, 0.68, DKGREY)

    # ── Row 3: judge → reliability card ──────────────────────────────────────
    Y3 = 0.42
    rbox(ax, JUDGE_X, Y3, BW, 0.7,
         "LLM Judge", "returns A / B / tie", "#E3F2FD", BLUE)

    harrow(ax, JUDGE_X + BW/2 + 0.05, JUDGE_X + BW/2 + 1.0, Y3, DKGREY)

    rbox(ax, 10.2, Y3, 2.6, 0.7,
         "Reliability Card",
         "accuracy · pos-bias Δ · verbosity · consistency · R",
         "#E8F5E9", GREEN)

    fig.tight_layout(pad=0.4)
    save(fig, "methodology.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Reliability vs Raw Accuracy
# ─────────────────────────────────────────────────────────────────────────────
def fig_reliability_vs_accuracy():
    fig, ax = plt.subplots(figsize=(9, 5.0))

    accuracy = [CARDS[l]["accuracy"]          for l in LANGS]
    reliab   = [CARDS[l]["reliability_score"] for l in LANGS]
    colors   = [LANG_COLORS[l]                for l in LANGS]
    xlabels  = [f"{LANG_LABELS[l]}\n({l})"   for l in LANGS]

    x   = np.arange(len(LANGS))
    w   = 0.28   # bar width — narrower than before
    gap = 0.08   # explicit gap between the two bars in each group

    # Accuracy bars: centred to the LEFT of x
    acc_cx   = x - (w / 2 + gap / 2)
    acc_bars = ax.bar(acc_cx, accuracy, w,
                      color=LGREY, alpha=0.70, label="Raw Accuracy",
                      zorder=3, edgecolor="#999999", linewidth=0.8)

    # Reliability bars: centred to the RIGHT of x
    rel_cx   = x + (w / 2 + gap / 2)
    rel_bars = ax.bar(rel_cx, reliab, w,
                      color=colors, alpha=0.90, label="Reliability Score",
                      zorder=3, edgecolor=colors, linewidth=0.9)

    # ── Value labels above each bar ──────────────────────────────────────────
    VPAD = 0.022   # vertical gap between bar top and label

    for cx, val in zip(acc_cx, accuracy):
        ax.text(cx, val + VPAD, f"{val:.2f}",
                ha="center", va="bottom", fontsize=8.5, color=DKGREY)

    for cx, val, col in zip(rel_cx, reliab, colors):
        ax.text(cx, val + VPAD, f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=col)

    # ── Threshold line ───────────────────────────────────────────────────────
    ax.axhline(0.65, color=RED, lw=1.3, ls="--", alpha=0.75, zorder=2)

    # Label anchored to the LEFT y-axis edge using a blended transform
    # (x in axes coords 0–1, y in data coords) — keeps it away from all bars
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(0.01, 0.638, "R = 0.65 — deployment threshold",
            transform=trans, ha="left", va="top",
            fontsize=8, color=RED, style="italic")

    # ── Axes formatting ──────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=10.5)
    ax.set_ylim(0, 1.07)
    ax.set_xlim(-0.55, len(LANGS) - 0.45)
    ax.set_ylabel("Score", fontsize=10, labelpad=8)
    ax.set_title("Reliability Score vs Raw Accuracy — Qwen2.5-7B-Instruct-4bit",
                 fontsize=11, fontweight="bold", color=BLACK, pad=12)

    legend_patches = [
        mpatches.Patch(color=LGREY, alpha=0.75, label="Raw Accuracy"),
        mpatches.Patch(color=BLUE,  alpha=0.90, label="Reliability Score"),
    ]
    ax.legend(handles=legend_patches, loc="upper right",
              fontsize=9.5, framealpha=0, borderpad=0)

    fig.tight_layout(pad=1.2)
    save(fig, "reliability_vs_accuracy.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Bias breakdown  (no per-panel captions, clean ylim headroom)
# ─────────────────────────────────────────────────────────────────────────────
def fig_bias_breakdown():
    fig = plt.figure(figsize=(12, 6.0))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.60, wspace=0.38,
                           left=0.07, right=0.97,
                           top=0.88,  bottom=0.10)

    x  = np.arange(len(LANGS))
    bw = 0.52

    panels = [
        ("accuracy",                "Accuracy",              None,  BLUE,   (0, 0)),
        ("position_bias_delta",     "Position Bias Δ",  0.0,   ORANGE, (0, 1)),
        ("verbosity_susceptibility","Verbosity Susceptibility", None, PURPLE, (1, 0)),
        ("order_consistency",       "Order Consistency",     0.65,  GREEN,  (1, 1)),
    ]

    xlabels = [f"{LANG_LABELS[l]}\n({l})" for l in LANGS]

    for key, title, hline, base_color, pos in panels:
        ax = fig.add_subplot(gs[pos])
        vals = [CARDS[l][key] for l in LANGS]

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

        # value labels — set ylim first so we know the headroom
        if key == "position_bias_delta":
            yabs = max(abs(v) for v in vals)
            ax.set_ylim(-(yabs + 0.08), yabs + 0.10)
        else:
            ax.set_ylim(0, max(vals) + 0.14)

        for bar, val, col in zip(bars, vals, bar_colors):
            fmt = f"{val:+.3f}" if key == "position_bias_delta" else f"{val:.3f}"
            va  = "bottom" if val >= 0 else "top"
            off = 0.014    if val >= 0 else -0.014
            ax.text(bar.get_x() + bar.get_width()/2, val + off, fmt,
                    ha="center", va=va, fontsize=8.5,
                    fontweight="bold", color=col)

        if hline is not None:
            ax.axhline(hline, color=LGREY, lw=1.1, ls="--", alpha=0.9, zorder=2)
            # label at the right edge of the axis, inside plot area
            xlim = ax.get_xlim()
            ax.text(xlim[1] - 0.03, hline + 0.008,
                    f"  {hline:.2f}", ha="right", va="bottom",
                    fontsize=7.5, color=LGREY)

        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold", color=BLACK, pad=8)
        ax.tick_params(axis="x", length=0)

    fig.suptitle(
        "Per-Language Bias Breakdown — Qwen2.5-7B-Instruct-4bit"
        "  (Aya dataset, 100 items / language)",
        fontsize=11, fontweight="bold", color=BLACK, y=0.96)

    save(fig, "bias_breakdown.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Reliability radar  (extra padding, no clipped labels)
# ─────────────────────────────────────────────────────────────────────────────
def fig_radar():
    metrics = ["Accuracy", "Order\nConsistency", "Pos-Bias\n(inverted)",
               "Verbosity\n(inverted)", "Reliability"]
    N      = len(metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2),
                             subplot_kw=dict(projection="polar"),
                             gridspec_kw=dict(wspace=0.55))
    fig.patch.set_facecolor(WHITE)
    fig.suptitle(
        "Reliability Radar — per language  (Qwen2.5-7B-Instruct-4bit)",
        fontsize=11, fontweight="bold", color=BLACK, y=1.01)

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

        ax.plot(angles, vals, color=col, linewidth=2.2, zorder=3)
        ax.fill(angles, vals, color=col, alpha=0.15, zorder=2)

        ax.set_facecolor(VLTGRY)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=8.5, color=DKGREY,
                           fontweight="normal")

        # push tick labels outward so they don't overlap the filled area
        for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
            deg = np.degrees(angle)
            if 45 < deg < 135:       # top labels — push up
                label.set_verticalalignment("bottom")
            elif 225 < deg < 315:    # bottom labels — push down
                label.set_verticalalignment("top")
            label.set_horizontalalignment("center")

        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"],
                           fontsize=6.5, color=LGREY)
        ax.tick_params(colors=LGREY, pad=3)
        ax.spines["polar"].set_color(BORDER)
        ax.yaxis.grid(True, color=BORDER,   linewidth=0.6)
        ax.xaxis.grid(True, color="#DEDEDE", linewidth=0.6)

        ax.set_title(
            f"{LANG_LABELS[lang]} ({lang})\nR = {c['reliability_score']:.3f}",
            fontsize=10, fontweight="bold", color=col, pad=18)

    fig.tight_layout(pad=1.5)
    save(fig, "radar_per_language.png")


if __name__ == "__main__":
    print("Generating paper figures...")
    fig_methodology()
    fig_reliability_vs_accuracy()
    fig_bias_breakdown()
    fig_radar()
    print(f"Done — 4 figures in ./{OUT}/")
