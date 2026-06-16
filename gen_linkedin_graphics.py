"""Generate clean LinkedIn graphics for BabelJudge."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
import numpy as np

OUT = "linkedin_assets"
os.makedirs(OUT, exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────────────
BG      = "#0D1117"
CARD    = "#161B22"
BORDER  = "#21262D"
ACCENT  = "#58A6FF"
GREEN   = "#3FB950"
ORANGE  = "#F0883E"
RED     = "#F85149"
YELLOW  = "#E3B341"
WHITE   = "#E6EDF3"
GREY    = "#8B949E"
PURPLE  = "#BC8CFF"

W, H = 12.0, 6.27   # figure inches at 150 dpi → 1800×940 px


def new_fig():
    fig = plt.figure(figsize=(W, H), facecolor=BG)
    return fig


def card_rect(ax, x, y, w, h, color=CARD, edge=BORDER, lw=1.2, radius=0.3):
    """Draw a rounded card using pixel-like data coords."""
    style = f"round,pad={radius}"
    rect = FancyBboxPatch((x, y), w, h, boxstyle=style,
                          facecolor=color, edgecolor=edge, linewidth=lw)
    ax.add_patch(rect)


def accent_bar(ax, x, y, h, color=ACCENT, w=5):
    """Thin vertical colored accent strip on the left of a card."""
    rect = patches.Rectangle((x, y), w, h, facecolor=color, linewidth=0)
    ax.add_patch(rect)


def save(fig, name):
    path = f"{OUT}/{name}"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 1 — HERO
# ─────────────────────────────────────────────────────────────────────────────
def slide1():
    fig = new_fig()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1200); ax.set_ylim(0, 627); ax.axis("off")

    # ── Left column ──
    # Left accent line
    ax.add_patch(patches.Rectangle((44, 100), 5, 420, facecolor=ACCENT, lw=0))

    ax.text(72, 500, "BabelJudge",
            fontsize=70, fontweight="bold", color=WHITE, va="top")

    ax.text(72, 408, "Auditing the Auditors",
            fontsize=24, color=ACCENT, va="top", fontstyle="italic")

    ax.text(72, 358,
            "A multilingual reliability benchmark for LLM-as-a-judge.\n"
            "Measures position bias, verbosity bias, and order-consistency\n"
            "across 65+ languages — no human annotation required.",
            fontsize=14.5, color=GREY, va="top", linespacing=1.75)

    ax.text(72, 118,
            "Open Source  ·  Apache-2.0  ·  github.com/Shreyaskc/BabelJudge",
            fontsize=12, color=GREY, va="bottom")

    # ── Right column — 3 stat cards ──
    stats = [
        ("0.665", "Overall Reliability", "Qwen2.5-7B, 4 languages", ACCENT),
        ("800",   "Judge Calls",         "per benchmark run",       GREEN),
        ("65+",   "Languages",           "supported via Aya",       PURPLE),
    ]
    cx, cw, ch, gap = 700, 460, 108, 18
    for i, (val, label, sub, col) in enumerate(stats):
        cy = 390 - i * (ch + gap)
        card_rect(ax, cx, cy, cw, ch, edge=col, lw=1.5)
        accent_bar(ax, cx, cy, ch, color=col, w=5)
        ax.text(cx + 24, cy + ch * 0.68, val,
                fontsize=34, fontweight="bold", color=col, va="center")
        ax.text(cx + 24, cy + ch * 0.34, label,
                fontsize=13, fontweight="bold", color=WHITE, va="center")
        ax.text(cx + 24, cy + ch * 0.12, sub,
                fontsize=10.5, color=GREY, va="center")

    # OPEN SOURCE pill top-right
    card_rect(ax, 998, 540, 178, 46, color=BG, edge=GREEN, lw=1.2, radius=0.5)
    ax.text(1087, 563, "OPEN SOURCE", fontsize=11, fontweight="bold",
            color=GREEN, ha="center", va="center")

    save(fig, "01_hero.png")


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 2 — THE PROBLEM
# ─────────────────────────────────────────────────────────────────────────────
def slide2():
    fig = new_fig()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1200); ax.set_ylim(0, 627); ax.axis("off")

    ax.text(600, 598, "Raw Accuracy Hides What Really Matters",
            fontsize=28, fontweight="bold", color=WHITE, ha="center", va="top")
    ax.text(600, 562, "Three hidden failure modes that silently break your eval pipeline",
            fontsize=14, color=GREY, ha="center", va="top")

    cards = [
        ("Position Bias",       "A/B",
         "The judge favors whichever\nresponse appears in slot A,\nirrespective of quality.",
         "+0.07Δ (en)   −0.08Δ (sw)", ORANGE),
        ("Verbosity Bias",      "LEN",
         "The judge picks longer text\neven when it is just padded\nrepetition — zero new info.",
         "0.25 susceptibility\n(Hindi)", YELLOW),
        ("Order Inconsistency", "FLIP",
         "Swap A and B: the judge\nflips its verdict. In Swahili\nthis happens 52% of the time.",
         "0.48 order-consistency\n(Swahili)", RED),
    ]

    M, GAP = 30, 20
    cw = (1200 - 2 * M - 2 * GAP) // 3   # 360
    ch = 460
    cy0 = 65   # card bottom

    for i, (title, badge, body, stat, col) in enumerate(cards):
        cx = M + i * (cw + GAP)
        card_rect(ax, cx, cy0, cw, ch, edge=col, lw=1.8)

        # Coloured top band
        ax.add_patch(patches.Rectangle((cx, cy0 + ch - 58), cw, 58,
                                       facecolor=col, alpha=0.12, lw=0,
                                       clip_on=True))
        # Badge chip
        card_rect(ax, cx + 12, cy0 + ch - 50, 56, 36,
                  color=col, edge=col, lw=0, radius=0.3)
        ax.text(cx + 40, cy0 + ch - 32, badge, fontsize=10,
                fontweight="bold", color=BG, ha="center", va="center")

        # Title
        ax.text(cx + 78, cy0 + ch - 32, title,
                fontsize=15, fontweight="bold", color=col, va="center")

        # Body
        ax.text(cx + cw // 2, cy0 + ch - 165, body,
                fontsize=13, color=GREY, ha="center", va="center", linespacing=1.65)

        # Divider
        ax.plot([cx + 24, cx + cw - 24], [cy0 + 115, cy0 + 115],
                color=BORDER, lw=1)

        # Stat
        ax.text(cx + cw // 2, cy0 + 75, stat,
                fontsize=13, fontweight="bold", color=col,
                ha="center", va="center", linespacing=1.5)

    ax.text(600, 32, "BabelJudge measures all three automatically, without human labels.",
            fontsize=13, color=ACCENT, ha="center", va="center", fontstyle="italic")

    save(fig, "02_the_problem.png")


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 3 — HOW IT WORKS
# ─────────────────────────────────────────────────────────────────────────────
def slide3():
    fig = new_fig()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1200); ax.set_ylim(0, 627); ax.axis("off")

    ax.text(600, 598, "How BabelJudge Creates Gold Labels — For Free",
            fontsize=26, fontweight="bold", color=WHITE, ha="center", va="top")

    # ── 4-box linear pipeline ──
    # Box dims
    BOX_H = 390
    BOX_Y = 90   # bottom y of all boxes

    # Box 1: Reference (x=28, w=155)
    b1x, b1w = 28, 155
    card_rect(ax, b1x, BOX_Y, b1w, BOX_H, edge=GREEN, lw=1.8)
    accent_bar(ax, b1x, BOX_Y, BOX_H, color=GREEN)
    ax.text(b1x + b1w // 2, BOX_Y + BOX_H - 34, "REFERENCE",
            fontsize=9, fontweight="bold", color=GREEN, ha="center", va="center")
    ax.text(b1x + b1w // 2, BOX_Y + BOX_H // 2 + 20,
            "High-quality\nhuman-written\nresponse",
            fontsize=12, color=WHITE, ha="center", va="center", linespacing=1.6)
    ax.text(b1x + b1w // 2, BOX_Y + 30, "GOLD LABEL\nby construction",
            fontsize=9, color=GREEN, ha="center", va="center",
            linespacing=1.4, fontstyle="italic")

    # Arrow 1→2
    arr_y = BOX_Y + BOX_H // 2
    ax.annotate("", xy=(220, arr_y), xytext=(185, arr_y),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.8))

    # Box 2: Perturbation engine (x=220, w=310)
    b2x, b2w = 220, 320
    card_rect(ax, b2x, BOX_Y, b2w, BOX_H, edge=ACCENT, lw=1.8)
    ax.text(b2x + b2w // 2, BOX_Y + BOX_H - 28, "PERTURBATION ENGINE",
            fontsize=9.5, fontweight="bold", color=ACCENT, ha="center", va="center")
    # Divider under title
    ax.plot([b2x + 16, b2x + b2w - 16], [BOX_Y + BOX_H - 48, BOX_Y + BOX_H - 48],
            color=BORDER, lw=1)
    perturbs = [
        ("truncate",        "Information loss",    ACCENT),
        ("shuffle",         "Coherence loss",      PURPLE),
        ("number_corrupt",  "Factual errors",      ORANGE),
        ("drop_entities",   "Missing specifics",   YELLOW),
        ("repeat_pad",      "Longer but worse  !",  RED),
    ]
    for j, (name, desc, col) in enumerate(perturbs):
        py = BOX_Y + BOX_H - 80 - j * 64
        ax.text(b2x + 22, py, name, fontsize=11, fontweight="bold",
                color=col, va="center")
        ax.text(b2x + 22, py - 18, desc, fontsize=9.5, color=GREY, va="center")
        if j < 4:
            ax.plot([b2x + 16, b2x + b2w - 16], [py - 30, py - 30],
                    color=BORDER, lw=0.6, alpha=0.7)

    # Arrow 2→3
    ax.annotate("", xy=(576, arr_y), xytext=(542, arr_y),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.8))

    # Box 3: Judge (x=576, w=200)
    b3x, b3w = 576, 200
    card_rect(ax, b3x, BOX_Y, b3w, BOX_H, edge=YELLOW, lw=1.8)
    ax.text(b3x + b3w // 2, BOX_Y + BOX_H - 28, "JUDGE MODEL",
            fontsize=9.5, fontweight="bold", color=YELLOW, ha="center", va="center")
    ax.plot([b3x + 16, b3x + b3w - 16], [BOX_Y + BOX_H - 48, BOX_Y + BOX_H - 48],
            color=BORDER, lw=1)
    ax.text(b3x + b3w // 2, BOX_Y + BOX_H // 2 + 10,
            "Both\npresentation\norders",
            fontsize=13, color=WHITE, ha="center", va="center", linespacing=1.6)
    ax.text(b3x + b3w // 2, BOX_Y + 60,
            "A → B\nB → A",
            fontsize=14, color=YELLOW, ha="center", va="center",
            linespacing=1.6, fontweight="bold")

    # Arrow 3→4
    ax.annotate("", xy=(812, arr_y), xytext=(778, arr_y),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.8))

    # Box 4: Metrics (x=812, w=360)
    b4x, b4w = 812, 360
    card_rect(ax, b4x, BOX_Y, b4w, BOX_H, edge=GREEN, lw=1.8)
    ax.text(b4x + b4w // 2, BOX_Y + BOX_H - 28, "RELIABILITY CARD",
            fontsize=9.5, fontweight="bold", color=GREEN, ha="center", va="center")
    ax.plot([b4x + 16, b4x + b4w - 16], [BOX_Y + BOX_H - 48, BOX_Y + BOX_H - 48],
            color=BORDER, lw=1)
    metrics = [
        ("Accuracy",               "0.770", GREEN),
        ("Position Bias Δ",   "−0.04", ORANGE),
        ("Verbosity Susc.",        "0.175", YELLOW),
        ("Order Consistency",      "0.640", PURPLE),
        ("Reliability Score",      "0.695", ACCENT),
    ]
    for j, (name, val, col) in enumerate(metrics):
        my = BOX_Y + BOX_H - 80 - j * 64
        ax.text(b4x + 22, my, name, fontsize=11, color=GREY, va="center")
        ax.text(b4x + b4w - 22, my, val, fontsize=13, fontweight="bold",
                color=col, va="center", ha="right")
        if j < 4:
            ax.plot([b4x + 16, b4x + b4w - 16], [my - 30, my - 30],
                    color=BORDER, lw=0.6, alpha=0.7)

    ax.text(600, 44, "No human annotation needed — gold labels come free from controlled degradation.",
            fontsize=12.5, color=GREY, ha="center", va="center", fontstyle="italic")

    save(fig, "03_how_it_works.png")


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 4 — REAL RESULTS  (pure gridspec, no canvas overlay)
# ─────────────────────────────────────────────────────────────────────────────
def slide4():
    fig = plt.figure(figsize=(W, H), facecolor=BG)

    # Title row + two-column body
    outer = gridspec.GridSpec(2, 1, figure=fig,
                              left=0.03, right=0.97,
                              top=0.96, bottom=0.04,
                              hspace=0.08,
                              height_ratios=[0.14, 0.86])

    ax_title = fig.add_subplot(outer[0])
    ax_title.set_facecolor(BG)
    ax_title.axis("off")
    ax_title.text(0.5, 0.80,
                  "First Real Run: Qwen2.5-7B · Aya Dataset · Apple Silicon",
                  transform=ax_title.transAxes,
                  fontsize=21, fontweight="bold", color=WHITE,
                  ha="center", va="center")
    ax_title.text(0.5, 0.18,
                  "en  ·  hi  ·  ar  ·  sw     |     10 refs / lang     |     800 judge calls",
                  transform=ax_title.transAxes,
                  fontsize=13, color=GREY, ha="center", va="center")

    # Body: bar chart left, insight right
    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1],
                                             width_ratios=[1.05, 0.95],
                                             wspace=0.06)
    ax_bar  = fig.add_subplot(inner[0])
    ax_info = fig.add_subplot(inner[1])

    # ── Bar chart ──
    ax_bar.set_facecolor(BG)
    for spine in ax_bar.spines.values():
        spine.set_color(BORDER)
    ax_bar.tick_params(colors=GREY)

    langs    = ["Hindi\n(hi)", "English\n(en)", "Arabic\n(ar)", "Swahili\n(sw)"]
    accuracy = [0.835, 0.815, 0.770, 0.660]
    reliab   = [0.714, 0.700, 0.695, 0.550]
    x = np.arange(len(langs))
    w = 0.36

    ax_bar.bar(x - w / 2, accuracy, w, label="Raw Accuracy",
               color=GREY, alpha=0.45, zorder=3)
    bar_colors = [GREEN, GREEN, ORANGE, RED]
    bars = ax_bar.bar(x + w / 2, reliab, w, label="Reliability Score",
                      color=bar_colors, alpha=0.90, zorder=3)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(langs, color=WHITE, fontsize=12)
    ax_bar.set_ylim(0, 1.0)
    ax_bar.set_yticks([0, 0.25, 0.50, 0.65, 0.75, 1.0])
    ax_bar.tick_params(axis="y", labelcolor=GREY, labelsize=10)
    ax_bar.axhline(0.65, color=ORANGE, lw=1.2, ls="--", alpha=0.7, zorder=2)
    ax_bar.text(3.5, 0.655, "min threshold", fontsize=8.5, color=ORANGE,
                va="bottom", ha="right")
    ax_bar.yaxis.grid(True, color=BORDER, lw=0.8, zorder=0)
    ax_bar.set_axisbelow(True)
    ax_bar.set_facecolor(BG)

    for bar, val in zip(bars, reliab):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
                    f"{val:.3f}", ha="center", va="bottom",
                    color=WHITE, fontsize=10, fontweight="bold")

    ax_bar.legend(frameon=False, labelcolor=GREY, fontsize=10,
                  loc="upper right")

    # ── Info boxes (right panel, using data coords 0-1) ──
    ax_info.set_facecolor(BG)
    ax_info.axis("off")
    ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1)

    insights = [
        ("Hindi & English",
         "Reliability 0.71 / 0.70\nAcceptable for automated eval",
         GREEN),
        ("Arabic",
         "Reliability 0.695 — marginal\nApply both-order protocol",
         ORANGE),
        ("Swahili",
         "Reliability 0.55 — order-consistency\n0.48: do NOT rely on this judge",
         RED),
    ]
    box_h = 0.24
    gap   = 0.04
    total = len(insights) * box_h + (len(insights) - 1) * gap
    start_y = (1 - total) / 2

    for i, (title, body, col) in enumerate(insights):
        by = start_y + (len(insights) - 1 - i) * (box_h + gap)
        # card background
        ax_info.add_patch(FancyBboxPatch((0.02, by), 0.96, box_h,
                                         boxstyle="round,pad=0.01",
                                         facecolor=CARD, edgecolor=col,
                                         linewidth=1.5))
        # colour left bar (use rectangle in data coords)
        ax_info.add_patch(patches.Rectangle((0.02, by), 0.025, box_h,
                                             facecolor=col, lw=0))
        ax_info.text(0.09, by + box_h * 0.70, title,
                     fontsize=13, fontweight="bold", color=col,
                     va="center", transform=ax_info.transData)
        ax_info.text(0.09, by + box_h * 0.28, body,
                     fontsize=10.5, color=GREY, va="center", linespacing=1.5,
                     transform=ax_info.transData)

    save(fig, "04_results.png")


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 5 — WHO BENEFITS
# ─────────────────────────────────────────────────────────────────────────────
def slide5():
    fig = new_fig()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1200); ax.set_ylim(0, 627); ax.axis("off")

    ax.text(600, 603, "Who Benefits — and What You Can Do",
            fontsize=28, fontweight="bold", color=WHITE, ha="center", va="top")
    ax.text(600, 567, "Six teams, six immediate actions",
            fontsize=14, color=GREY, ha="center", va="top")

    # 3×2 grid of cards
    tiles = [
        ("Eval Engineers",
         "Know which languages your\nautomated eval cannot be\ntrusted for.",
         ACCENT),
        ("RLHF / DPO Teams",
         "Measure label noise from\nyour judge before it corrupts\ntraining data.",
         PURPLE),
        ("Safety Teams",
         "Find the language coverage gaps\nin automated content screening\nbefore users encounter them.",
         RED),
        ("Product Teams",
         "Route low-resource languages\nto a better judge or human\nreview pipelines.",
         GREEN),
        ("ML Researchers",
         "Qualify benchmark results\nwith the judge's known\nreliability score.",
         ORANGE),
        ("Judge Developers",
         "Pinpoint which language and\nbias type to fix in your\nnext fine-tune.",
         YELLOW),
    ]

    M, GAP = 28, 14
    COLS = 3
    cw = (1200 - 2 * M - (COLS - 1) * GAP) // COLS  # 372
    ROWS = 2
    # cards occupy y=68 to y=530 (462px)
    CARD_AREA_H = 462
    CARD_AREA_Y = 68
    ch = (CARD_AREA_H - (ROWS - 1) * GAP) // ROWS   # 224

    for i, (title, body, col) in enumerate(tiles):
        row, col_i = divmod(i, COLS)
        cx = M + col_i * (cw + GAP)
        cy = CARD_AREA_Y + (ROWS - 1 - row) * (ch + GAP)

        card_rect(ax, cx, cy, cw, ch, edge=col, lw=1.6)
        # Left accent strip
        ax.add_patch(patches.Rectangle((cx, cy), 5, ch, facecolor=col, lw=0))

        ax.text(cx + 22, cy + ch - 34, title,
                fontsize=13.5, fontweight="bold", color=col, va="center")
        ax.plot([cx + 18, cx + cw - 18], [cy + ch - 54, cy + ch - 54],
                color=BORDER, lw=1)
        ax.text(cx + 22, cy + ch // 2 - 10, body,
                fontsize=11.5, color=GREY, va="center", linespacing=1.65)

    # CTA bar
    card_rect(ax, 28, 4, 1144, 52, color=BG, edge=ACCENT, lw=1.2, radius=0.4)
    ax.text(600, 30, "Star it  ·  Try it  ·  Contribute      "
            "github.com/Shreyaskc/BabelJudge",
            fontsize=14, color=ACCENT, ha="center", va="center", fontweight="bold")

    save(fig, "05_who_benefits.png")


if __name__ == "__main__":
    print("Generating LinkedIn graphics...")
    slide1()
    slide2()
    slide3()
    slide4()
    slide5()
    print("Done — 5 images in ./linkedin_assets/")
