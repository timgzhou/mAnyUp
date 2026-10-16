"""Paper material, model: the components of the mixed-resolution model and what each one reads
(a block diagram; exp.viz.paper.model shows the same pipeline as data).

    python -m exp.viz.paper.model_diagram [--tile 40091@r0c0]

-> figures/paper/model/model.{png,pdf}

The thumbnails are real: panels of exp.viz.paper.model for that tile
(figures/paper/model/<tile>/materials). The diagram describes the configuration the figures
use: crops enter at block 0 (--inject_layer 0), --readout native_set, a learned month-level
query (--joint) and a learned date-level query (--date_level free --date_selector xattn).

Three rows = three levels of the input, coarse to fine; time and resolution get finer downwards:
    seasons   every ps16 patch of every seasonal composite          (dense)
    months    the picked (month, ps4 cell) crops of monthly means   (sparse)
    dates     the picked (date, ps4 cell) crops of acquisitions     (sparser)
All three go through the SAME patch embedding and ViT blocks, as one token sequence.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path("figures/paper/model")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
INK, MUTED, RULE = "#111111", "#555555", "#c9c9c9"
# one colour per role: data (neutral), shared encoder weights, the two queries, the readout
C_DATA, C_ENC, C_QUERY, C_READ = "#f4f4f2", "#dbe7f5", "#fbe3d6", "#e2f0dc"
E_ENC, E_QUERY, E_READ = "#3b6ea8", "#c8551f", "#3f8a3a"
ROW = {"season": 6.55, "month": 4.15, "date": 1.75}          # row centres (inches from bottom)
W, H = 17.0, 8.6


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tile", default="40091@r0c0")
    a = ap.parse_args()
    mat = OUT / a.tile.replace("@", "_") / "materials"
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W), ax.set_ylim(0, H), ax.axis("off")

    def box(x, y, w, h, title, body="", fc=C_DATA, ec=RULE, lw=1.0, fs=8.5):
        """Rounded box centred at (x, y)."""
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, fc=fc, ec=ec, lw=lw,
                                    boxstyle="round,pad=0,rounding_size=0.08", zorder=2))
        if body:
            ax.text(x, y + h / 2 - 0.13, title, ha="center", va="top", fontsize=fs,
                    weight="bold", color=INK, zorder=3)
            ax.text(x, y - 0.16, body, ha="center", va="center", fontsize=6.9, color=INK,
                    zorder=3, linespacing=1.35)
        else:
            ax.text(x, y, title, ha="center", va="center", fontsize=fs, weight="bold",
                    color=INK, zorder=3)

    def arrow(pts, color=INK, lw=1.1, label=None, at=0.5, dy=0.1, ha="center", ls="-"):
        """Polyline through pts with an arrow head at the end; label near segment 0."""
        xs, ys = zip(*pts)
        ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, zorder=1, solid_capstyle="round")
        ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=1,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                    shrinkA=0, shrinkB=0, mutation_scale=9))
        if label:
            (x0, y0), (x1, y1) = pts[0], pts[1]
            ax.text(x0 + at * (x1 - x0), y0 + at * (y1 - y0) + dy, label, ha=ha, va="bottom",
                    fontsize=6.6, color=color if color != INK else MUTED, linespacing=1.3)

    def thumb(name, x, y, s, edge=RULE, lw=0.6):
        p = mat / f"{name}.png"
        if not p.exists():
            return
        ax.imshow(mpimg.imread(p), extent=(x - s / 2, x + s / 2, y - s / 2, y + s / 2),
                  zorder=3, interpolation="nearest")
        ax.add_patch(plt.Rectangle((x - s / 2, y - s / 2), s, s, fill=False, ec=edge, lw=lw,
                                   zorder=4))

    def first(prefix):
        """A material whose name starts with prefix (date panels carry their date)."""
        hits = sorted(mat.glob(prefix + "*.png"))
        return hits[len(hits) // 2].stem if hits else prefix

    yS, yM, yD = ROW["season"], ROW["month"], ROW["date"]
    fig.text(0.012, 0.965, "Mixed-resolution OlmoEarth: components and what each one reads",
             fontsize=13, weight="bold", color=INK)
    fig.text(0.012, 0.935, "one encoder, one token sequence · coarse tokens everywhere, fine "
             "tokens only where and when a query asks for them", fontsize=9, color=MUTED)

    # ---- column 1: the three input levels (fine -> coarse by averaging)
    x_in = 1.25
    lv = [("season", "Seasonal composites", "4 seasons · 64 × 64 px", ["3_season_S2_2", "3_season_S1A_2"]),
          ("month", "Monthly composites", "12 months", ["2_month_S2_07_May", "2_month_S1A_07_May"]),
          ("date", "Acquisitions", "every date of each\nmodality (own dates)",
           [first("1_acquisition_S2_2"), first("1_acquisition_S1A_3")])]
    for key, title, sub, names in lv:
        y = ROW[key]
        ax.add_patch(FancyBboxPatch((x_in - 1.0, y - 0.78), 2.0, 1.56, fc=C_DATA, ec=RULE, lw=1.0,
                                    boxstyle="round,pad=0,rounding_size=0.08", zorder=2))
        ax.text(x_in, y + 0.66, title, ha="center", va="top", fontsize=8.5, weight="bold", zorder=3)
        for j, n in enumerate(names):
            thumb(n, x_in - 0.42 + j * 0.84, y - 0.02, 0.74)
        ax.text(x_in, y - 0.7, sub.replace("\n", " "), ha="center", va="bottom", fontsize=6.6,
                color=MUTED, zorder=3)
    arrow([(x_in, yD + 0.78), (x_in, yM - 0.78)], MUTED, label="  mean per month", at=0.5, dy=-0.07, ha="left")
    arrow([(x_in, yM + 0.78), (x_in, yS - 0.78)], MUTED, label="  mean of 3 months", at=0.5, dy=-0.07, ha="left")

    # ---- column 2: the two queries (between the rows they connect)
    x_q, qw, qh = 4.35, 2.5, 1.12
    yQm, yQd = (yS + yM) / 2 - 0.02, (yM + yD) / 2 - 0.02
    box(x_q, yQm, qw, qh, "Month-level query",
        "scores every (month, ps4 cell): 12 × 16 × 16\ntop k are cropped   (k = 1/6 of them)\ntrained to predict each crop's loss gain",
        C_QUERY, E_QUERY, 1.3)
    box(x_q, yQd, qw, qh, "Date-level query",
        "scores every (modality, date, ps4 cell)\ncross-attention to the season tokens · top 2k\nits score also gates the crop in the readout",
        C_QUERY, E_QUERY, 1.3)
    thumb("4_month_query_score_07_May", x_q + qw / 2 + 0.45, yQm, 0.62, E_QUERY, 0.9)
    thumb(first("5_date_query_score_S2_2"), x_q + qw / 2 + 0.45, yQd, 0.62, E_QUERY, 0.9)

    # ---- column 3: patch embedding (shared) + encodings
    x_pe, pw, ph = 7.75, 2.3, 1.2
    box(x_pe, yS, pw, ph, "Patch embed · ps 16", "every patch: 4 × 4 per season\n+ position · time slot · month · channel",
        C_ENC, E_ENC, 1.3)
    box(x_pe, yM, pw, ph, "Patch embed · ps 4", "picked crops only (4 × 4 px each)\n+ same encodings at the crop's pixel\nposition and time  + high-res type emb.",
        C_ENC, E_ENC, 1.3)
    box(x_pe, yD, pw, ph, "Patch embed · ps 4", "picked crops only, one modality each\n+ same encodings, time = day inside\nthe season  + high-res + date type emb.",
        C_ENC, E_ENC, 1.3)
    for y0, y1 in ((yS, yM), (yM, yD)):
        ax.text(x_pe, (y0 + y1) / 2, "same weights\n(FlexiViT: one kernel, resized)", ha="center",
                va="center", fontsize=6.4, color=E_ENC, style="italic")

    # ---- column 4: the ViT, one sequence
    x_v, vw = 10.35, 1.75
    ax.add_patch(FancyBboxPatch((x_v - vw / 2, yD - 0.85), vw, yS - yD + 1.7, fc=C_ENC, ec=E_ENC,
                                lw=1.6, boxstyle="round,pad=0,rounding_size=0.1", zorder=2))
    ax.text(x_v, (yS + yD) / 2 + 0.95, "OlmoEarth ViT", ha="center", va="center", fontsize=10,
            weight="bold", zorder=3)
    ax.text(x_v, (yS + yD) / 2 - 0.05,
            "12 blocks, pretrained\nfully fine-tuned\n\nall tokens in ONE\nsequence per sample:\n"
            "joint self-attention\nseason ↔ month ↔ date\n\n(blocks [0, L) per level,\n[L, 12) "
            "jointly; here L = 0)", ha="center", va="center", fontsize=7.2, zorder=3,
            linespacing=1.4)

    # ---- column 5: output tokens
    x_t = 12.45
    toks = [("season", "6_tokens_season_2", "season tokens\nps16 · dense"),
            ("month", "6_tokens_month_07_May", "month tokens\nps4 · sparse"),
            ("date", first("6_tokens_date_S2_2"), "date tokens\nps4 · sparser")]
    for key, n, cap in toks:
        thumb(n, x_t, ROW[key], 0.95, INK, 0.9)
        ax.text(x_t, ROW[key] - 0.54, cap, ha="center", va="top", fontsize=7, color=INK)

    # ---- column 6: the readout
    x_h, hw = 14.1, 1.55
    ax.add_patch(FancyBboxPatch((x_h - hw / 2, yD - 0.85), hw, yS - yD + 1.7, fc=C_READ, ec=E_READ,
                                lw=1.6, boxstyle="round,pad=0,rounding_size=0.1", zorder=2))
    ax.text(x_h, yS + 0.42, "Readout", ha="center", va="center", fontsize=10, weight="bold", zorder=3)
    ax.text(x_h, (yS + yD) / 2 + 0.75,
            "1 · native head\none linear layer on\nevery token:\nD → 19 × 16 × 16\n"
            "ps16 token: its 16×16 px\nps4 token: averaged\nto its 4 × 4 px",
            ha="center", va="center", fontsize=7.1, zorder=3, linespacing=1.4)
    ax.text(x_h, (yS + yD) / 2 - 1.25,
            "2 · temporal pool\nper ps4 cell, over the\ntokens covering it:\n"
            "softmax( score(token)\n+ season prior + gate )\n→ weighted sum of\ntheir logits",
            ha="center", va="center", fontsize=7.1, zorder=3, linespacing=1.4)
    x_o = 16.0
    thumb("7_prediction", x_o, yM + 0.75, 1.1, INK, 0.9)
    ax.text(x_o, yM + 1.36, "prediction\n64 × 64 px", ha="center", va="bottom", fontsize=7.2)
    thumb("7_label", x_o, yM - 0.95, 1.1, RULE, 0.9)
    ax.text(x_o, yM - 1.56, "label", ha="center", va="top", fontsize=7.2)
    arrow([(x_o, yM - 0.4), (x_o, yM + 0.2)], MUTED, ls=(0, (3, 2)))
    ax.text(x_o + 0.08, yM - 0.1, "loss", ha="left", va="center", fontsize=6.8, color=MUTED)

    # ---- arrows: the three token paths
    x_gate = x_q + qw / 2 + 0.45                                   # where a query's picks select the crops
    arrow([(x_in + 1.0, yS), (x_pe - pw / 2, yS)], label="all patches", at=0.82)
    for y, lab in ((yM, "picked (month, cell) crops"), (yD, "picked (date, cell) crops")):
        ax.plot([x_in + 1.0, x_gate - 0.13], [y, y], color=INK, lw=1.1, zorder=1)
        ax.add_patch(plt.Circle((x_gate, y), 0.13, fc="white", ec=E_QUERY, lw=1.4, zorder=3))
        ax.text(x_gate, y, "×", ha="center", va="center", fontsize=9, color=E_QUERY, zorder=4)
        arrow([(x_gate + 0.13, y), (x_pe - pw / 2, y)])
    for y in (yS, yM, yD):
        arrow([(x_pe + pw / 2, y), (x_v - vw / 2, y)])
        arrow([(x_v + vw / 2, y), (x_t - 0.52, y)])
        arrow([(x_t + 0.52, y), (x_h - hw / 2, y)])
    arrow([(x_h + hw / 2, yM + 0.75), (x_o - 0.6, yM + 0.75)])

    # ---- arrows: what the queries read (orange) and what they decide
    Q = E_QUERY
    x_tr = 2.5                                     # trunk carrying the season features down
    ax.plot([x_tr, x_tr], [yS, yQd + 0.25], color=Q, lw=1.1, zorder=1)
    ax.add_patch(plt.Circle((x_tr, yS), 0.045, fc=Q, ec=Q, zorder=3))
    arrow([(x_tr, yQm + 0.25), (x_q - qw / 2, yQm + 0.25)], Q)
    arrow([(x_tr, yQd + 0.25), (x_q - qw / 2, yQd + 0.25)], Q)
    ax.text(x_tr + 0.06, yS - 0.12, "season tokens — month query: after a crop-free pass\nthrough the same ViT; date "
            "query: at the injection depth L", ha="left", va="top", fontsize=6.4, color=Q, linespacing=1.25)
    # each level's own cheap view: the ps4 patch embedding of every cell
    for yq, yrow, lab in ((yQm, yM, "ps4 patch embedding of every\n(month, cell) — no ViT"),
                          (yQd, yD, "ps4 patch embedding of every\n(date, cell) — no ViT")):
        xr = 2.8
        ax.plot([xr, xr], [yrow, yq - 0.25], color=Q, lw=1.1, zorder=1)
        ax.add_patch(plt.Circle((xr, yrow), 0.045, fc=Q, ec=Q, zorder=3))
        arrow([(xr, yq - 0.25), (x_q - qw / 2, yq - 0.25)], Q)
        ax.text(xr + 0.06, yrow + 0.1, lab, ha="left", va="bottom", fontsize=6.4, color=Q,
                linespacing=1.25)
        # the decision: which crops pass
        arrow([(x_q + qw / 2, yq), (x_gate - 0.31, yq)], Q)
        arrow([(x_gate, yq - 0.31), (x_gate, yrow + 0.13)], Q)
        ax.text(x_gate + 0.07, (yq - 0.31 + yrow + 0.13) / 2, "top-k\nmask", ha="left", va="center",
                fontsize=6.4, color=Q, linespacing=1.25)

    # ---- legend of roles
    for j, (fc, ec, name) in enumerate(((C_DATA, RULE, "data"),
                                        (C_ENC, E_ENC, "encoder (shared weights)"),
                                        (C_QUERY, E_QUERY, "queries: where and when to look closer"),
                                        (C_READ, E_READ, "readout"))):
        x = 0.35 + [0, 0.95, 3.0, 6.2][j]
        ax.add_patch(FancyBboxPatch((x, 0.22), 0.26, 0.2, fc=fc, ec=ec, lw=1.0,
                                    boxstyle="round,pad=0,rounding_size=0.04"))
        ax.text(x + 0.34, 0.32, name, va="center", fontsize=7.5, color=INK)
    ax.text(W - 0.3, 0.32, "token counts on a 64 px tile: 4 × 16 season patches · k = 512 of 3,072 "
            "month-cells · 2k = 1,024 date-cells", ha="right", va="center", fontsize=7.2, color=MUTED)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"model.{ext}", dpi=220 if ext == "png" else 300)
    plt.close(fig)
    print("wrote", OUT / "model.png")


if __name__ == "__main__":
    main()
