"""Accuracy vs compute of the PASTIS method across datasets (scripts/slurm/mixres_datasets.sh,
jobs mxds_*): entropy (classification) / season-variance (regression) space with the train-set
threshold + readout-attention time, against grid + grid and random + random at the matched
crop budget and the fixed-resolution models. One panel per dataset, both resolution pairs on one
axis; RMSE axes inverted so up is better everywhere. Two figures:
    figures/datasets_tradeoff_seg[_xhr].png     PASTIS, PLANTEUR, BioMassters, YieldSAT (one map
                                                per time series)
    figures/datasets_tradeoff_change[_xhr].png  GEOID-Flood series, EuroMineNet (change detection;
                                                how to select there is still open)

    python -m exp.viz.datasets_tradeoff [--attn cross_hr] [--x gmacs|mem|speed]

--attn cross_hr: the mixed arms with model.attn cross_hr (crop tokens read the season tokens and
each other, season tokens never read crops; ATTN=cross_hr in the launcher, the PASTIS "ours")

--x mem / speed (--attn cross_hr, the segmentation figure only): the x axis is the inference peak
memory or throughput measured by exp.mixres.bench_speed on one GPU (results/<task>/mixres_speed.csv,
fixed_speed.csv) instead of GMACs -> figures/datasets_tradeoff_seg_xhr_{mem,speed}.png
"""
import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402
import pandas as pd  # noqa: E402

from exp.viz import res_pairs_tradeoff as rp  # noqa: E402
from exp.viz.spacetime_tradeoff import BASE_COL, PAIR, QUAD_COL, XHR, pair_name  # noqa: E402
from manyup.mixres.config import RunConfig, apply_overrides  # noqa: E402

SEG = [   # task, title, y label, lower is better
    ("pastis", "PASTIS (crop types)", "test mIoU", False),
    ("planteur", "PLANTEUR (crop types)", "test mIoU", False),
    ("biomassters", "BioMassters (biomass)", "test RMSE", True),
    ("yieldsat_raw", "YieldSAT (yield)", "test RMSE", True),
]
CHANGE = [
    ("geoidseries_flood", "GEOID-Flood, S1 6-month series", "test flood IoU", False),
    ("euromine", "EuroMineNet (mine mask per year)", "test OF1", False),
]
QS = (0.125, 0.25, 0.5)
# a seasonal model already below the compute-matched monthly ones is no baseline to build on:
# the fixed patch sizes left out per task
DROP_FIXED = {"yieldsat_raw": {4}}


def half_pair(ps, geo):
    """Whether the pair's crop tokens are half the season patch (c8 at ps16, c10p5 at ps10)."""
    c = re.match(r"c(\d+)(?:p(\d+))?", geo)
    return 2 * int(c.group(2) or c.group(1)) == ps


# the pairs drawn: half-patch crops only, no ps4 seasons (S4 + S2 costs too much), in Set2
PAIRS = [k for k in PAIR if half_pair(*k) and k[0] > 4]
PAIR_COL = dict(zip(PAIRS, plt.get_cmap("Set2").colors))
ATTN_TAG = {"joint": "", "cross_hr": "-xhr"}
XT = ""     # the run-name tag of the mixed arms' attention (main --attn)
K = 1.0     # marker / line / font scale of the panels (the paper figure: smaller)
LABEL_PS = None     # True: only each curve's two end points labelled (the paper figure); None: all
JOIN = True         # MiSTR's q_s points joined by a line (the paper figure: lone stars)
LABEL_PT = None     # point label size in pt (the paper figure); None: scaled by K
XM = "gmacs"        # the x axis: GMACs, or a measured column of exp.mixres.bench_speed (main --x)
X_COL = {"mem": "peak_mem_gb", "speed": "samples_per_s"}
X_LABEL = {"gmacs": "GMACs per sample", "mem": "peak memory (GB)",
           "speed": "throughput (samples/s)"}

def rows(task):
    d = pd.read_csv(f"results/{task}/mixres.csv", low_memory=False)
    return d.drop_duplicates("run", keep="last")


def test_cost(r, base):
    """GMACs per sample at the run's TEST crop count. exp.mixres.train traces GMACs on the first
    val batches, where an adaptive (global-threshold) selector crops a different share than over
    the test split (euromine: more, biomassters: fewer); the fixed-n controls crop the same
    everywhere. Cost = the crop-free pass (base: the fixed model at that patch size) + the run's
    own traced cost per crop x its test crops per sample."""
    g, cc, tc = r.gmacs, r.compute_crops_per_sample, r.test_crops_per_sample
    if base is None or not cc or pd.isna(cc) or pd.isna(tc):
        return g
    return base + (g - base) / cc * tc


def measured(task, kind):
    """exp.mixres.bench_speed's rows (kind: mixres / fixed), the last per key, or None."""
    f = Path(f"results/{task}/{kind}_speed.csv")
    if not f.exists():
        return None
    d = pd.read_csv(f)
    if kind == "mixres":       # one seed was measured per configuration: key by configuration
        return d.assign(cfg=d.run.str.replace(r"_seed\d+_\w+$", "", regex=True)
                        ).drop_duplicates("cfg", keep="last").set_index("cfg")
    return d.drop_duplicates(["family", "patch"], keep="last").set_index(["family", "patch"])


def seed_mean(r, base):
    """(x, metric) averaged over the matched runs (one per seed), or None. x: GMACs, or (--x)
    the measured value of the runs' configuration."""
    if not len(r):
        return None
    if XM == "gmacs":
        return (sum(test_cost(x, base) for _, x in r.iterrows()) / len(r), r.test_primary.mean())
    task = re.match(r"^mixres_(\w+?)_ps\d", r.run.iloc[0])[1]
    m = measured(task, "mixres")
    cfg = r.run.str.replace(r"_seed\d+_\w+$", "", regex=True)
    x = [m.at[c, X_COL[XM]] for c in cfg if m is not None and c in m.index]
    return (sum(x) / len(x), r.test_primary.mean()) if x else None


# --attn cross_hr: ours and the controls as ONE run each (train.stage1_*: oracle epochs, then
# the arm; SCHED=1 in the launcher), every seed averaged; euromine (no oracle stage) and the
# joint figure: warm from the oracle / fixed model
def one_run(task):
    return bool(XT) and task != "euromine"


def ours(d, task, ps, keep, base=None):
    """(GMACs, metric) of the global-threshold arm at q_s 1/8, 1/4, 1/2 (mean over seeds)."""
    hs = {"one": "-hhone" if task != "euromine" else "",
          "half": "-hhhalf-hlhalf" if task != "euromine" else None}[keep]
    if hs is None:
        return []
    pts = []
    for q in QS:
        pre = rf"^mixres_{task}_ps{ps}_quadg-skip-qs{q:g}-qt0\.5{hs}-crop-score{XT}"
        warm = pre + r"_n0_c\d+(?:s\d+)?_warm_seed\d+_"
        m = (seed_mean(d[d.run.str.contains(pre + rf"-s1o\d+_n0_c{ps // 2}_seed\d+_")], base)
             if one_run(task) else None)
        # (no one-run yet: the two-step oracle -> warm start, the same arm, just more seed noise)
        m = m or seed_mean(d[d.run.str.contains(warm)], base)
        if m:
            pts.append(m)
    return sorted(pts)


def sched_pair(d, task, ps, geo, base=None):
    """(GMACs, metric) of ours as ONE run (the selector schedule, scripts/slurm/mixres_pairs.sh:
    oracle epochs, then ours with cross attention; no warm start) at season patch ps and crop
    geometry geo (c12: 12 px crops; c10p5: 10 px windows as ps5 crops), mean over seeds."""
    pts = []
    for q in QS:
        pat = (rf"^mixres_{task}_ps{ps}_quadg-skip-qs{q:g}-qt0\.5-hhone-crop-score-xhr-s1o\d+"
               rf"_n0_{geo}_seed\d+_")
        m = seed_mean(d[d.run.str.contains(pat)], base)
        if m:
            pts.append(m)
    return sorted(pts)


def control(d, task, ps, kind, base=None):
    """(GMACs, metric) of grid + grid / random + random, one season, at the matched n (mean
    over seeds)."""
    ns = {32: (2, 4, 8), 16: (8, 16, 32), 8: (32, 64, 128)}[ps]
    if task == "euromine":   # its (year, window) pair controls (scripts/slurm/euromine/mixres.sh)
        tag = {"grid": "season_grid-pairs", "random": "season_random-pairs"}[kind]
    else:
        tag = {"grid": r"grid-top\+none-month_grid\d",
               "random": r"random-top\+season_random-every_location-one"}[kind]
    pts = []
    for n in ns:
        pre = rf"^mixres_{task}_ps{ps}_{tag}{XT}"
        pat = (pre + rf"-s1o\d+_n{n}_c\d+(?:s\d+)?_seed\d+_" if one_run(task)
               else pre + rf"_n{n}_c\d+(?:s\d+)?_warm_seed\d+_")
        m = seed_mean(d[d.run.str.contains(pat)], base)
        if m:
            pts.append(m)
    return sorted(pts)


# fixed-resolution families (as spacetime_baselines): key -> (line, marker, filled, legend label)
FAM = {
    "sea": ("--", "s", False, "fixed resolution, seasonal inputs"),
    "mon": ("-", "D", True, "fixed resolution, monthly inputs"),
    "date": ("-", "D", True, "fixed resolution, per-date inputs (GEOID)"),
    "gmon": ("--", "s", False, "fixed resolution, monthly inputs (GEOID)"),
    "year": ("-", "D", True, "fixed resolution, per-year inputs (EuroMineNet)"),
}
FAM_UNIT = {"sea": "sea", "mon": "mon", "date": "date", "gmon": "mon", "year": "year"}


def fixed(task, d):
    """The fixed-resolution models with the mixed models' input (64 px tiles, the whole series,
    pretrained): seasonal and monthly tokens at every patch size run (PASTIS: also the finer
    sweep) -> [(family, patch, GMACs, metric)], each the mean over its seeds."""
    if task in ("pastis", "planteur", "biomassters", "yieldsat_raw"):
        # seasonal and monthly tokens (--lr_steps all): the legacy fixed runs at every patch
        # size (scripts/slurm/fixed_sweep.sh), the mixed models' attn_temporal_pool head
        # (OlmoEarth's own series fine-tune, oe_monthly, is NOT drawn: a different head)
        out = [r for r in legacy_fixed(task, "sea") + legacy_fixed(task, "mon")
               if r[1] not in DROP_FIXED.get(task, ())]
        # (yieldsat_raw's earlier monthly runs with other heads -- ps8 / 4 bilinear, ps16 native;
        # ps4 1.324 -- are left out: every fixed point has the mixed models' head)
        return out
    out = []
    if task == "euromine":   # every year at one patch size, all years jointly, no crops
        n0 = d[d.run.str.contains(r"^mixres_euromine_ps\d+_season_random-pairs_n0_c8s8_seed\d+_")]
        n0 = n0[n0["model.independent_steps"].astype(str) != "True"]
        for ps, r in n0.groupby(n0["model.lr_patch"].astype(int)):
            out.append(("year", ps, r.gmacs.mean(), r.test_primary.mean()))
    else:   # geoidseries_flood, no crops (quadrant q_s 0): month tokens and every date its own
        # token (task geoidseries_flood_dates); GMACs from the re-measured eval-only run
        # (_warm_frozen) where there is one
        dd = rows(f"{task}_dates") if Path(f"results/{task}_dates/mixres.csv").exists() else None
        for src, t, fam in ((d, task, "gmon"), (dd, f"{task}_dates", "date")):
            for ps in (16, 8, 4) if src is not None else ():
                r = src[src.run.str.contains(rf"^mixres_{t}_ps{ps}_quads-skip-qs0-qt0\.5_n0_")]
                ev = r.run.str.contains("_warm_frozen_")
                if (~ev).any():
                    x = r[ev].gmacs.iloc[-1] if ev.any() else r[~ev].gmacs.mean()
                    out.append((fam, ps, x, r[~ev].test_primary.mean()))
    return out


# GMACs used where a legacy fixed run has no trace in results/<task>/inject_flops.csv yet:
# planteur at PASTIS's traces (same S2 + S1 64 px input), yieldsat monthly as traced by hand
GMAC_FALLBACK = {("pastis", "sea"): {16: 1.73, 8: 10.55, 4: 100.19},   # zero-crop traces
                 # monthly ps16 / 8 / 4 until traced: OlmoEarth series at the same patch
                 ("pastis", "mon"): {16: 6.26, 8: 53.66, 4: 673.23},
                 ("biomassters", "mon"): {16: 5.14, 8: 42.09, 4: 512.86},
                 ("planteur", "sea"): {16: 1.70, 8: 10.41, 4: 100.19},
                 ("planteur", "mon"): {16: 6.26, 8: 53.66, 4: 673.23},
                 ("yieldsat_raw", "mon"): {16: 2.10, 8: 13.60, 4: 137.80},
                 # ps8 / ps4 seasons as rp.fixed has them (zero-crop traces / --flops)
                 ("yieldsat_raw", "sea"): {8: 4.04, 4: 34.33},
                 ("biomassters", "sea"): {8: 10.32, 4: 99.28}}


def oe_monthly(task):
    """OlmoEarth's series fine-tune (every month its own token) at one tile per sample (64 px, or
    padded to whole patches), mean over seeds (_seedS) -> [("mon", patch, GMACs, metric)]."""
    o = pd.read_csv(f"results/{task}/olmoearth.csv", low_memory=False)
    o = o[(o["mode"] == "fft") & (o.model_size == "tiny") & (o.temporal_mode == "series")
          & (o.lr == 1e-3)].drop_duplicates("run", keep="last")
    fl = pd.read_csv(f"results/{task}/flops.csv", usecols=range(6))
    fl = fl[fl.model_size == "tiny"].drop_duplicates("run", keep="last").set_index("run").gmacs
    ep = o.epochs.min()
    out = []
    for ps in rp.PATCHES:
        ts = max(64, -(-64 // ps) * ps)
        r = o[o.run.str.fullmatch(rf"oe_{task}_tiny_series_ps{ps}ts{ts}_fft_ep{ep}(?:_seed\d+)?")]
        if len(r) and f"ps{ps}ts{ts}_ts" in fl.index:
            out.append(("mon", ps, fl[f"ps{ps}ts{ts}_ts"], r.test_primary.mean()))
    return out


def legacy_fixed(task, fam):
    """The exp.upsamplers.finetune_inject fixed runs (--selector none --readout native_set):
    seasonal tokens or every month its own token (--lr_steps all), each patch size the mean over
    its seeds, GMACs from inject_flops.csv (else GMAC_FALLBACK) -> [(family, patch, GMACs,
    metric)]."""
    ep = "ep10" if task == "biomassters" else "ep50"
    f = pd.read_csv(f"results/{task}/inject_fft.csv", low_memory=False)
    f = f.drop_duplicates("run", keep="last")
    fl = pd.read_csv(f"results/{task}/inject_flops.csv", usecols=range(6))
    fl = fl.drop_duplicates("run", keep="last").set_index("run").gmacs
    out = []
    for ps in (16, 14, 12, 10, 8, 6, 4, 2):
        name = (f"inject_fft_{task}_tiny_none_native_set{'_lrall' if fam == 'mon' else ''}"
                f"{'' if ps == 16 else f'_lrps{ps}'}_lr0.001_{ep}")
        r = f[f.run.astype(str).str.fullmatch(re.escape(name) + r"(_seed\d+)?")]
        x = fl.get(name, GMAC_FALLBACK.get((task, fam), {}).get(ps))
        if XM != "gmacs":
            m = measured(task, "fixed")
            x = m.at[(fam, ps), X_COL[XM]] if m is not None and (fam, ps) in m.index else None
        if len(r) and x is not None:
            out.append((fam, ps, x, r.test_primary.mean()))
    return out


# time unit of each token kind per task: (per-step / monthly fixed, seasonal fixed, crops)
T_UNIT = {"euromine": ("year", "year", "year"), "geoidseries_flood": ("date", "mon", "date")}
PAIR_SIDE = {32: "left", 16: "above", 8: "left"}    # where a resolution pair's name sits
SIDE_FIX = {("planteur", 8): "above", ("biomassters", 16): "right",    # (clear of other labels)
            ("euromine", 16): "right"}


def overlap(fx, d, task):
    """The paper figure: the fixed models within a factor 2 of MiSTR's most expensive point, not
    the ones far past it."""
    lr = {ps: x for f, ps, x, _ in fx if f == "sea"}
    p = [t for ps in (16, 8) for t in ours(d, task, ps, "one", lr.get(ps)) or []]
    for ps, geo in PAIRS:
        if (ps, geo) not in ((16, "c8"), (8, "c4")):
            p += sched_pair(d, task, ps, geo, lr.get(ps)) or []
    if XM == "speed":                           # (cost: the lower throughputs)
        low = min(t[0] for t in p) / 2
        return [t for t in fx if t[2] >= low]
    top = 2 * max(t[0] for t in p)
    return [t for t in fx if t[2] <= top]


def panel(ax, task, title, ylabel, lower):
    """One dataset, in the style of spacetime_tradeoff.draw_baselines: the fixed models (monthly
    solid diamonds, seasonal dashed squares), grid / random in gray, ours as a starred curve per
    resolution pair, named beside it."""
    d = rows(task)
    tm, ts, tc = T_UNIT.get(task, ("mon", "sea", "mon"))
    ours_col = XHR[2] if XT else QUAD_COL
    fx = fixed(task, d)
    lr_all = fx
    if LABEL_PS is not None:
        fx = overlap(fx, d, task)
    for fam in FAM:
        pts = sorted((x, y) for f, _, x, y in fx if f == fam)
        if not pts:
            continue
        ls, mk, filled, _ = FAM[fam]
        ax.plot(*zip(*pts), ls + mk, color=rp.INK, lw=1.2 * K, ms=7 * K,
                mfc=rp.INK if filled else rp.SURFACE, mec=rp.INK, mew=1.4 * K, alpha=0.75,
                zorder=3)
    # seasonal-style labels below their point, the rest above (the paper figure: every label at
    # the lower right of its point, clear of MiSTR's curves above the fixed ones)
    for fam, ps, x, y in fx:
        up = FAM[fam][0] != "--"
        if LABEL_PS is not None and ps not in (16, min(p for f, p, *_ in fx if f == fam)):
            continue
        if LABEL_PS is not None and ps < 16 and not up and any(   # (the ends side by side: one)
                f != fam and p == min(q for g, q, *_ in fx if g == f) and 1 / 2.5 < x2 / x < 2.5
                for f, p, x2, _ in fx):
            continue
        xy, ha, va = (((4, -4), "left", "top") if LABEL_PS is not None else
                      ((0, 7 if up else -7), "center", "bottom" if up else "top"))
        if LABEL_PS is not None and XM == "speed" and (up or ps < 16):  # (clear of the stars)
            xy, ha, va = (((0, -8), "center", "top") if up and ps < 16 else
                          ((0, 8), "center", "bottom") if ps < 16 else
                          ((-4, -4), "right", "top"))
        ax.annotate(rf"$S_{{{ps}}}T_{{{FAM_UNIT.get(fam, 'mon')}}}$", (x, y),
                    xytext=(xy[0] * K, xy[1] * K), textcoords="offset points", ha=ha, va=va,
                    fontsize=LABEL_PT or 8.5 * K, zorder=5)
    # the crop-free pass of a mixed model at season patch ps: the fixed model of that patch on
    # the season tokens (euromine: its years jointly; geoid: month tokens)
    lr_fam = {"euromine": "year", "geoidseries_flood": "gmon"}.get(task, "sea")
    lr_pass = {ps: x for f, ps, x, _ in lr_all if f == lr_fam}
    base = {"grid": ("--", "o"), "random": (":", "d")}
    # (euromine: no ps8 + ps4, ps4 already hurts at fixed resolution)
    for ps in (32, 16) if task == "euromine" else (16, 8):
        # (one-way attention: one season per location only, as the PASTIS "ours")
        seg = XT and task not in ("euromine", "geoidseries_flood")
        # (segmentation figure with cross attention: no grid / random controls)
        for kind, (ls, mk) in base.items() if not seg else ():
            p = control(d, task, ps, kind, lr_pass.get(ps))
            if p:
                ax.plot(*zip(*p), ls + mk, color=BASE_COL, lw=1.6, ms=6, alpha=0.55, zorder=4)
        col = PAIR_COL.get((ps, f"c{ps // 2}"), ours_col) if seg else ours_col
        for keep, ls in (("one", "-"),) + ((("half", "--"),) if not XT else ()):
            p = ours(d, task, ps, keep, lr_pass.get(ps))
            if p:
                ax.plot(*zip(*p), (ls if JOIN else "") + "*", color=col, lw=2.4 * K,
                        ms=13 * K, mec=col, mew=1.2 * K, zorder=6)
        p = ours(d, task, ps, "one", lr_pass.get(ps))
        if not p or seg:        # (segmentation figure: the pairs are named in the legend)
            continue
        side = SIDE_FIX.get((task, ps), PAIR_SIDE[ps])
        x, y = p[0] if side == "left" else p[-1]
        off, ha, va = {"left": ((-12, 0), "right", "center"),
                       "right": ((12, 0), "left", "center"),
                       "above": ((0, 12), "center", "bottom")}[side]
        ax.annotate(rf"$S_{{{ps}}}T_{{{ts}}}+S_{{{ps // 2}}}T_{{{tc}}}$", (x, y), xytext=off,
                    textcoords="offset points", ha=ha, va=va, fontsize=10, color=ours_col,
                    zorder=7)
    if XT and task not in ("euromine", "geoidseries_flood"):   # the other pairs, one run each
        for ps, geo in PAIRS:
            if (ps, geo) in ((16, "c8"), (8, "c4")):          # (drawn above: warm two-step)
                continue
            p = sched_pair(d, task, ps, geo, lr_pass.get(ps))
            if p:
                ax.plot(*zip(*p), ("-" if JOIN else "") + "*", color=PAIR_COL[(ps, geo)],
                        lw=2.4 * K, ms=13 * K, mec=PAIR_COL[(ps, geo)], mew=1.2 * K, zorder=6)
    ax.set_xscale("log")
    if lower:
        ax.invert_yaxis()
    ax.grid(True, color=rp.GRID, lw=0.8 * K)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(title, fontsize=9 if LABEL_PT else 12 * K, color="black", loc="left", fontweight="bold")
    ax.set_xlabel("GMACs per sample (log scale)")
    ax.set_ylabel(ylabel + (" (flipped direction)" if lower else ""))


FIGS = {   # name -> datasets, grid, figsize, title
    "seg": (SEG, (2, 2), (15, 12), "Mixed resolution against fixed patch sizes"),
    "change": (CHANGE, (1, 2), (15, 6.8),
               "Change detection: mixed resolution against fixed patch sizes"),
}


def draw_paper(name):
    """The paper version of a figure, one row across the CVPR text width (6.875 in): tall
    panels, no title, a few fixed points named, MiSTR's resolution pairs (colour) and the fixed-resolution models in legends below."""
    global K, LABEL_PS, JOIN, LABEL_PT
    specs, *_ = FIGS[name]
    K, LABEL_PS, JOIN, LABEL_PT = 0.55, True, False, 7.5
    # (CVPR typesets in Times: Liberation Serif = Times New Roman metrics, STIX maths; DejaVu
    # for the legend's star; TrueType in the PDF, no Type 3 fonts)
    with plt.rc_context({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "font.family": ["Liberation Serif", "DejaVu Sans"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, len(specs), figsize=(6.875, 3.75), facecolor=rp.SURFACE,
                                 squeeze=False)
        fams = set()
        for ax, spec in zip(axes.flatten(), specs):
            ax.set_facecolor(rp.SURFACE)
            panel(ax, *spec)
            ax.set_box_aspect(1.6)
            ax.margins(0.1)      # (room for the labels of the end points)
            lo, hi = ax.get_xlim()
            ax.set_xlim(lo / 1.8, hi * 2.6) if XM == "speed" else ax.set_xlim(lo, hi * 3.2)
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
            ax.xaxis.set_minor_formatter(NullFormatter())   # (no 2 x 10^2 under a decade)
            ax.set_xlabel(X_LABEL[XM] if ax in axes[-1] else "")
            # (one y label for all: up is better; each panel names its metric in its title)
            ax.set_ylabel("test metric $\\rightarrow$" if ax is axes[0, 0] else "")
            name_, kind = spec[1].rstrip(")").split(" (")
            ax.set_title(name_, loc="left", fontweight="bold", pad=13)
            ax.text(0, 1.02, f"{kind}, {spec[2].removeprefix('test ')}"
                    f"{' (flipped)' if spec[3] else ''}", fontsize=9,
                    transform=ax.transAxes, va="bottom")
            if XM == "mem":     # (under a decade per panel: plain GB ticks, no 3x10^-1 labels)
                lo, hi = ax.get_xlim()
                ax.set_xticks([t for t in (0.5, 1, 2, 4, 8) if lo <= t <= hi])
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                ax.xaxis.set_minor_formatter(NullFormatter())
            fams |= {f for f, *_ in fixed(spec[0], rows(spec[0]))}
        # (cells as wide as tall: no gap beside the square panels)
        fig.subplots_adjust(left=0.065, right=0.985, bottom=0.29, top=0.9, wspace=0.3)
        # (left: MiSTR, one colour per resolution pair; right: the fixed-resolution models; the
        # two side by side, centred under the panels with their tick and axis labels)
        star = dict(marker="*", ls="none", ms=13 * K, mew=0)
        h = [plt.Line2D([], [], color=PAIR_COL[k], **star) for k in PAIRS]
        leg = fig.legend(h, [pair_name(*k).replace("+", "$ & $") for k in PAIRS],
                         loc="lower left", borderaxespad=0, ncol=3, frameon=False,
                         title="\u2605 MiSTR (ours)", title_fontproperties={"weight": "bold"},
                         alignment="left", handlelength=1.2, columnspacing=1.2)
        fig.add_artist(leg)
        h, lab = [], []
        for fam in FAM:
            if fam in fams:
                ls, mk, filled, label = FAM[fam]
                h.append(plt.Line2D([], [], color=rp.INK, marker=mk, ms=7 * K, ls=ls,
                                    lw=1.2 * K, mfc=rp.INK if filled else rp.SURFACE,
                                    alpha=0.75))
                lab.append(label.replace("fixed resolution, ", ""))
        leg2 = fig.legend(h, lab, loc="lower left", borderaxespad=0, frameon=False,
                          title="fixed-res baselines", title_fontproperties={"weight": "bold"},
                          alignment="left", handlelength=2.6)
        r = fig.canvas.get_renderer()
        to_fig = fig.transFigure.inverted()
        span = to_fig.transform_bbox(Bbox.union([ax.get_tightbbox(r) for ax in axes.flat]))
        w1, w2 = (to_fig.transform_bbox(lg.get_window_extent(r)).width for lg in (leg, leg2))
        gap = 0.05
        left = (span.x0 + span.x1 - w1 - gap - w2) / 2
        leg.set_bbox_to_anchor((left, 0.0), transform=fig.transFigure)
        leg2.set_bbox_to_anchor((left + w1 + gap, 0.0), transform=fig.transFigure)
        out = Path(f"figures/datasets_tradeoff_{name}{XT.replace('-', '_')}"
                   f"{'' if XM == 'gmacs' else '_' + XM}.png")
        for o, dpi in ((out, 300), (out.with_suffix(".pdf"), None)):   # (no empty margin)
            fig.savefig(o, dpi=dpi, facecolor=rp.SURFACE, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    K, LABEL_PS, JOIN, LABEL_PT = 1.0, None, True, None
    print(out)


def draw(name):
    if XT and name == "seg":
        return draw_paper(name)
    specs, (nr, nc), size, title = FIGS[name]
    fig, axes = plt.subplots(nr, nc, figsize=size, facecolor=rp.SURFACE)
    fams = set()
    for ax, spec in zip(axes.flatten(), specs):
        ax.set_facecolor(rp.SURFACE)
        panel(ax, *spec)
        fams |= {f for f, *_ in fixed(spec[0], rows(spec[0]))}
    col = XHR[2] if XT else QUAD_COL
    if XT and name == "seg":    # ours per resolution pair, coloured as in spacetime_baselines
        h = [plt.Line2D([], [], color=PAIR_COL[k], marker="*", ms=13, lw=2.4)
             for k in PAIRS]
        lab = [f"ours {pair_name(*k)}" for k in PAIRS]
    else:
        h = [plt.Line2D([], [], color=col, marker="*", ms=13, lw=2.4)]
        lab = ["ours: uncertainty + attention" + (", cross attention" if XT else "")]
    for fam in FAM:              # every fixed family drawn in this figure
        if fam in fams:
            ls, mk, filled, label = FAM[fam]
            h.append(plt.Line2D([], [], color=rp.INK, marker=mk, ms=7, ls=ls, lw=1.2,
                                mfc=rp.INK if filled else rp.SURFACE, alpha=0.75))
            lab.append(label)
    if not (XT and name == "seg"):
        h += [plt.Line2D([], [], color=BASE_COL, marker="o", ms=6, ls="--", lw=1.6,
                         alpha=0.55),
              plt.Line2D([], [], color=BASE_COL, marker="d", ms=6, ls=":", lw=1.6, alpha=0.55)]
        lab += ["grid + grid", "random + random"]
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=10,
               handlelength=3)
    rows_ = -(-len(h) // 4)
    fig.suptitle(title, x=0.012, y=0.985, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.025 * rows_ * (2 if nr == 1 else 1) + 0.01, 1, 0.96))
    out = Path(f"figures/datasets_tradeoff_{name}{XT.replace('-', '_')}.png")
    fig.savefig(out, dpi=150, facecolor=rp.SURFACE)
    plt.close(fig)
    print(out)


def main():
    global XT, XM
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", default="joint", choices=sorted(ATTN_TAG))
    ap.add_argument("--x", default="gmacs", choices=("gmacs", *X_COL))
    a = ap.parse_args()
    XT, XM = ATTN_TAG[a.attn], a.x
    if XM != "gmacs" and not XT:
        raise SystemExit("--x mem / speed: measured for the --attn cross_hr runs only")
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "black",
                         "axes.labelcolor": "black", "xtick.color": "black",
                         "ytick.color": "black", "text.color": "black",
                         "font.family": "DejaVu Sans"})
    for name in FIGS if XM == "gmacs" else ("seg",):
        draw(name)


if __name__ == "__main__":
    main()
