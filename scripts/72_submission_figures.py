"""Journal figures for the JFS submission (JOFS-26-539), generated from saved outputs only.

Figure 1  schematic of target construction and the executed raw-byte Siamese model (redrawn; loss terms as executed:
          smooth L1 with beta = 0.05, margin ranking over random in-batch pairs, embedding-cosine term)
Figure 2  predicted versus composite target by pair-sampling mode (heuristic weights, seed 42, test pairs)
Figure 3  provenance tasks on the test pool (baselines and the five principal models, 95% cluster-bootstrap CIs)
Figure 4  rank-based examiner-effort curves (heuristic seed 42 shortlist)
Figure 5  greedy chaining of consecutive held-out blocks
Figure 6  Govdocs1 conditions (same-file AUC within extension group)

Each figure is written as 600-dpi TIFF (LZW), 600-dpi PNG and vector PDF, at most 7.0 in wide, on a white background,
without figure titles (titles belong in the legends).
usage: python scripts/72_submission_figures.py <output_dir>
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from paths import PAIRS, RESULTS  # noqa: E402

L = os.path.join(RESULTS, "local")
B = os.path.join(RESULTS, "bootstrap")
G = os.path.join(RESULTS, "govdocs1")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RESULTS, "figures_submission")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, WHITE, FILL = "#111111", "#4a4a4a", "#e3e3e3", "#ffffff", "#f1f2f4"
plt.rcParams.update({
    "font.family": "Arial", "font.size": 7.5, "mathtext.fontset": "custom", "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold", "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": WHITE, "axes.facecolor": WHITE,
    "savefig.facecolor": WHITE, "legend.frameon": False, "axes.linewidth": 0.7, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "pdf.fonttype": 42, "axes.axisbelow": True,
    "axes.unicode_minus": True})

BASE = {"hist_cosine": "Histogram cosine", "uniform_fusion_4": "Mean of four byte measures",
        "ngram_cosine": "Bigram cosine", "entropy_sim": "Entropy similarity",
        "compress_sim": "Compression-ratio similarity", "random": "Random scores"}
MODELS = {"heuristic_s42": "Heuristic, seed 42", "heuristic_s43": "Heuristic, seed 43",
          "heuristic_s44": "Heuristic, seed 44", "uniform_s42": "Uniform, seed 42",
          "learnability_s42": "Learnability-aware, seed 42"}


def jl(*p):
    p = os.path.join(*p)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".tif"), dpi=600, bbox_inches="tight", pad_inches=0.04,
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("figure", name)


# ---------------------------------------------------------------- Figure 1: schematic
def _box(ax, x, y, w, h, lines, fill=WHITE, title=None, fs=7.3, lw=0.8):
    """Box with an optional bold title line and centred body lines (line height in data units: 127 units = 8.9 in)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", fc=fill, ec=INK2, lw=lw))
    rows = ([(title, True)] if title else []) + [(t, False) for t in lines]
    lh = fs * 1.5 / 72 * (127 / 8.9)
    y0 = y + h / 2 + (len(rows) - 1) * lh / 2
    for k, (t, bold) in enumerate(rows):
        ax.text(x + w / 2, y0 - k * lh, t, ha="center", va="center", fontsize=fs + (0.4 if bold else 0),
                fontweight="bold" if bold else "normal", color=INK)


def _arrow(ax, p0, p1, ls="-", lw=0.8):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", lw=lw, color=INK2, ls=ls,
                                                     shrinkA=0, shrinkB=0, mutation_scale=7))


def _line(ax, xs, ys, ls="-", lw=0.8):
    ax.plot(xs, ys, color=INK2, lw=lw, ls=ls, solid_capstyle="butt")


def fig_schematic():
    fig = plt.figure(figsize=(7.0, 8.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 127)
    ax.axis("off")
    ax.grid(False)
    # Panel A
    ax.text(1, 125, "A", fontsize=10, fontweight="bold", va="top")
    ax.text(5, 125, "Composite training target", fontsize=9.5, fontweight="bold", va="top")
    _line(ax, [1, 99], [121.2, 121.2], lw=0.5)
    _box(ax, 1, 111, 72, 8, [r"$f_i,\ f_j$  (512 bytes each)"], title="Raw fragment pair")
    _box(ax, 76, 111, 23, 8, ["File, extension,", "offsets, file size"], title="Provenance records", fs=6.9)
    xs = [1, 25.5, 50, 76]
    heads = [("Byte statistics", ["Histogram cosine", "Entropy agreement", r"$s_{\rm hist},\ s_{\rm ent}$"]),
             ("Byte sequences", ["Hashed byte bigrams", "512 hash bins", r"$s_{\rm ngram}$"]),
             ("Compressibility", ["zlib ratio, level 6", "Ratio agreement", r"$s_{\rm comp}$"]),
             ("Provenance context", ["Relative position", "Extension agreement", r"$s_{\rm pos},\ s_{\rm type}$"])]
    for x, (t, lines) in zip(xs, heads):
        _box(ax, x, 92, 23, 14, lines, title=t, fs=6.9)
    _line(ax, [37, 37], [111, 108.5])
    _line(ax, [12.5, 61.5], [108.5, 108.5])
    for x in (12.5, 37, 61.5):
        _arrow(ax, (x, 108.5), (x, 106))
    _arrow(ax, (87.5, 111), (87.5, 106))
    _line(ax, [12.5, 87.5], [89.5, 89.5])
    for x in (12.5, 37, 61.5, 87.5):
        _line(ax, [x, x], [92, 89.5])
    _arrow(ax, (27, 89.5), (27, 87))
    _box(ax, 1, 79, 52, 8, [r"$\mathbf{s}_{ij} = (s_{\rm hist}, s_{\rm ent}, s_{\rm ngram}, s_{\rm comp}, s_{\rm pos}, s_{\rm type})$"],
         title="Six similarity components", fs=7.0)
    _box(ax, 56, 79, 43, 8, ["Heuristic / uniform / learnability-aware", r"learnability-aware: from validation $R^2$ only"],
         title="Component weights $w_k$", fs=6.8)
    _line(ax, [27, 27], [79, 76.5])
    _line(ax, [77.5, 77.5], [79, 76.5])
    _line(ax, [27, 38], [76.5, 76.5])
    _line(ax, [77.5, 62], [76.5, 76.5])
    _arrow(ax, (38, 76.5), (38, 74))
    _arrow(ax, (62, 76.5), (62, 74))
    _box(ax, 26, 65, 48, 9, [r"$y_{ij} = \mathrm{clip}\left(\sum_k w_k\, s_{ij}^{(k)},\ 0,\ 1\right)$"],
         fill=FILL, title="Composite target $y_{ij}$", fs=7.4)
    # supervision link (dashed) to the training objective
    _line(ax, [74, 98.2], [69.5, 69.5], ls=(0, (3, 2)))
    _line(ax, [98.2, 98.2], [69.5, 12.5], ls=(0, (3, 2)))
    _arrow(ax, (98.2, 12.5), (96.6, 12.5), ls="-")
    ax.text(97.2, 62.6, r"training supervision only: $y_{ij}$", fontsize=6.5, color=INK2, ha="right", va="center")
    # Panel B
    ax.text(1, 60, "B", fontsize=10, fontweight="bold", va="top")
    ax.text(5, 60, "Raw-byte Siamese model", fontsize=9.5, fontweight="bold", va="top")
    ax.text(97, 59.6, "1,772,289 parameters", fontsize=7, color=INK2, ha="right", va="top")
    _line(ax, [1, 99], [56.2, 56.2], lw=0.5)
    _box(ax, 1, 45.5, 16, 7.5, [r"512 bytes $\rightarrow [0,1]$"], title="Fragment $f_i$", fs=6.8)
    _box(ax, 1, 34.5, 16, 7.5, [r"512 bytes $\rightarrow [0,1]$"], title="Fragment $f_j$", fs=6.8)
    _box(ax, 23, 45.5, 18, 7.5, [r"Encoder $E_\theta$"], fs=7.4)
    _box(ax, 23, 34.5, 18, 7.5, [r"Encoder $E_\theta$"], fs=7.4)
    _line(ax, [32, 32], [42, 45.5], ls=(0, (2, 2)), lw=0.6)
    ax.text(33, 43.75, "shared weights", fontsize=6.3, color=INK2, va="center")
    _arrow(ax, (17, 49.25), (23, 49.25))
    _arrow(ax, (17, 38.25), (23, 38.25))
    _arrow(ax, (41, 49.25), (57, 49.25))
    _arrow(ax, (41, 38.25), (57, 38.25))
    ax.text(49, 50.4, r"$\mathbf{z}_i$", fontsize=8, ha="center")
    ax.text(49, 39.4, r"$\mathbf{z}_j$", fontsize=8, ha="center")
    _box(ax, 57, 33, 39.6, 21.5, [r"$[\mathbf{z}_i,\ \mathbf{z}_j,\ |\mathbf{z}_i-\mathbf{z}_j|,\ \mathbf{z}_i\odot\mathbf{z}_j,\ \cos(\mathbf{z}_i,\mathbf{z}_j)]$",
                                  "3-layer MLP + sigmoid", r"$\hat{y}_{ij} \in [0, 1]$"],
         title="Pairwise regression head", fs=7.0)
    _arrow(ax, (76.8, 33), (76.8, 28))
    ax.text(78, 30.5, r"$\hat{y}_{ij}$", fontsize=8)
    # shared encoder detail
    ax.add_patch(FancyBboxPatch((1, 1), 54, 29.5, boxstyle="square,pad=0", fc=FILL, ec=INK2, lw=0.8))
    ax.text(3, 28.8, r"Shared encoder $E_\theta$", fontsize=7.6, fontweight="bold", va="top")
    _box(ax, 3, 9.5, 24, 15.5, ["64, 128, 256, 512 channels", "kernels 9, 7, 5, 3", "batch norm., GELU, max pooling",
                                "global average pooling"], title="1-D CNN branch", fs=6.5)
    _box(ax, 29, 9.5, 24, 15.5, ["byte projection to 128-D", "sinusoidal positions", "3 pre-norm layers, 8 heads",
                                 "feed-forward 256; mean pooling"], title="Transformer branch", fs=6.5)
    _line(ax, [15, 15], [9.5, 7.6])
    _line(ax, [41, 41], [9.5, 7.6])
    _line(ax, [15, 41], [7.6, 7.6])
    _arrow(ax, (28, 7.6), (28, 6.6))
    _box(ax, 6, 2, 44, 4.6, [r"concatenate $\rightarrow$ 2-layer fusion MLP $\rightarrow$ 256-D $\mathbf{z}$, $L_2$-normalized"], fs=6.6)
    ax.text(36, 28.75, "input: 512 scaled byte values", fontsize=6.4, color=INK2, ha="center", va="center")
    _line(ax, [28, 28], [27.9, 26.8])
    _line(ax, [15, 41], [26.8, 26.8])
    _arrow(ax, (15, 26.8), (15, 25))
    _arrow(ax, (41, 26.8), (41, 25))
    # training objective
    _box(ax, 57, 1, 39.6, 27, [r"$\mathcal{L} = \mathcal{L}_{\rm reg} + 0.3\,\mathcal{L}_{\rm rank} + 0.2\,\mathcal{L}_{\rm emb}$",
                               r"$\mathcal{L}_{\rm reg}$ = smooth L1$_\beta(\hat{y}, y)$,  $\beta = 0.05$",
                               r"$\mathcal{L}_{\rm rank}$: margin 0.02, random in-batch pairs",
                               r"$\mathcal{L}_{\rm emb}$ = smooth L1$_\beta((\cos(\mathbf{z}_i,\mathbf{z}_j)+1)/2,\ y)$",
                               "AdamW; lowest validation loss kept"],
         title="Training objective", fs=6.6)
    save(fig, "Figure_1")


# ---------------------------------------------------------------- Figure 2: predicted vs target
MODE_LABEL = {"same_file": "Same file", "same_type_diff_file": "Same extension, different file",
              "diff_type": "Different extensions", "random": "Unrestricted draw"}


def fig_pred_vs_target(run="heuristic_s42"):
    d = np.load(os.path.join(L, "predictions", f"{run}.npz"))
    modes = pd.read_csv(os.path.join(PAIRS, "test_pairs.csv"), usecols=["pair_mode"]).pair_mode.to_numpy()
    reg = jl(L, "regression", f"{run}.json")
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 5.4), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    for k, (ax, mode) in enumerate(zip(axes.ravel(), ["same_file", "same_type_diff_file", "diff_type", "random"])):
        idx = np.flatnonzero(modes == mode)
        idx = rng.choice(idx, min(3000, len(idx)), replace=False)
        ax.scatter(d["y"][idx], d["p_ab"][idx], s=3.5, c=BLUE, alpha=0.3, linewidths=0, rasterized=True)
        ax.plot([0, 1], [0, 1], ls="--", lw=0.9, c=INK2)
        bm = reg["by_pair_mode"][mode]
        r2txt = f"{bm['r2']:.2f}".replace("-", "−")
        ax.set_title(f"{'ABCD'[k]}  {MODE_LABEL[mode]}\n$R^2$ = {r2txt}; n = {bm['n']:,} pairs", loc="left",
                     color=INK, fontsize=7.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
    for ax in axes[1]:
        ax.set_xlabel("Composite target (heuristic weights)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted similarity")
    fig.subplots_adjust(wspace=0.12, hspace=0.32)
    save(fig, "Figure_2")


def _ci(bs, m, k):
    try:
        return bs["methods"][m]["metrics"][k]["ci95"]
    except (KeyError, TypeError):
        return None


# ---------------------------------------------------------------- Figure 3: provenance
def fig_provenance():
    bs = jl(B, "local_summary.json")
    pts = [(m, lab, jl(L, "provenance", f"test__{m}.json")) for m, lab in list(BASE.items()) + list(MODELS.items())]
    order = sorted(pts, key=lambda t: t[2]["retrieval_all"]["hit@1"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 0.3 * len(order) + 1.6), sharey=True,
                               gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.08})
    for i, (m, lab, p) in enumerate(order):
        col = BLUE if m in BASE else ORANGE
        c1, c2 = _ci(bs, m, "auc_same_file_within_type"), _ci(bs, m, "auc_same_type")
        a.plot(c1, [i - 0.17] * 2, c=col, lw=0.8)
        a.plot(c2, [i + 0.17] * 2, c=col, lw=0.8, alpha=0.6)
        a.plot(p["auc_same_file_within_type"], i, "o", c=col, ms=5, mec=WHITE, mew=0.7)
        a.plot(p["auc_same_type"], i, "D", c=col, ms=4.2, mec=WHITE, mew=0.7)
        h1, h2 = p["retrieval_all"]["hit@1"], p["retrieval_type"]["hit@1"]
        r1, r2 = _ci(bs, m, "retrieval_all_hit@1"), _ci(bs, m, "retrieval_type_hit@1")
        b.plot([h1, h2], [i, i], c=col, lw=0.7, alpha=0.45)
        b.plot(r1, [i - 0.17] * 2, c=col, lw=0.8)
        b.plot(r2, [i + 0.17] * 2, c=col, lw=0.8, alpha=0.6)
        b.plot(h1, i, "o", c=col, ms=5, mec=WHITE, mew=0.7)
        b.plot(h2, i, "o", mfc=WHITE, mec=col, ms=5, mew=1.0)
    a.axvline(0.5, ls=":", lw=0.8, c=INK2)
    a.set_xlim(0.47, 0.72)
    a.set_yticks(range(len(order)))
    a.set_yticklabels([t[1] for t in order], fontsize=7)
    a.set_xlabel("AUC")
    a.set_title("A  Pair discrimination", loc="left", color=INK, fontsize=8)
    b.set_xlabel("Same-file retrieval Hit@1")
    b.set_title("B  Same-file retrieval", loc="left", color=INK, fontsize=8)
    b.tick_params(axis="y", length=0)
    handles = [Line2D([], [], marker="o", ls="", c=INK2, mec=WHITE, ms=5, label="A: same file within extension group"),
               Line2D([], [], marker="D", ls="", c=INK2, mec=WHITE, ms=4.2, label="A: same extension group"),
               Line2D([], [], marker="o", ls="", c=INK2, mec=WHITE, ms=5, label="B: all candidates (practical)"),
               Line2D([], [], marker="o", ls="", mfc=WHITE, mec=INK2, ms=5, label="B: extension known (oracle)"),
               Line2D([], [], marker="s", ls="", c=BLUE, ms=5, label="byte-statistic baseline"),
               Line2D([], [], marker="s", ls="", c=ORANGE, ms=5, label="learned model")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.8, bbox_to_anchor=(0.52, -0.02))
    fig.subplots_adjust(bottom=0.2)
    save(fig, "Figure_3")


# ---------------------------------------------------------------- Figure 4: examiner effort
def fig_effort():
    s = jl(L, "shortlist.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for k, (ax, task, title) in enumerate(zip(axes, ("retrieval", "adjacency_pooled"),
                                              ("Same-file retrieval", "Adjacent block, pooled candidates"))):
        d = pd.read_csv(os.path.join(L, "per_query", f"shortlist__heuristic_s42__{task}.csv"))
        d = d[d.two_stage_rank != -1]
        n = len(d)
        top = max(d.direct_hist_rank.max(), d.direct_learned_rank.max()) + 1
        ks = np.unique(np.round(np.logspace(0, np.log10(top), 80)).astype(int))
        for col, lab, r in ((BLUE, "Histogram cosine", d.direct_hist_rank.to_numpy()),
                            (ORANGE, "Learned score (heuristic, seed 42)", d.direct_learned_rank.to_numpy())):
            ax.plot(ks, [(r < kk).mean() for kk in ks], c=col, lw=1.4, label=lab)
        M = s["heuristic_s42"][task]["selected_M"]
        r = np.nan_to_num(d.two_stage_rank.to_numpy(dtype=float), nan=np.inf)
        kk = ks[ks <= M]
        ax.plot(kk, [(r < x).mean() for x in kk], c=AQUA, lw=1.4, ls="--",
                label="Learned shortlist re-ranked by histogram cosine")
        ax.axhline(0.8, ls=":", lw=0.7, c=INK2)
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Candidates inspected per query (k)")
        ax.set_ylabel("Share of queries with a true match in the top k")
        ax.set_title(f"{'AB'[k]}  {title} ({n:,} queries; M = {M:,})", loc="left", color=INK, fontsize=7.6)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=6.8, bbox_to_anchor=(0.5, -0.07))
    fig.subplots_adjust(bottom=0.22, wspace=0.3)
    save(fig, "Figure_4")


def family_legend(fig, y=-0.02):
    handles = [Line2D([], [], marker="o", ls="", c=BLUE, mec=WHITE, ms=5, label="byte-statistic baseline"),
               Line2D([], [], marker="s", ls="", c=ORANGE, mec=WHITE, ms=4.6, label="learned model")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, y))


# ---------------------------------------------------------------- Figure 5: chaining
def fig_reconstruction():
    bs = jl(B, "recon_summary.json")
    rows = [(m, lab, jl(L, "reconstruction", f"{m}.json")) for m, lab in list(BASE.items()) + list(MODELS.items())]
    rows.sort(key=lambda t: t[2]["adjacency_recall"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 0.3 * len(rows) + 1.3), sharey=True, gridspec_kw={"wspace": 0.18})
    for i, (m, lab, r) in enumerate(rows):
        col, mk = (BLUE, "o") if m in BASE else (ORANGE, "s")
        for ax, x, key in ((a, r["adjacency_recall"], "adjacency_recall"),
                           (b, r["oracle_stop"]["chain_purity_weighted"], "chain_purity_weighted")):
            ci = bs["methods"][m][key]["ci95"]
            ax.plot(ci, [i, i], c=col, lw=0.9)
            ax.plot(x, i, mk, c=col, ms=5 if mk == "o" else 4.6, mec=WHITE, mew=0.7)
    a.set_yticks(range(len(rows)))
    a.set_yticklabels([t[1] for t in rows], fontsize=7)
    a.set_xlim(0, None)
    b.set_xlim(0, None)
    b.tick_params(axis="y", length=0)
    a.set_xlabel("Share of true adjacent pairs linked")
    b.set_xlabel("Chain purity (fragments in each chain's majority file)")
    a.set_title("A  Adjacency recall, all accepted links", loc="left", color=INK, fontsize=8)
    b.set_title("B  Chain purity at the oracle stop", loc="left", color=INK, fontsize=8)
    family_legend(fig, -0.01)
    fig.subplots_adjust(bottom=0.2)
    save(fig, "Figure_5")


# ---------------------------------------------------------------- Figure 6: Govdocs1 conditions
def fig_govdocs():
    conds = ["aligned", "unaligned", "encrypted", "compressed", "b4096"]
    titles = {"aligned": "A  Aligned", "unaligned": "B  Unaligned", "encrypted": "C  AES-256",
              "compressed": "D  zlib", "b4096": "E  4096-byte"}
    methods = list(BASE) + list(MODELS)
    bs = jl(B, "govdocs_summary.json")["conditions"]
    fig, axes = plt.subplots(1, 5, figsize=(7.0, 0.27 * len(methods) + 1.3), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    for ax, c in zip(axes, conds):
        for i, m in enumerate(methods):
            r = jl(G, f"{c}__{m}.json")
            col = BLUE if m in BASE else ORANGE
            ci = bs[c]["methods"][m]["auc_same_file_within_type"]["ci95"]
            ax.plot(ci, [i, i], c=col, lw=1.0, alpha=0.8, solid_capstyle="butt")
            ax.plot(r["auc_same_file_within_type"], i, "o" if m in BASE else "s", c=col, ms=4.4, mec=WHITE, mew=0.6)
        ax.axvline(0.5, ls=":", lw=0.8, c=INK2)
        ax.set_title(titles[c], loc="left", color=INK, fontsize=7.6)
        ax.set_xlim(0.35, 1.0)
        ax.set_xticks([0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis="x", labelsize=6.5)
    axes[0].set_yticks(range(len(methods)))
    axes[0].set_yticklabels([BASE.get(m, MODELS.get(m)) for m in methods], fontsize=6.8)
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0)
    fig.text(0.6, 0.075, "Same-file AUC within extension group", ha="center", fontsize=7.5)
    family_legend(fig, -0.035)
    fig.subplots_adjust(bottom=0.2)
    save(fig, "Figure_6")


if __name__ == "__main__":
    for fn in (fig_schematic, fig_pred_vs_target, fig_provenance, fig_effort, fig_reconstruction, fig_govdocs):
        fn()
