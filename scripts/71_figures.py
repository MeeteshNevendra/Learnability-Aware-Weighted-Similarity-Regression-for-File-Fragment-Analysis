"""Figures from saved outputs (PNG 300 dpi + PDF). Palette: validated categorical slots blue / orange / aqua
(dataviz reference palette, all-pairs check passed; aqua below 3:1 contrast, so every series is also labelled).
Output: results/figures/"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from paths import RESULTS  # noqa: E402

L = os.environ.get("FSR_LOCAL_OUT", os.path.join(RESULTS, "local"))
SPLIT = os.environ.get("FSR_EVAL_SPLIT", "test")
B = os.path.join(RESULTS, "bootstrap")
G = os.path.join(RESULTS, "govdocs1")
F = os.environ.get("FSR_FIG_OUT", os.path.join(RESULTS, "figures"))
os.makedirs(F, exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "legend.frameon": False})
BASE = {"hist_cosine": "Histogram cosine", "uniform_fusion_4": "Mean of four", "ngram_cosine": "Bigram cosine",
        "entropy_sim": "Entropy sim.", "compress_sim": "Compression sim.", "random": "Random"}
MODELS = {"heuristic_s42": "Heuristic s42", "heuristic_s43": "Heuristic s43", "heuristic_s44": "Heuristic s44",
          "uniform_s42": "Uniform", "learnability_s42": "Learnability-aware", "pos_type": "s_pos+s_type target",
          "comp_hist": "s_hist target"}


def jl(*p):
    p = os.path.join(*p)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def save(fig, name):
    fig.savefig(os.path.join(F, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(F, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("figure", name)


def fig_pred_vs_target(run="heuristic_s42"):
    z = os.path.join(L, "predictions", f"{run}.npz")
    if not os.path.exists(z):
        return
    d = np.load(z)
    from paths import PAIRS
    modes = pd.read_csv(os.path.join(PAIRS, f"{SPLIT}_pairs.csv"), usecols=["pair_mode"]).pair_mode.to_numpy()
    reg = jl(L, "regression", f"{run}.json")
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 5.6), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    for ax, mode in zip(axes.ravel(), ["same_file", "same_type_diff_file", "diff_type", "random"]):
        idx = np.flatnonzero(modes == mode)
        idx = rng.choice(idx, min(3000, len(idx)), replace=False)
        ax.scatter(d["y"][idx], d["p_ab"][idx], s=4, c=BLUE, alpha=0.25, linewidths=0)
        ax.plot([0, 1], [0, 1], ls="--", lw=1, c=INK2)
        r2 = reg["by_pair_mode"][mode]["r2"]
        ax.set_title(f"{mode.replace('_', ' ')}  (R² = {r2:.2f}, n = {reg['by_pair_mode'][mode]['n']:,})", color=INK, fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    for ax in axes[1]:
        ax.set_xlabel("Composite target (heuristic weights)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted similarity")
    fig.suptitle(f"Predicted versus target, {run}, {SPLIT} pairs (3,000 sampled per mode)", color=INK, fontsize=9)
    save(fig, "F1_pred_vs_target_by_mode")


def place_labels(ax, items, fontsize=6.5):
    """Greedy collision-avoiding labels: items = [(x, y, text)]. Tries four offsets per label and keeps the first
    whose bounding box overlaps no earlier label."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    placed = []
    for x, y, text in sorted(items, key=lambda t: -t[1]):
        for dx, dy, ha in ((5, 3, "left"), (5, -9, "left"), (-5, 3, "right"), (-5, -9, "right"), (5, 10, "left"), (5, -16, "left")):
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize, color=INK2, ha=ha)
            bb = t.get_window_extent(renderer=r).expanded(1.05, 1.15)
            if not any(bb.overlaps(p) for p in placed):
                placed.append(bb)
                break
            t.remove()
        else:
            t = ax.annotate(text, (x, y), xytext=(5, 3), textcoords="offset points", fontsize=fontsize, color=INK2)
            placed.append(t.get_window_extent(renderer=r))


def _ci(bs, m, k):
    try:
        return bs["methods"][m]["metrics"][k]["ci95"]
    except (KeyError, TypeError):
        return None


def fig_provenance():
    bs = jl(B, "local_summary.json") if SPLIT == "test" else None
    pts = []
    for m, lab in list(BASE.items()) + list(MODELS.items()):
        p = jl(L, "provenance", f"{SPLIT}__{m}.json")
        if p:
            pts.append((m, lab, p))
    if not pts:
        return
    from matplotlib.lines import Line2D
    # One row per method in both panels (rows sorted by retrieval Hit@1), so no free-floating labels are needed.
    order = sorted(pts, key=lambda t: t[2]["retrieval_all"]["hit@1"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 0.3 * len(order) + 1.7), sharey=True,
                               gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.08})
    lo = 1.0
    for i, (m, lab, p) in enumerate(order):
        col = BLUE if m in BASE else ORANGE
        x1, x2 = p["auc_same_file_within_type"], p["auc_same_type"]
        lo = min(lo, x1, x2)
        c1, c2 = _ci(bs, m, "auc_same_file_within_type"), _ci(bs, m, "auc_same_type")
        if c1:
            a.plot(c1, [i - 0.17, i - 0.17], c=col, lw=0.8)
        if c2:
            a.plot(c2, [i + 0.17, i + 0.17], c=col, lw=0.8, alpha=0.7)
        a.plot(x1, i, "o", c=col, ms=5.5, mec=SURF, mew=0.8)
        a.plot(x2, i, "D", c=col, ms=4.6, mec=SURF, mew=0.8)
        h1, h2 = p["retrieval_all"]["hit@1"], p["retrieval_type"]["hit@1"]
        r1, r2 = _ci(bs, m, "retrieval_all_hit@1"), _ci(bs, m, "retrieval_type_hit@1")
        b.plot([h1, h2], [i, i], c=col, lw=0.8, alpha=0.5)
        if r1:
            b.plot(r1, [i - 0.17, i - 0.17], c=col, lw=0.8)
        if r2:
            b.plot(r2, [i + 0.17, i + 0.17], c=col, lw=0.8, alpha=0.7)
        b.plot(h1, i, "o", c=col, ms=5.5, mec=SURF, mew=0.8)
        b.plot(h2, i, "o", mfc=SURF, mec=col, ms=5.5, mew=1.1)
    a.axvline(0.5, ls=":", lw=0.8, c=INK2)
    a.set_xlim(min(lo, 0.5) - 0.02, None)
    a.set_yticks(range(len(order)))
    a.set_yticklabels([t[1] for t in order], fontsize=6.5)
    a.set_xlabel("AUC (dotted line: chance)")
    a.set_title("A  Pair discrimination", loc="left", color=INK, fontsize=8)
    b.set_xlabel("Same-file retrieval Hit@1")
    b.set_title("B  Same-file retrieval", loc="left", color=INK, fontsize=8)
    b.tick_params(axis="y", length=0)
    # legend columns fill top to bottom: panel A markers | panel B markers | colours
    handles = [Line2D([], [], marker="o", ls="", c=INK2, mec=SURF, ms=5.5, label="A: same file, within extension group"),
               Line2D([], [], marker="D", ls="", c=INK2, mec=SURF, ms=4.6, label="A: same extension group"),
               Line2D([], [], marker="o", ls="", c=INK2, mec=SURF, ms=5.5, label="B: all candidates (practical)"),
               Line2D([], [], marker="o", ls="", mfc=SURF, mec=INK2, ms=5.5, label="B: extension known (oracle)"),
               Line2D([], [], marker="s", ls="", c=BLUE, ms=5, label="byte-statistic baseline"),
               Line2D([], [], marker="s", ls="", c=ORANGE, ms=5, label="learned model")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.8, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(bottom=0.17)
    save(fig, "F2_provenance")


def fig_effort():
    s = jl(L, "shortlist.json")
    if not s or "heuristic_s42" not in s:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax, task, title in zip(axes, ("retrieval", "adjacency_pooled"), ("Same-file retrieval", "Adjacent block, pooled candidates")):
        p = os.path.join(L, "per_query", f"shortlist__heuristic_s42__{task}.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        d = d[d.two_stage_rank != -1]
        n = len(d)
        ks = np.unique(np.round(np.logspace(0, np.log10(max(d.direct_hist_rank.max(), d.direct_learned_rank.max()) + 1), 60)).astype(int))
        for col, lab, r in ((BLUE, "Histogram cosine", d.direct_hist_rank.to_numpy()), (ORANGE, "Learned (heuristic s42)", d.direct_learned_rank.to_numpy())):
            ax.plot(ks, [(r < k).mean() for k in ks], c=col, lw=1.6, label=lab)
        M = s["heuristic_s42"][task]["selected_M"]
        r = d.two_stage_rank.to_numpy()
        kk = ks[ks <= M]
        ax.plot(kk, [np.nan_to_num(r, nan=np.inf).__lt__(k).mean() for k in kk], c=AQUA, lw=1.6, ls="--",
                label="Learned shortlist, re-ranked by histogram (ends at k = M)")
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Candidates inspected (k)")
        ax.set_ylabel("Queries with a true match in top k")
        ax.set_title(f"{title}\n{n:,} queries; shortlist M = {M:,}", loc="left", color=INK, fontsize=8)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.06))
    fig.subplots_adjust(bottom=0.2, wspace=0.3)
    save(fig, "F3_examiner_effort")


def family_legend(fig, y=-0.02):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", c=BLUE, mec=SURF, ms=5.5, label="byte-statistic baseline"),
               Line2D([], [], marker="s", ls="", c=ORANGE, mec=SURF, ms=5, label="learned model")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, y))


def fig_govdocs():
    conds = ["aligned", "unaligned", "encrypted", "compressed", "b4096"]
    methods = [m for m in list(BASE) + ["heuristic_s42", "heuristic_s43", "heuristic_s44", "uniform_s42", "learnability_s42"]
               if os.path.exists(os.path.join(G, f"aligned__{m}.json"))]
    if not methods:
        return
    titles = {"aligned": "Aligned", "unaligned": "Unaligned", "encrypted": "AES-256 encrypted",
              "compressed": "zlib compressed", "b4096": "4,096-byte blocks"}
    bs = (jl(B, "govdocs_summary.json") or {}).get("conditions", {}) if SPLIT == "test" else {}
    fig, axes = plt.subplots(1, 5, figsize=(9.6, 0.27 * len(methods) + 1.4), sharey=True)
    for ax, c in zip(axes, conds):
        for i, m in enumerate(methods):
            r = jl(G, f"{c}__{m}.json")
            if not r:
                continue
            col = BLUE if m in BASE else ORANGE
            b = bs.get(c, {}).get("methods", {}).get(m, {})
            ci = (b.get("metrics", b).get("auc_same_file_within_type") or {}).get("ci95")
            if ci:
                ax.plot(ci, [i, i], c=col, lw=1.2, alpha=0.75, solid_capstyle="butt")
            ax.plot(r["auc_same_file_within_type"], i, "o" if m in BASE else "s", c=col, ms=5, mec=SURF, mew=0.8)
        ax.axvline(0.5, ls=":", lw=0.8, c=INK2)
        ax.set_title(titles[c], color=INK, fontsize=8)
        ax.set_xlim(0.35, 1.0)
    axes[0].set_yticks(range(len(methods)))
    axes[0].set_yticklabels([BASE.get(m, MODELS.get(m, m)) for m in methods], fontsize=6.5)
    axes[2].set_xlabel("Same-file AUC within extension group, 95% CI (Govdocs1, no retraining; dotted line: chance)")
    family_legend(fig, -0.01)
    fig.subplots_adjust(bottom=0.22)
    save(fig, "F4_govdocs_conditions")


def fig_reconstruction():
    rows = []
    bs = jl(B, "recon_summary.json") if SPLIT == "test" else None
    for m, lab in list(BASE.items()) + list(MODELS.items()):
        r = jl(L, "reconstruction", f"{m}.json")
        if r:
            rows.append((m, lab, r))
    if not rows:
        return
    pool = jl(L, "reconstruction", "pool.json") or {}
    rows.sort(key=lambda t: t[2]["adjacency_recall"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 0.3 * len(rows) + 1.5), sharey=True, gridspec_kw={"wspace": 0.2})
    for i, (m, lab, r) in enumerate(rows):
        col, mk = (BLUE, "o") if m in BASE else (ORANGE, "s")
        x1, x2 = r["adjacency_recall"], r["oracle_stop"]["chain_purity_weighted"]
        meth = (bs or {}).get("methods", {}).get(m, {})
        for ax, x, key in ((a, x1, "adjacency_recall"), (b, x2, "chain_purity_weighted")):
            ci = meth.get(key, {}).get("ci95")
            if ci:
                ax.plot(ci, [i, i], c=col, lw=0.9)
            ax.plot(x, i, mk, c=col, ms=5.5 if mk == "o" else 5, mec=SURF, mew=0.8)
    a.set_yticks(range(len(rows)))
    a.set_yticklabels([t[1] for t in rows], fontsize=6.5)
    a.set_xlim(0, None)
    b.set_xlim(0, None)
    b.tick_params(axis="y", length=0)
    a.set_xlabel("Share of true neighbour pairs linked")
    b.set_xlabel("Fragments in the chain's majority file (weighted)")
    a.set_title("A  Adjacency recall (all accepted links)", loc="left", color=INK, fontsize=8)
    b.set_title("B  Chain purity (stop at n - F links, F known)", loc="left", color=INK, fontsize=8)
    if pool:
        fig.suptitle(f"Greedy chaining of 24-block runs: {pool.get('files')} test files, {pool.get('fragments'):,} fragments "
                     "pooled without source identity", x=0.02, ha="left", color=INK, fontsize=8.5, y=0.97)
    family_legend(fig, 0.0)
    fig.subplots_adjust(bottom=0.22, top=0.86)
    save(fig, "F5_reconstruction")


if __name__ == "__main__":
    for fn in (fig_pred_vs_target, fig_provenance, fig_effort, fig_govdocs, fig_reconstruction):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print("figure skipped:", fn.__name__, type(e).__name__, e)
