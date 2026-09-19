#!/usr/bin/env python3
"""plot_cuda_replication.py — charts for docs/CUDA_REPLICATION.md.

Reads ONLY the committed analysis_*.json files (no hardcoded numbers) and produces
docs/cuda_replication.png with four panels:
  1. McNemar 2x2 contingency table for the CUDA run (fp16 vs int4, PopQA n=100)
  2. McNemar disagreement cells b/c for CUDA vs the original MLX paired models
  3. INT4-FP16 accuracy difference with 95% bootstrap CI, all paired runs (the "null" view)
  4. Controlled scale experiment: memorization GAP vs model size at fixed INT4

Usage: python src/plot_cuda_replication.py   (from repo root)
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")

def load(name):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)

# ---------- data (all from committed JSON) ----------
cuda = load("analysis_cuda_popqa_qwen05.json")
mlx = {t: load(f"analysis_popqa_{t}.json") for t in ("qwen05", "qwen15", "llama1b")}

def paired(d):
    pc, cm = d["per_precision"], d["comparisons"]["qa_int4_vs_fp16"]
    return dict(fp16=pc["fp16"]["qa_accuracy"], int4=pc["int4"]["qa_accuracy"],
                n=pc["fp16"]["n_qa"], b=cm["mcnemar"]["b"], c=cm["mcnemar"]["c"],
                p=cm["mcnemar"]["p"], **cm["acc_diff_ci"])

runs = [("CUDA\nqwen05\n(RTX 5060)", paired(cuda))] + \
       [(f"MLX\n{t}\n(Apple)", paired(mlx[t])) for t in ("qwen05", "qwen15", "llama1b")]

scale = []
for tag, size in (("qwen05", 0.5), ("qwen15", 1.5), ("qwen3b", 3.0)):
    d = load(f"analysis_scale_{tag}.json")
    scale.append((size, d["per_precision"]["int4"]["mem_GAP"]))

# ---------- figure ----------
fig, ax = plt.subplots(2, 2, figsize=(14, 10.5))

# Panel 1: McNemar 2x2 for the CUDA run
r = runs[0][1]
n = r["n"]
# a = both right, d = both wrong; derive from accuracies and b/c
a = round(r["fp16"] * n) - r["b"]
dd = n - a - r["b"] - r["c"]
cells = [[a, r["b"]], [r["c"], dd]]
labels = [["a: both correct", "b: FP16 only\n(INT4 lost)"],
          ["c: INT4 only\n(INT4 gained)", "d: both wrong"]]
colors = [["#dfe9f3", "#f7c9c9"], ["#c9e8c9", "#e8e8e8"]]
ax0 = ax[0, 0]
ax0.set_xlim(0, 2); ax0.set_ylim(0, 2); ax0.set_aspect("equal"); ax0.axis("off")
for i in range(2):
    for j in range(2):
        ax0.add_patch(plt.Rectangle((j, 1 - i), 1, 1, facecolor=colors[i][j], edgecolor="k", lw=1.5))
        ax0.text(j + .5, 1 - i + .62, str(cells[i][j]), ha="center", va="center", fontsize=26, weight="bold")
        ax0.text(j + .5, 1 - i + .22, labels[i][j], ha="center", va="center", fontsize=9.5)
ax0.text(0.5, 2.08, "INT4 correct", ha="center", fontsize=11); ax0.text(1.5, 2.08, "INT4 wrong", ha="center", fontsize=11)
ax0.text(-0.08, 1.5, "FP16\ncorrect", ha="right", va="center", fontsize=11)
ax0.text(-0.08, 0.5, "FP16\nwrong", ha="right", va="center", fontsize=11)
chi2 = (abs(r["b"] - r["c"]) - 1) ** 2 / (r["b"] + r["c"])
ax0.set_title(f"McNemar 2×2 — CUDA run (Qwen2.5-0.5B, PopQA n={n})\n"
              f"only b vs c matters:  b={r['b']}, c={r['c']}  →  χ²={chi2:.2f},  p={r['p']}",
              fontsize=11.5, pad=28)

# Panel 2: b vs c per run (CUDA vs MLX)
ax1 = ax[0, 1]
names = [nm for nm, _ in runs]
bs = [rr["b"] for _, rr in runs]; cs = [rr["c"] for _, rr in runs]
x = range(len(runs)); w = 0.38
ax1.bar([i - w/2 for i in x], bs, w, color="#e07b7b", label="b = INT4 lost (FP16 only correct)")
ax1.bar([i + w/2 for i in x], cs, w, color="#7bbf7b", label="c = INT4 gained (INT4 only correct)")
for i, (_, rr) in enumerate(runs):
    ax1.text(i, max(rr["b"], rr["c"]) + 0.6, f"p = {rr['p']}", ha="center", fontsize=10, weight="bold")
ax1.set_xticks(list(x)); ax1.set_xticklabels(names, fontsize=9)
ax1.set_ylabel("# questions that flipped"); ax1.set_ylim(0, max(bs + cs) + 4)
ax1.set_title("McNemar disagreement cells: CUDA replication vs original MLX\n"
              "b ≈ c in every run → no detectable precision effect", fontsize=11.5)
ax1.legend(fontsize=8.5, loc="upper right"); ax1.grid(axis="y", alpha=.3)

# Panel 3: accuracy diff with bootstrap CI (the null view)
ax2 = ax[1, 0]
for i, (nm, rr) in enumerate(runs):
    col = "tab:blue" if i == 0 else "tab:gray"
    ax2.errorbar(rr["diff"], i, xerr=[[rr["diff"] - rr["lo"]], [rr["hi"] - rr["diff"]]],
                 fmt="o", color=col, capsize=5, lw=2, ms=8)
    ax2.text(rr["hi"] + 0.012, i, f"{rr['diff']:+.2f}  [{rr['lo']:+.2f}, {rr['hi']:+.2f}]", va="center", fontsize=9)
ax2.axvline(0, color="k", lw=1.2, ls="--")
ax2.set_yticks(list(range(len(runs)))); ax2.set_yticklabels([n.replace("\n", " ") for n in names], fontsize=9)
ax2.set_xlim(-0.2, 0.3); ax2.invert_yaxis()
ax2.set_xlabel("INT4 − FP16 accuracy difference (95% bootstrap CI)")
ax2.set_title("Every CI straddles zero — replicated null\nCUDA (blue) lands inside the MLX range", fontsize=11.5)
ax2.grid(axis="x", alpha=.3)

# Panel 4: scale experiment
ax3 = ax[1, 1]
sizes = [s for s, _ in scale]; gaps = [g for _, g in scale]
ax3.plot(sizes, gaps, "o-", color="tab:purple", lw=2.5, ms=10)
for s, g in scale:
    ax3.annotate(f"{g:.3f}", (s, g), textcoords="offset points", xytext=(8, 6), fontsize=10)
ax3.set_xlabel("model size (B params) — Qwen2.5 family, ALL at INT4")
ax3.set_ylabel("memorization GAP (memorized − control reconstruction)")
ax3.set_title("Controlled scale experiment: precision fixed, only size varies\n"
              f"GAP rises {gaps[0]:.3f} → {gaps[-1]:.3f} (~{gaps[-1]/gaps[0]:.0f}×) — scale drives memorization",
              fontsize=11.5)
ax3.set_xticks(sizes); ax3.grid(alpha=.3); ax3.set_ylim(0, max(gaps) * 1.25)

plt.suptitle("Quantization × Memorization/Factuality — NVIDIA CUDA replication backing the MLX study\n"
             "All values read from committed results/analysis_*.json", fontsize=13, y=1.0)
plt.tight_layout()
out = os.path.join(ROOT, "docs", "cuda_replication.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
