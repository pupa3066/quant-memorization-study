# RESULTS — The Effect of Low-Bit Quantization on Memorization and Factual Recall in Language Models

> All numbers below are MEASURED on real models (MLX Qwen2.5-0.5B-Instruct at bf16/int8/int4).
> Data files: `runs_popqa.jsonl` (factuality), `runs_mem.jsonl` (memorization).
> Reproduce: see README.md. Honest status: small-model study; findings scoped accordingly.

## TL;DR
1. **Factuality — NULL result (well-powered).** INT4 is statistically indistinguishable from FP16 on
   aggregate closed-book factual accuracy (N=100 real PopQA facts). p=1.0.
2. **Memorization — directional, underpowered.** The memorization-reconstruction signal declines
   monotonically FP16 → INT8 → INT4 (N=40 passages, 3 precisions). Direction matches the hypothesis,
   but the effect is small because a 0.5B model memorizes weakly; the CI upper bound touches 0.
3. **Methodological result.** A pilot (N=8) suggested INT4 hurt factuality; scaling to N=100 removed
   that effect entirely. The pilot was a false positive. Scaling corrected it.

## 1. Factuality (closed-book QA)
Dataset: PopQA (akariasai/PopQA), 100 items, popularity-binned by subject Wikipedia pageviews
(`s_pop`), alias-aware scoring. Same base model at two precisions.

| Precision | Overall acc | High-pop | Low-pop |
|---|---|---|---|
| fp16 (bf16) | 0.29 | 0.48 | 0.10 |
| int4        | 0.28 | 0.36 | 0.20 |

- McNemar (int4 vs fp16): **b=11, c=10, χ²=0.0, p=1.0** — disagreements cancel in both directions.
- Accuracy-difference bootstrap CI: **−0.01 [−0.10, +0.08]** — tight, straddles zero.
- **Interpretation:** no significant aggregate effect of INT4 quantization on factual accuracy.
  The high-pop 0.48→0.36 hints at some degradation but low-pop is at the model's knowledge floor
  (0.10–0.20), so the popularity split is inconclusive. **H3 is NOT supported by this data.**

## 2. Memorization (high-surprisal reconstruction)
Corpus: 20 public-domain famous passages (memorized candidates) + 20 synthetic controls
(`data/mem_corpus.json`). Probe: mask top-surprisal tokens, test greedy-argmax reconstruction at the
token-ID level. Signal = GAP (memorized recon − control recon). Three precisions.

| Precision | Recon (memorized) | Recon (control) | GAP |
|---|---|---|---|
| fp16 (bf16) | 0.025 | 0.000 | **0.025** |
| int8        | 0.017 | 0.000 | **0.017** |
| int4        | 0.000 | 0.000 | **0.000** |

- Paired reconstruction-score difference vs fp16 (memorized items, N=20):
  - int4 − fp16: **−0.025, CI [−0.067, 0.0]**
  - int8 − fp16: **−0.008, CI [−0.025, 0.0]**
- **Interpretation:** the memorization signal declines monotonically with precision and vanishes at
  INT4 — the hypothesized direction (H1). BUT the FP16 baseline gap is tiny (0.025) because a 0.5B
  model barely memorizes under a strict greedy criterion, and both CIs' upper bounds touch 0.
  **This is directionally consistent and suggestive, NOT statistically conclusive.**

## 3. Threats / honest caveats
- **Model scale:** 0.5B memorizes weakly → small FP16 baseline → limited power on the memorization side.
  The single most important next step is repeating on a 3B–7B model where the FP16 memorization gap
  is large enough to measure an INT4 collapse conclusively.
- **Contamination unknown:** we cannot prove a passage was in pretraining → we report the
  memorized-vs-control GAP, not absolute rates.
- **Strict scoring:** greedy-argmax exact reconstruction is conservative; a top-k criterion would
  raise all rates and possibly the effect size.
- **Small QA subset artifact:** on the 8-item default QA set, int8 scored highest (0.875) — a
  non-monotonic small-N artifact; do not over-read. The 100-item PopQA result is the reliable one.

## 4. What this establishes
- A **working, reproducible apparatus** for measuring efficiency-intervention effects on both
  memorization and factuality, with real statistics (McNemar, bootstrap CIs, paired designs).
- A **defensible null** on factuality and a **directional, honestly-underpowered** memorization signal.
- A **clear, scoped next experiment** (larger model, scaled memorization corpus, INT8 midpoint) that
  would turn the memorization direction into a conclusive result.

## 5. Alignment
This is a data-centric-interpretability question: using an efficiency knob (quantization) as the
independent variable and model behavior (memorization, factuality) as the dependent variable. It
extends black-box memorization probing (Ravichander et al., arXiv:2503.12072) to the precision axis.
