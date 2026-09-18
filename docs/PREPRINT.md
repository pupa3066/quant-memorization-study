# Does Quantization Change What Language Models Memorize and Recall? A Multi-Model Study

**Purnima Pathak**
Independent Researcher · purnima.pathak19@gmail.com · ORCID 0009-0000-6441-7962
DOI: 10.6084/m9.figshare.33858859 · Code: https://github.com/pupa3066/quant-memorization-study

---

## Abstract
Low-bit quantization (INT8/INT4) is widely deployed to fit language models on constrained hardware.
Its effect on task accuracy is well studied; its effect on *what a model memorizes* and *how reliably
it recalls facts* is less so. Treating precision as an independent variable and model behavior as the
dependent variable, we measure both across **six models** (0.5B–3B; Qwen2.5, Llama-3.2, Phi-3.5,
Gemma-2) on Apple Silicon. We find: **(1) factual accuracy is robust to INT4** — a replicated null
across every model with paired data (McNemar p = 1.0, 1.0, 0.38); **(2) memorization is dominated by
model scale, not precision** — a clean "INT4 erases memorization" effect seen on a single 0.5B model
did **not** replicate across models. We report the finding honestly, including a corrected pilot false
positive, and outline the experiment (larger models) that would resolve the memorization question.

## 1. Introduction
Quantization is a deployment necessity, not a research curiosity: it is how models run on phones,
laptops, and edge devices. Yet efficiency interventions may quietly change *behavior* beyond accuracy.
Building on black-box memorization probing (Ravichander et al., 2025, arXiv:2503.12072), we ask:

- **RQ1:** Does INT4 quantization change closed-book factual accuracy?
- **RQ2:** Does quantization change verbatim memorization (reconstruction of high-surprisal tokens)?

## 2. Method
**Independent variable:** precision of the *same* model weights {FP16/bf16, INT8, INT4} (MLX variants),
holding prompts, decoding (temp=0), and seed fixed.

**Factuality probe.** Closed-book QA on PopQA, 100 items, popularity-stratified by subject Wikipedia
pageviews; alias-aware exact-match scoring. Paired McNemar (INT4 vs FP16) + bootstrap CIs.

**Memorization probe.** 20 public-domain memorized-candidate passages + 20 length-matched synthetic
controls. For the highest-surprisal token positions (scored by the model itself), we feed the exact
prefix and test greedy-argmax reconstruction at the token-ID level. Signal = memorized−control
reconstruction **GAP** (robust to unknown pretraining membership). We also report **AUC** of using
reconstruction score to separate memorized from control passages (a membership/detectability metric).

**Models.** Qwen2.5-0.5B/1.5B/3B, Llama-3.2-1B, Phi-3.5-mini, Gemma-2-2b. FP16+INT4 paired where both
MLX variants exist; larger models INT4-only (8GB constraint).

## 3. Results
### 3.1 Factuality — replicated null
| Model | FP16 acc | INT4 acc | McNemar p | acc-diff 95% CI |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.29 | 0.28 | 1.0 | [−0.10, 0.08] |
| Qwen2.5-1.5B | 0.27 | 0.22 | 0.38 | [−0.14, 0.04] |
| Llama-3.2-1B | 0.29 | 0.28 | 1.0 | [−0.10, 0.07] |

Across all paired models, INT4 does not significantly change factual accuracy; every CI straddles 0.

### 3.2 Memorization — dominated by scale, not precision
| Model | FP16 GAP | INT8 GAP | INT4 GAP |
|---|---|---|---|
| Qwen2.5-0.5B | 0.025 | 0.017 | 0.000 |
| Qwen2.5-1.5B | 0.060 | — | 0.033 |
| Llama-3.2-1B | 0.065 | — | 0.074 |
| Gemma-2-2b | — | — | 0.045 |
| Phi-3.5-mini | — | — | 0.088 |
| Qwen2.5-3B | — | — | 0.105 |

The 0.5B model shows a clean monotonic decline (0.025→0.017→0.000), but this does **not** generalize:
Llama-3.2-1B's INT4 GAP (0.074) exceeds its FP16 GAP (0.065), and INT4-only larger models memorize
*more* (Qwen2.5-3B 0.105, Phi-3.5-mini 0.088) than small FP16 models. Detectability AUC rises with
scale (0.5B 0.55 → 3B 0.625). **Scale drives memorization; INT4's effect is small and inconsistent.**

## 4. Discussion & honest limitations
- The tidy single-model "INT4 erases memorization" result was a **small-model artifact**; multi-model
  replication corrected it. A separate N=8 factuality pilot false positive was likewise corrected by
  scaling to N=100. We report these corrections deliberately.
- Small models (≤3B) memorize weakly, so FP16 memorization baselines are low; the **decisive test is a
  larger model** (7B–70B) where the FP16 memorization signal is strong enough to see whether INT4
  collapses it. This is the primary open question (§6).
- Scope: English, MLX quantization, ≤3B, single decoding sample. Contamination is unknown, hence the
  memorized−control GAP rather than absolute rates.

## 5. Related work
Black-box memorization probing (Ravichander et al., 2025, arXiv:2503.12072). Quantization tradeoff literature focuses on
accuracy/perplexity; we add memorization/factuality as dependent variables. A cross-modality check
(diffusion INT4, cosine >0.997 image fidelity) corroborates the "INT4 preserves task output" side.

## 6. Future work (and where compute is the bottleneck)
1. **Scale the memorization test to 7B–70B models** (strong FP16 baseline) — needs GPU/cluster memory
   beyond an 8GB laptop. This would turn the memorization question from inconclusive to decisive.
2. **Per-domain / per-layer memorization** under quantization (poetry vs prose vs code; which layers).
3. **INT8 midpoints for all models** + additional quantization schemes (group size, NF4).

## 7. Reproducibility
Pre-registered design, RAM/disk-safe multi-model sweep, from-scratch statistics (McNemar, bootstrap
CIs, AUC). All run logs and per-model analyses are in the repository. DOI: 10.6084/m9.figshare.33858859.

## 8. Provenance — how this research topic arose
This study did not begin as a research question; it began as an **engineering constraint** in a
companion project (AnimeVlog), and the reframe from engineering to science is the reason it exists.

- **Origin (engineering):** AnimeVlog generates personalized anime fully on-device on consumer Apple
  Silicon, including 8GB machines. SDXL's UNet (2.57B params, ~4897MB fp16) does not fit alongside the
  OS on 8GB, forcing INT4 quantization. A pure-PyTorch per-group INT4 quantizer for MPS was built and
  measured: fp16 4897MB → int4 ~1759MB (3.8× at group_size=128), cosine 0.99778–0.99799, with a
  layer-sensitivity result (skip first/last → 0.99997) and NF4 measured *worse* than uniform
  (per-group calibration already handled the distribution). [MEASURED 2026-08-13]
- **The honest pivot:** AnimeVlog's own assessment concluded this INT4 work was *not* a publishable
  research contribution ("per-group asymmetric quantization is textbook; an engineering workaround").
  That honest self-assessment redirected the effort: the quantization *engineering* measured only
  **output fidelity** (cosine of images/weights). The unstudied question was what the same efficiency
  knob does to **model behavior** — memorization and factual recall. That question is this study:
  precision as the independent variable, behavior as the dependent variable.
- **Cross-modality relationship (recorded, measured):** AnimeVlog INT4 [vision: cosine >0.997 image
  fidelity] **SUPPORTS** this study [text: INT4 factuality null] on the claim "INT4 preserves the
  intended task output across modalities"; **NEUTRAL** on memorization (image fidelity ≠ verbatim
  recall). The vision result gives this text study cross-modality external validity. Methodologically,
  AnimeVlog's group-size / layer-sensitivity ablation design is transferable to probing how
  quantization granularity affects memorization (a future axis, §6).
- **Shared principle:** both lines, and the companion Consistent Context Kit, operationalize one
  measured thesis — *spend the expensive resource only where it changes behavior* (bits here; context
  in the kit). The kit's `precision_advisor.py` consumes this study's real `analysis_*.json`.

## 9. Contributions & cross-hardware provenance
- Original study (Apple Silicon / MLX): Purnima Pathak (ORCID 0009-0000-6441-7962).
- NVIDIA CUDA / RTX 5060 cross-hardware replication (v1.1.0): Akshay Upadhyay
  (ORCID 0009-0000-1531-3198), contributed via PR #1 and merged after author + provenance
  confirmation. The CUDA numbers came from the same measurement harness (`--backend hf`), additive to
  the MLX path (same probes and analysis code); see `docs/CUDA_REPLICATION.md`. The factuality null
  and the scale-drives-memorization finding both replicate on the second hardware family.
