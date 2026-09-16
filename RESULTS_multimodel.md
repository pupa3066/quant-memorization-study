# Multi-Model Results — Quantization × Memorization/Factuality

> Auto-generated tables from per-model analysis_*.json. All numbers MEASURED on 6 real models
> (Qwen2.5-0.5B/1.5B/3B, Llama-3.2-1B, Phi-3.5-mini, Gemma-2-2b) via MLX. Missing cells = that
> precision variant was not run (only 4-bit available on HF for the 3B/mini models on 8GB).

## HONEST CONCLUSION (what 6 models actually show)
1. **Factuality: INT4 ≈ FP16 — a replicated NULL.** On all 3 models with paired fp16/int4 data
   (qwen05, qwen15, llama1b), INT4 does not significantly change factual accuracy (McNemar p=1.0,
   1.0, 0.38). This is a solid, multi-model null.
2. **Memorization: dominated by MODEL SCALE, not precision — the single-model "int4 erases
   memorization" story does NOT replicate.** Of 3 paired models, 2 decline with int4 (qwen05
   0.025→0.0; qwen15 0.06→0.033) but 1 INCREASES (llama1b 0.065→0.074, verified against raw scores).
   Larger int4-only models memorize MORE (qwen3b GAP 0.105, phi35 0.0875) than small fp16 models —
   i.e. scale drives the memorization signal; int4's effect is small and inconsistent.
3. **Methodological point:** the tidy monotonic decline seen at 0.5B was a small-model artifact.
   Running 6 models corrected it. The defensible claim is the factuality null + "scale dominates
   memorization"; NOT "quantization erases memorization."

## Factuality (PopQA): INT4 vs FP16
| Model | fp16 acc | int4 acc | McNemar p | acc-diff CI |
|---|---|---|---|---|
| gemma2b | None | 0.3 | None | — |
| llama1b | 0.29 | 0.28 | 1.0 | [-0.1, 0.07] |
| phi35 | None | 0.26 | None | — |
| qwen05 | 0.29 | 0.28 | 1.0 | [-0.1, 0.08] |
| qwen15 | 0.27 | 0.22 | 0.3827 | [-0.14, 0.04] |
| qwen3b | None | 0.29 | None | — |

## Memorization GAP by precision (memorized − control reconstruction)
| Model | fp16 GAP | int8 GAP | int4 GAP |
|---|---|---|---|
| gemma2b | None | None | 0.045 |
| llama1b | 0.0648 | None | 0.0736 |
| phi35 | None | None | 0.0875 |
| qwen05 | 0.025 | 0.0167 | 0.0 |
| qwen15 | 0.06 | None | 0.0333 |
| qwen3b | None | None | 0.105 |

## Memorization DETECTABILITY: AUC (memorized vs control) by precision
AUC 0.5 = memorization undetectable by the probe; higher = more detectable.
| Model | fp16 AUC | int8 AUC | int4 AUC |
|---|---|---|---|
| gemma2b | None | None | 0.575 |
| llama1b | 0.575 | None | 0.625 |
| phi35 | None | None | 0.6 |
| qwen05 | 0.55 | 0.525 | 0.5 |
| qwen15 | 0.575 | None | 0.55 |
| qwen3b | None | None | 0.625 |

## How to read this
- Factuality: p≈1.0 with a CI straddling 0 ⇒ INT4 indistinguishable from FP16 (null).
- Memorization: look for a monotonic fp16≥int8≥int4 decline AND a nonzero fp16 baseline.
  A larger fp16 GAP (bigger/more-memorizing model) makes any INT4 collapse more conclusive.
- Effect sizes + intervals matter more than any single point estimate. Underpowered rows
  (tiny fp16 GAP) are suggestive only.
