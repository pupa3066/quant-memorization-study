# Multi-Model Results — The Effect of Low-Bit Quantization on Memorization and Factual Recall in Language Models

> Auto-generated from per-model analysis_*.json by combine_results.py.
> All numbers MEASURED on real models. Missing cells = precision variant not run.

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
