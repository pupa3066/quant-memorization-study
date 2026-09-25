# The Effect of Low-Bit Quantization on Memorization and Factual Recall in Language Models — Pre-Registered Study Design

**Status:** design v1 (2026-09-15). Pre-registration-style; results filled after real runs.
**Research area / alignment:** Data-Centric Interpretability (how training data shapes behavior) +
reliability (why factuality fails).

## 1. Motivation & prior result
Ravichander et al., 2025 (arXiv:2503.12072, ACL Outstanding Paper) detect training-data memorization in
black-box LLMs via **information-guided probing**: mask high-surprisal tokens in a passage and test
whether the model can reconstruct them. High reconstruction ⇒ the passage is memorized.

Efficiency interventions (INT8/INT4 quantization) are deployed widely for edge/on-device inference.
Their effect on *memorization* and *factuality* is under-characterized.

## 2. Research questions (falsifiable)
- **RQ1 (memorization):** Does quantization (FP16 → INT8 → INT4) measurably change a model's ability
  to reconstruct high-surprisal tokens in likely-memorized text, vs full precision?
- **RQ2 (factuality):** Does quantization degrade closed-book factual-QA accuracy, and is the
  degradation *uneven* across fact types/popularity?

## 3. Hypotheses
- **H1:** INT4 reduces high-surprisal reconstruction rate vs FP16 (memorization partially erased by
  quantization noise). Directional; could also be null (memorization robust) — a null is publishable.
- **H2:** Factual-QA accuracy at INT8 is non-inferior to FP16 (margin δ=3pp); INT4 shows a measurable drop.
- **H3:** Any degradation is larger for low-frequency / long-tail facts than for high-frequency facts.

## 4. Independent variable
Precision level of the SAME model weights: {FP16, INT8, INT4}. Nothing else changes (same prompts,
same decoding temp=0, same seed). This isolates precision as the cause.

## 5. Materials
- **Model:** a small open LLM that fits 8GB (e.g. a 0.5B–1.5B instruct or its quantized MLX variants).
  Pin exact model + revision in results.
- **Memorization probe set:** passages plausibly in pretraining (public-domain / widely-quoted text)
  + control passages (post-cutoff or synthetic) that should NOT be memorized.
- **Factuality set:** a small closed-book factual-QA set with a popularity/frequency label per item.

## 6. Metrics (dependent variables)
- Memorization: **high-surprisal token reconstruction rate** (exact-match on masked high-surprisal
  tokens), per precision. Report memorized-set vs control-set gap (the real signal).
- Factuality: exact/normalized-match accuracy per precision; stratified by popularity bin.
- Report per-precision means + bootstrap CIs; paired by item across precisions.

## 7. Statistics
- Paired across precision levels on the same items. McNemar (paired binary correct/incorrect) for
  FP16-vs-INT8 and FP16-vs-INT4. Bootstrap CIs for rate differences. Non-inferiority test for H2 (δ=3pp).
- Stratified analysis (popularity bins) for H3. Holm correction across H1–H3.
- Report effect sizes + CIs, not just p-values. Pre-register δ, α before running.

## 8. Threats to validity
- **Contamination unknown:** we can't be certain a passage was in pretraining → use the
  memorized-vs-control GAP, not absolute rates, as the signal.
- **Quantization method confound:** report the exact quantization scheme (group size, scheme) —
  reuse measured settings from apple-ml/AnimeVlog int4 logs for provenance.
- **Small model / small N:** frame as PRELIMINARY evidence + a scalable harness, not a final claim.
- **Decoding nondeterminism:** temp=0, fixed seed, single sample; note residual nondeterminism.

## 9. Novelty / honest positioning
Not new: quantization; black-box memorization probing (Ravichander et al., 2025, arXiv:2503.12072).
New here: **measuring memorization & factuality as a function of the precision knob**, using that
probing paradigm, with my on-device quantization tooling as the experimental lever. Framed as a
data-centric-interpretability question, not a hardware one.

## 10. Reproducibility
Harness (`harness/`) logs full config, model revision, precision, seeds, per-item results to JSONL.
`analysis.py` recomputes all stats. This doc is the pre-registration; do not alter hypotheses post-hoc.

## 11. HONEST STATUS
No real numbers yet. Any file claiming results must be produced by an actual run on a real model.
Fabricated numbers are forbidden (rules: measured facts only; label [MEASURED]/[CLAIM]).
