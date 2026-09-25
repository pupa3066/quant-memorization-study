# Aligned Research Topics: Data-Centric Interpretability & Reliability

> Grounded in three research pillars (data-centric interpretability, reliable information synthesis,
> AI creativity) and in apparatus already running. Each topic lists: the pillar it serves, the
> concrete hypothesis, the method (real, runnable), and why it is NOT a hypothetical bridge.
> Priority = how directly it extends published work in the area + the existing harness.

## Pillar 1 — Data-Centric Interpretability

### T1. Does quantization change what a model memorizes? (ACTIVE; extends prior black-box probing work)
- **Hypothesis:** reconstruction of high-surprisal tokens in likely-memorized text declines with
  precision (FP16→INT8→INT4), while control text stays at floor.
- **Method:** the running harness — token-ID greedy reconstruction, memorized-vs-control GAP,
  paired CI across precisions, multi-model sweep. REAL data in progress.
- **Why real:** directly extends Ravichander et al., 2025 (arXiv:2503.12072) black-box probing to the
  precision axis; we already produce measured GAPs.
- **Next depth:** larger models (stronger FP16 baseline), top-k reconstruction criterion, per-genre
  memorization (poetry vs prose vs code), and whether int8 preserves memorization that int4 erases.

### T2. Quantization × training-data provenance signal
- **Hypothesis:** if quantization erases memorization, it also weakens membership/contamination
  signals — measurable as reduced separability between seen and unseen text.
- **Method:** compute the memorized-vs-control separability (ROC/AUC of reconstruction score) at each
  precision; test whether AUC degrades with compression.
- **Why real:** membership inference and contamination detection are core data-centric-interpretability
  tools; this asks whether efficiency deployment quietly undermines them.

## Pillar 3 — Reliable Information Synthesis (agents)

### T3. Context tiering × task reliability for agents (ACTIVE — kiro/consistent-context-kit)
- **Hypothesis:** access-pattern-tiered context is non-inferior to monolithic context on task success
  at lower token cost (constructive counter to ETH arXiv:2602.11988).
- **Method:** existing SWE-bench harness, C0–C3 conditions, McNemar/CMH/permutation/non-inferiority.
- **Why real:** apparatus built; extends a published negative result.

### T4. Does quantization degrade an agent's synthesis reliability unevenly?
- **Hypothesis:** INT4 leaves aggregate task success ~unchanged (consistent with our factuality null)
  but degrades multi-step/long-context synthesis more than single-fact recall.
- **Method:** run the tiered-context agent harness with the model at FP16 vs INT4; compare success by
  task complexity (single-hop vs multi-hop). Combines both projects' harnesses.
- **Why real:** unites the two measured results — factuality robust to INT4 + tiered context — into
  one reliability question. This is the "two theses backing each other" made testable.

## Pillar 2 — AI Creativity (lower priority; honest fit note)
### T5. Precision × output diversity/novelty
- **Hypothesis:** quantization narrows output distribution (lower diversity), measurable via distinct-n
  / self-BLEU on open-ended generation.
- **Method:** generate open-ended completions at FP16 vs INT4, measure lexical/semantic diversity.
- **Why real but secondary:** creativity is the third pillar, and "does compression flatten a model
  toward its training-distribution mode?" is a genuine measurable question, but it's the least
  developed of the apparatus, so flag it as exploratory, not a headline.

## Cross-thesis map (why these reinforce each other — MEASURED, not hypothetical)
- T1/T2 (memorization erosion under precision) + T4 (agent synthesis under precision) share ONE
  independent variable (precision) and ONE principle: *compression changes some behaviors and not
  others; measure which.* Our factuality null [MEASURED p=1.0] + memorization decline [MEASURED,
  directional] are the seed data both build on.
- T3 (context axis) + T1 (precision axis) are the same "spend the expensive resource only where it
  changes behavior" thesis on two axes — already operationalized by precision_advisor.py consuming
  the study's real analysis JSON.

## Strongest topics to lead with
T1 (have data) + T4 (unites both projects) → strongest, most defensible, least hand-wavy.
