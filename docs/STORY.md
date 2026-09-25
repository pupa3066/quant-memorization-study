# The Story: Why This Research Exists and How It Connects

> Narrative for reviewers. Explains the WHY, the WHAT, the requirement it answers, and how this work
> and the Consistent Context Kit research mutually back each other. Measured claims only.

## Why (the motivation)
Quantization (INT8/INT4) is deployed everywhere for on-device and edge inference — it's how models
fit on constrained hardware. The efficiency tradeoff on *accuracy* is well studied. The tradeoff on
*what a model memorizes and how reliably it behaves* is not. If compression quietly erases (or
preserves) memorized training data, that matters for copyright, privacy, and contamination auditing —
exactly the questions data-centric interpretability asks. So: **use the efficiency knob as an
independent variable, and measure model behavior as the dependent variable.**

## What (the actual contribution)
1. A reproducible apparatus that measures, on real models across precisions:
   - **Memorization** via high-surprisal token reconstruction (extends black-box probing,
     Ravichander et al., 2025, arXiv:2503.12072) — reported as a memorized-vs-control GAP and as an
     **AUC detectability** metric (membership/contamination framing).
   - **Factuality** via popularity-stratified closed-book QA (PopQA).
2. Real statistics from scratch (McNemar, bootstrap CIs, paired designs, AUC).
3. A multi-model sweep (multiple architectures/sizes) so findings aren't single-model artifacts.

## The requirement it answers (for a reliability-focused reviewer)
"Develop a scientific understanding of how AI models work, to improve reliability." This work asks a
precise, measurable sub-question of that: *which behaviors survive compression and which don't?* It
produces falsifiable, honestly-scoped answers (including a null and honestly-labeled underpowered
results) rather than leaderboard numbers.

## How it connects to — and defends — the Consistent Context Kit thesis
The context kit's thesis: **spend the expensive resource only where it changes behavior** (load
context by access pattern; don't pay for context that doesn't change the outcome).

This study is the *same thesis on a different axis*: spend precision only where it changes behavior.
- MEASURED support: factuality is robust to INT4 (null, p=1.0) ⇒ for factual-recall workloads the
  expensive resource (bits) can be cut safely — analogous to the kit's finding that monolithic
  always-on context is wasteful.
- MEASURED support: memorization/detectability erodes with precision ⇒ for verbatim-recall workloads
  the resource must be preserved — analogous to loading the specific context a task actually needs.
- OPERATIONAL link (not hypothetical): the kit's `precision_advisor.py` READS this study's real
  `analysis_*.json` and emits a workload-aware recommendation. Two projects, one measured principle,
  wired together in code.

## How it defends other theses (cross-validation)
Each new MEASURED result here is checked against other projects' claims (SUPPORTS/CONTRADICTS/REFINES).
- Factuality null [MEASURED] SUPPORTS the context-kit principle (cut cost where behavior is unchanged).
- If a larger model later shows a strong memorization GAP that INT4 collapses, that REFINES the
  precision-advisor's guidance (raise the precision floor for verbatim workloads) — and that update
  flows back into the kit. The two research lines are designed to correct each other with data.

## Honesty ledger (what is NOT claimed)
- Small models (0.5B) memorize weakly → memorization effects are directional, not conclusive; the
  multi-model sweep with larger models is the test of whether they hold.
- A pilot false positive (N=8 factuality "drop") was caught and corrected by scaling N. Documented,
  not hidden.
