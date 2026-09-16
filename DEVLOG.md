# DEVLOG — Quantization × Memorization/Factuality

A running, honest engineering + science log. Records what broke, what caused it, what the fix was,
and WHY the design is currently shaped the way it is. Measured facts only; pilot numbers are labeled.

---

## 0. Premise (why this study exists)
Bridge between hardware-efficiency background (on-device INT4/INT8 quantization) and NLP
data-centric interpretability (Ravichander et al., arXiv:2503.12072 — detecting memorized training
data in black-box LLMs via high-surprisal token reconstruction).

Question: **does quantization (FP16→INT4) measurably change what a model memorizes and how reliably
it answers factual questions?** Precision = independent variable; model behavior = dependent variable.

---

## 1. Environment setup — what led to what

### Bug 1: `pip install` refused — externally-managed environment (PEP 668)
- **Symptom:** `pip install numpy mlx mlx-lm` exited 1 with a PEP 668 note; nothing installed.
- **Cause:** system Python 3.13 on macOS is externally-managed; pip blocks global installs to avoid
  breaking the OS Python.
- **Wrong fix (rejected):** `--break-system-packages` — risks the system Python. Non-reversible-ish.
- **Fix applied:** isolated `python3 -m venv .venv`, install into it. Non-destructive.
- **Design consequence:** the project ships a venv-based workflow; `.venv/` is gitignored (425 MB).
  All run commands use `./.venv/bin/python`.

### Model choice
- 8 GB RAM machine → smallest capable model. Chose **Qwen2.5-0.5B-Instruct** with two real precisions
  from mlx-community: `-bf16` (full-precision proxy, labeled fp16) and `-4bit` (INT4).
- Both load and generate correctly (smoke test: "capital of France" → "Paris" at both precisions).
- **Design consequence:** "fp16" in the code/results means the bf16 checkpoint; INT4 is the 4-bit
  MLX quantization. Same base model, so precision is the only variable — the required control.

---

## 2. The memorization probe — the important bug and its fix

### Bug 2 (SCIENTIFIC BUG): first probe measured nothing → all-zeros artifact
- **Symptom:** first run gave mem reconstruction = 0/4 for EVERY item at both precisions, including
  "It was the best of times" and "Call me Ishmael" (text the model almost certainly memorized).
  A uniform 0.0 is a tell that the *instrument*, not the model, is the problem.
- **Cause (root):** the probe rebuilt a text prefix by decoding token strings and re-joining with
  spaces (`" ".join(tokens)`), then did FREE generation from that lossy prefix and string-matched.
  Two failures: (a) detokenize→retokenize is lossy and misaligns positions; (b) free generation asks
  "continue the text" not "predict THIS masked token", so it never tested reconstruction.
  Evidence in old rows: target `'was'` → model produced `'A. I'` (garbage prefix).
- **Fix applied:** rewrote to operate at the **token-ID level**:
  1. `token_surprisals_ids(ids)` — one forward pass, per-position surprisal = −logp of the actual
     next token, aligned exactly to `ids[1:]`.
  2. Select top-k highest-surprisal positions (the informative tokens, per the paper's paradigm).
  3. For each such position `t`, feed the model the EXACT prefix `ids[:t]` and take greedy
     `argmax` next-token id (`predict_next_id`). Score = fraction where prediction == `ids[t]`.
  This is a proper deterministic reconstruction test; no lossy re-tokenization, no free generation.
- **Design consequence:** the backend now exposes `encode / token_surprisals_ids / predict_next_id /
  decode` instead of a `reconstruct(prefix, n)` string method. A numeric `score` field
  (reconstruction fraction) was added to `Result` so the analysis can average it, not just count hits.

### Why the design uses the memorized-vs-control GAP (not absolute rate)
- We cannot *prove* a passage was in pretraining. So absolute reconstruction is uninterpretable.
- Control passages are synthetic nonsense that CANNOT be memorized verbatim. The **GAP**
  (memorized recon − control recon) is the memorization signal, robust to the contamination-unknown
  threat. This is a deliberate validity choice (DESIGN.md §8).

---

## 3. The factuality probe — the ceiling-effect bug

### Bug 3 (DESIGN BUG): facts too easy → both precisions scored 100% → no measurable effect
- **Symptom:** first QA set (Paris, Shakespeare, Bhutan, transistor) → 4/4 at BOTH precisions.
  McNemar b=c=0. You literally cannot see a quantization effect if both are at ceiling.
- **Cause:** all facts were high-popularity; a 0.5B model already knows them, so precision can't
  differentiate.
- **Fix applied:** added HARD long-tail facts (francium discoverer, Kazakhstan capital, Treaty of
  Westphalia year, tungsten atomic number) and a `popularity` label (high|low) per item, so the
  analysis stratifies. Long-tail is where quantization degradation is expected to appear (H3).
- **Design consequence:** QA set is popularity-stratified by design; analysis reports high-pop vs
  low-pop separately. This is what surfaced the real effect (below).

---

## 4. Current pilot result (N=8 each — PILOT, not a finding)
After fixes, real run on real models (see runs.jsonl / analysis.json):
- Memorization GAP: **fp16 = +0.10, int4 = 0.00** — the memorization signal present at fp16
  DISAPPEARS at int4 (H1 direction). int4 emitted junk token `'ero'` at masked positions.
- Factuality: **fp16 0.75 → int4 0.50 overall**; high-pop **1.0 → 1.0** (unaffected), low-pop
  **0.67 → 0.33** (degraded). Degradation concentrated in the long tail (H3 direction).
- Significance: McNemar b=2 c=0, **p=0.48 — NOT significant** at N=8. Direction is consistent;
  power is not there. This is a working apparatus + a promising pilot, nothing more.

### Honesty caveats (recorded so no one over-reads this)
- N=8 is tiny; p=0.48. No claim of significance.
- 0.5B model memorizes weakly, so even fp16 gap is small (0.10). Bigger models → larger gap.
- Greedy argmax reconstruction is strict; a top-k reconstruction criterion would raise both rates.

---

## 5. Roadmap to a real result (why current design is a foundation, not the end)
1. Scale N: 200–500 memorization passages (public-domain + post-cutoff controls) + a real
   popularity-labeled QA set (e.g. PopQA). → statistical power.
2. Bigger model + add INT8 middle point (FP16→INT8→INT4 trend the hypotheses predict).
3. Re-run analysis.py for powered McNemar / CI / gap.

## 6. Design invariants (do not regress)
- Same base model across precisions (precision is the ONLY variable).
- Token-ID-level probing (never detokenize→retokenize for scoring).
- Report GAP for memorization, popularity-stratified accuracy for factuality.
- Never fabricate: no backend → `--dry` prints plan and exits; empty data → analysis returns zeros,
  not invented numbers.
