# Quantization × Memorization & Factuality

Does reducing a model's numeric precision (FP16 → INT4) change **what it memorizes** and **how
reliably it answers factual questions**? A small, honest, reproducible study bridging on-device
quantization with data-centric interpretability (Ravichander et al., arXiv:2503.12072).

> **Status: PILOT.** The apparatus runs end-to-end on real models and shows a directionally
> consistent effect, but N is tiny and results are NOT statistically significant. See DEVLOG.md §4.

## Files
- `DESIGN.md` — pre-registered design: RQs, hypotheses (H1–H3), metrics, statistics, threats.
- `harness.py` — measurement harness. Memorization probe (token-ID-level high-surprisal
  reconstruction) + factuality probe (popularity-stratified closed-book QA). MLX backend.
- `analysis.py` — per-precision metrics: memorization GAP, factuality accuracy by popularity,
  paired McNemar, bootstrap CIs.
- `DEVLOG.md` — deep engineering log: every bug, its root cause, the fix, and design rationale.
- `runs.jsonl` / `analysis.json` — the current pilot data + computed metrics (real, labeled pilot).

## Reproduce (Apple Silicon)
```sh
python3 -m venv .venv
./.venv/bin/python -m pip install numpy mlx mlx-lm
# dry plan (no model, no cost):
./.venv/bin/python harness.py --dry
# real run (downloads two precisions of the same small model):
./.venv/bin/python harness.py --run \
  --fp16 mlx-community/Qwen2.5-0.5B-Instruct-bf16 \
  --int4 mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --out runs.jsonl
./.venv/bin/python analysis.py runs.jsonl
```

## Current results
**Pilot (N=8, `runs.jsonl`):** suggested int4 hurt factuality — but this was small-sample noise.
**Scaled (N=100 real PopQA facts, `runs_popqa.jsonl`):**
- Factuality: fp16 **0.29** vs int4 **0.28**; McNemar p=**1.0**, diff CI **[-0.10, +0.08]** →
  **INT4 statistically indistinguishable from FP16** on aggregate factual accuracy (a NULL result).
  Scaling N corrected the pilot's false positive.
- Memorization (still N=8, underpowered): fp16 GAP **+0.10** → int4 **0.00** — suggestive that
  int4 erases weak memorization; needs a scaled memorization corpus + bigger model to confirm.

Run with `--popqa 50` to reproduce the scaled QA result. See DEVLOG.md §5b.

## Honesty policy
No fabricated numbers. No backend → prints plan and exits. Empty data → zeros, not invented results.
