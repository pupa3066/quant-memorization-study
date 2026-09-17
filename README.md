# Quantization × Memorization & Factuality

[![DOI](https://img.shields.io/badge/DOI-10.6084%2Fm9.figshare.33858859-blue)](https://doi.org/10.6084/m9.figshare.33858859)

Does reducing a model's numeric precision (FP16 → INT8 → INT4) change **what it memorizes** and **how
reliably it answers factual questions**? A reproducible, honestly-scoped study across **6 models**,
bridging on-device quantization with data-centric interpretability (Ravichander et al., arXiv:2503.12072).

> **Headline (6 models):** INT4 leaves factual accuracy statistically unchanged (replicated null),
> while memorization is driven by **model scale, not precision** — a single-model "INT4 erases
> memorization" result did *not* replicate. A pilot false positive was caught and corrected by
> scaling. See `RESULTS_multimodel.md`.

## Repository structure
```
README.md               ← you are here (entry point)
RESULTS_multimodel.md   ← the canonical headline results (6 models)
src/                    ← all code
  harness.py            memorization + factuality probes (MLX backend)
  analysis.py           per-precision metrics: GAP, accuracy, McNemar, bootstrap CIs
  separability.py       memorization detectability (AUC)
  efficiency.py         systems metrics: weight footprint, latency, throughput
  tradeoff.py           joins behavior × efficiency
  combine_results.py    aggregates per-model analyses → RESULTS_multimodel.md
  overnight_sweep.sh    RAM/disk-safe multi-model sweep · launch_overnight.sh  scheduler
results/                ← all run logs (runs_*.jsonl) + computed metrics (*_*.json)
data/                   ← mem_corpus.json (memorized + control passages)
docs/                   ← detailed writeups (see index below)
```
**docs/:** `DESIGN.md` (pre-registration) · `DEVLOG.md` (every bug + fix + rationale) ·
`RESULTS.md` (single-model detail) · `EFFICIENCY.md` (systems tradeoff + kernel-support finding) ·
`EFFICIENCY_animevlog.md` + `CROSSVALIDATION_animevlog.md` (cross-modality) · `PREPRINT.md`
(preliminary writeup) · `ALIGNED_RESEARCH_TOPICS.md` · `STORY.md`.

## Reproduce (Apple Silicon)
```sh
python3 -m venv .venv
./.venv/bin/python -m pip install numpy mlx mlx-lm
./.venv/bin/python src/harness.py --dry                       # plan only, no model, no cost
# real run (downloads two precisions of the same model):
./.venv/bin/python src/harness.py --run \
  --fp16 mlx-community/Qwen2.5-0.5B-Instruct-bf16 \
  --int4 mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --out results/runs.jsonl
./.venv/bin/python src/analysis.py results/runs.jsonl
# full multi-model sweep → RESULTS_multimodel.md:
./.venv/bin/python src/combine_results.py
```

## Current results (6 models: Qwen2.5-0.5B/1.5B/3B, Llama-3.2-1B, Phi-3.5-mini, Gemma-2-2b)
- **Factuality: INT4 ≈ FP16, a replicated null** (McNemar p=1.0, 1.0, 0.38 on paired models).
- **Memorization: dominated by model scale, not precision** — the single-0.5B "int4 erases
  memorization" story did NOT replicate. Honest, mixed result.
- **Efficiency (systems):** INT4 = ~72% smaller weights + 2.4× decode on MLX; but int4 latency
  *depends on fused-kernel support* (MLX fused → faster; PyTorch-MPS → 6.3× slower per-op). See
  `docs/EFFICIENCY.md`.
- **Methodology:** an N=8 factuality pilot false positive and a single-model memorization artifact
  were both corrected by scaling. Measure, don't assert.

Full numbers + how-to-read: `RESULTS_multimodel.md`. Engineering log: `docs/DEVLOG.md`.

## Honesty policy
No fabricated numbers. No backend → prints plan and exits. Empty data → zeros, not invented results.
Underpowered results are labeled as such.
