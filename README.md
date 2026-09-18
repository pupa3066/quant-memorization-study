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
  harness.py            memorization + factuality probes (MLX + CUDA/Transformers backends)
  analysis.py           per-precision metrics: GAP, accuracy, McNemar, bootstrap CIs
  separability.py       memorization detectability (AUC)
  efficiency.py         systems metrics: weight footprint, latency, throughput (MLX + CUDA)
  tradeoff.py           joins behavior × efficiency
  combine_results.py    aggregates per-model analyses → RESULTS_multimodel.md
  plot_cuda_replication.py  charts: McNemar 2×2, CUDA-vs-MLX cells/CIs, scale → docs/cuda_replication.png
  overnight_sweep.sh    RAM/disk-safe multi-model sweep (Apple) · launch_overnight.sh  scheduler
  overnight_sweep.ps1   Windows/NVIDIA sweep (local, CUDA) · launch_overnight.ps1  scheduler
results/                ← all run logs (runs_*.jsonl) + computed metrics (*_*.json)
data/                   ← mem_corpus.json (memorized + control passages)
docs/                   ← detailed writeups (see index below)
```
requirements-cuda.txt   ← CUDA-backend deps (torch/transformers/bitsandbytes) for Windows+NVIDIA
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

## Reproduce (Windows + NVIDIA GPU — local models, offline)
Runs entirely on **local model files** with CUDA; no network at run time (`HF_HUB_OFFLINE=1`).
INT8/INT4 use `bitsandbytes` to quantize the same fp16 weights at load. PowerShell:
```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt

# 0) Pre-download weights ONCE into a local folder (do this while online), e.g.:
#    huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir C:\models\Qwen2.5-0.5B-Instruct
#    (set $env:HF_TOKEN first for gated repos like Llama/Gemma)

# 1) Plan only (no model, no cost):
.\.venv\Scripts\python.exe src\harness.py --dry --backend hf

# 2) Real run against LOCAL folders (fp16 baseline + int4 via bitsandbytes):
.\.venv\Scripts\python.exe src\harness.py --run --backend hf `
  --fp16 C:\models\Qwen2.5-0.5B-Instruct `
  --int4 C:\models\Qwen2.5-0.5B-Instruct `
  --mem-corpus data\mem_corpus.json --out results\runs.jsonl
.\.venv\Scripts\python.exe src\analysis.py results\runs.jsonl

# 3) Efficiency (adds GPU peak memory + latency), and full sweep:
.\.venv\Scripts\python.exe src\efficiency.py --backend hf `
  --models fp16=C:\models\Qwen2.5-0.5B-Instruct int4=C:\models\Qwen2.5-0.5B-Instruct `
  --out results\efficiency_qwen05.json
.\src\overnight_sweep.ps1 -ModelsRoot C:\models   # sequential, VRAM-safe, results-only commits
```
Notes:
- `--backend auto` picks CUDA when available, else MLX. The example forces `hf`.
- For an offline factuality run, provide a local PopQA dump and pass `--popqa 50 --popqa-local data\popqa_local.jsonl`; otherwise the built-in QA set is used.
- Size the model ladder in `src\overnight_sweep.ps1` to your VRAM.

## Current results (6 models: Qwen2.5-0.5B/1.5B/3B, Llama-3.2-1B, Phi-3.5-mini, Gemma-2-2b)
- **Factuality: INT4 ≈ FP16, a replicated null** (McNemar p=1.0, 1.0, 0.38 on paired models).
  **Reproduced on NVIDIA CUDA (RTX 5060): p=1.0, CI [−0.07, +0.05]** — see
  `docs/CUDA_REPLICATION.md` and `docs/cuda_replication.png`.
- **Memorization: dominated by model scale, not precision** — the single-0.5B "int4 erases
  memorization" story did NOT replicate. Honest, mixed result. **Controlled single-family test on
  CUDA (Qwen2.5 at fixed INT4): GAP 0.008 → 0.025 → 0.095 for 0.5B → 1.5B → 3B.**
- **Efficiency (systems):** INT4 = ~72% smaller weights + 2.4× decode on MLX; but int4 latency
  *depends on fused-kernel support* (MLX fused → faster; PyTorch-MPS → 6.3× slower per-op). See
  `docs/EFFICIENCY.md`.
- **Methodology:** an N=8 factuality pilot false positive and a single-model memorization artifact
  were both corrected by scaling. Measure, don't assert.

Full numbers + how-to-read: `RESULTS_multimodel.md`. Engineering log: `docs/DEVLOG.md`.

## Honesty policy
No fabricated numbers. No backend → prints plan and exits. Empty data → zeros, not invented results.
Underpowered results are labeled as such.
