# Changelog

All notable changes to this study are documented here. Versions follow semantic versioning for the
research artifact; the Figshare DOI carries its own version count (Figshare v2 contains semantic v1.1.0).

## v1.1.0 — 2026-09-17
**Cross-hardware replication (NVIDIA CUDA / RTX 5060).**
Added by contributor **Akshay Upadhyay (@akshbhu)**; integrated + reviewed by Purnima Pathak.
- Added an NVIDIA CUDA backend (Transformers + bitsandbytes) via a `--backend hf|mlx|auto` flag in
  `src/efficiency.py` / `src/harness.py` (additive; the MLX/Apple-Silicon path is unchanged).
- Cross-platform peak-RSS (psutil) and local-dir weight-footprint measurement (Windows support).
- New measured data on RTX 5060: `results/analysis_cuda_*`, `results/runs_*local*`,
  `results/analysis_scale_*`, `results/separability_scale_*`, plus PowerShell sweep scripts.
- **Finding:** the factuality null and the "memorization is driven by model scale, not precision"
  result **reproduce on a second hardware family** (CUDA), strengthening external validity.
- Provenance: same harness/metrics as the MLX runs (comparable). Contributor credited in CITATION.cff.

## v1.0.0 — 2026-09-16
Initial release. 6-model study (Qwen2.5-0.5B/1.5B/3B, Llama-3.2-1B, Phi-3.5-mini, Gemma-2-2b) on
Apple Silicon (MLX). Factuality robust to INT4 (replicated null); memorization dominated by model
scale, not precision. Pre-registered design, from-scratch statistics, honest nulls + corrected
false positives. DOI: 10.6084/m9.figshare.33858859.
