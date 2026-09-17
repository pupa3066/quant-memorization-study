# Cross-Hardware Efficiency Data — Contribution Schema

> For the cross-hardware efficiency study (NVIDIA RTX + peer hardware + Apple MLX/MPS).
> To be COMPARABLE, every contributed file MUST use this schema and the SAME metrics as
> `src/efficiency.py`. Data measured with a different methodology is NOT comparable — re-run the
> harness (or an equivalent that reports these exact fields) rather than force-fitting old numbers.

## Required fields (per precision, per model, per hardware)
```json
{
  "hardware": "NVIDIA RTX 4090 24GB",          // exact chip + VRAM
  "backend": "PyTorch-CUDA 2.x + <int4 kernel>",// runtime + whether a FUSED int4 kernel exists
  "fused_int4_kernel": true,                    // CRITICAL: does this backend have fused int4 matmul?
  "measured_by": "Purnima Pathak",              // or peer name — for attribution
  "harness_version": "efficiency.py@<commit>",  // provenance: same tool = comparable
  "model": "Qwen2.5-0.5B-Instruct",
  "prompt": "<same fixed prompt as efficiency.py>",
  "per_precision": [
    {"precision":"fp16","weight_mb":0,"load_s":0,"prefill_ms":0,"decode_tok_s":0,"peak_mem_mb":0},
    {"precision":"int8","weight_mb":0,"...":0},
    {"precision":"int4","weight_mb":0,"...":0}
  ]
}
```

## Rules (integrity + provenance)
1. **Same harness or equivalent metrics.** If RTX data was measured ad-hoc, re-run `efficiency.py`
   on the RTX so fields match. Otherwise it's apples-vs-oranges and cannot be combined.
2. **Attribution.** `measured_by` names who produced each hardware's data. Peer-contributed data →
   the peer is a CONTRIBUTOR/co-author (added to CITATION.cff + any writeup). Non-negotiable.
3. **Provenance.** Record exact hardware, backend, and whether a fused int4 kernel exists — this is
   the variable that explains the latency direction (the whole cross-hardware finding).
4. **No fabrication.** Missing a field → leave null, don't guess. Combiner reports what exists only.

## File naming
`results/hw_<label>.json`  e.g. `results/hw_rtx4090.json`, `results/hw_<peer-hardware>.json`,
`results/hw_m1_mlx.json` (our existing MLX data, reformatted).

## The finding this enables (hypothesis, to be confirmed by the real data)
int4 memory win is universal; int4 LATENCY win appears only where a fused int4 kernel exists
(expected: MLX ✓, CUDA/RTX ✓, PyTorch-MPS ✗). Cross-hardware evidence turns the single-machine
observation into a real systems result. State it ONLY after the numbers are in.
