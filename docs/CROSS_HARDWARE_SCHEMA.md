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
5. **Latency is a per-hardware CONTRAST, never a cross-hardware replication.** Decode-speed / latency
   numbers must stay one-row-per-hardware and must NOT be averaged or pooled across devices, because
   latency is hardware-specific by design (it tracks `fused_int4_kernel`, not precision alone). Only
   memory reduction and behavioral metrics (factuality accuracy, memorization GAP) may be stated as
   cross-hardware REPLICATIONS. Reporting "int4 is Nx faster" as a device-independent claim is a 6b
   integrity failure; the finding is that the latency DIRECTION depends on the backend's kernel.

## File naming
`results/hw_<label>.json`  e.g. `results/hw_rtx4090.json`, `results/hw_<peer-hardware>.json`,
`results/hw_m1_mlx.json` (our existing MLX data, reformatted).

## Cross-hardware workflow rules (what RTX/CUDA is and is not for)
> These bind before any cross-hardware run or merge. They encode which questions the second hardware
> family actually answers, and which are Mac-specific by nature.

- **CH1: Back up this machine's study data BEFORE any cross-hardware merge.** The M1/MPS results are
  the PRIMARY artifact and the baseline everything else is compared against. Snapshot results/ (and
  the analysis JSONs) before importing any hw_*.json. Do not risk the baseline.
- **CH2: RTX/CUDA is FOR the questions it can decisively answer:** (a) the 7B memorization test
  (strong FP16 baseline that ≤3B on 8GB cannot provide; this is the decisive experiment that earns
  the preprint); (b) factuality-null + memorization-GAP REPLICATION on a second hardware family
  (external validity). This is the real, wanted cross-hardware data.
- **CH3: RTX/CUDA is NOT for fixing/replicating Mac-specific phenomena:** the MPS int4 latency numbers
  (fused-kernel-dependent; CUDA has fused kernels so numbers differ by design, a CONTRAST, see Rule 5,
  never a replication) and the 8GB unified-memory safety / AnimeVlog swap-thrash panic (exists only
  because Apple Silicon shares one memory pool with no discrete VRAM). Do not expect RTX to reproduce
  either; claiming it did would be a 6b integrity failure.
- **CH4: Attribution gate (standing rule 21a / N14) applies to every contributed hw_*.json.** Before
  any RTX data lands: contributor name + ORCID in the PR, provenance confirmation posted,
  CITATION.cff + codemeta.json updated with their credit, and Pupa's explicit "merge now"; merge
  timestamp AFTER the confirmation timestamp. If any item is missing, DO NOT MERGE; state which.

## The finding this enables (hypothesis, to be confirmed by the real data)
int4 memory win is universal; int4 LATENCY win appears only where a fused int4 kernel exists
(expected: MLX ✓, CUDA/RTX ✓, PyTorch-MPS ✗). Cross-hardware evidence turns the single-machine
observation into a real systems result. State it ONLY after the numbers are in.
