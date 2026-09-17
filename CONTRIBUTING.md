# Contributing cross-hardware data

Thanks for testing on your hardware! This study compares quantization efficiency across devices, so
contributions must be **comparable** and **attributed**. A PR check (`src/check_contribution.py`, run
automatically in CI) validates your file before merge.

## How to contribute
1. Measure with the same harness (guarantees comparable metrics):
   ```sh
   python3 src/efficiency.py --models fp16=<model> int8=<model> int4=<model> --out results/hw_<label>.json
   ```
   If you can't run our harness, produce a JSON with the SAME fields (see `docs/CROSS_HARDWARE_SCHEMA.md`).
2. Fill the provenance + attribution fields honestly:
   - `hardware` (exact chip + memory), `backend` (runtime + version),
   - `fused_int4_kernel`: true/false — **the key variable** explaining latency direction,
   - `measured_by`: **your name** (you'll be credited as a contributor / co-author),
   - `harness_version`: how it was measured.
3. Open a PR adding `results/hw_<label>.json`. The check must pass (green) to merge.

## What the check enforces (and why)
- **Schema**: all required fields present → comparable across devices.
- **Attribution**: `measured_by` must be a real name → research integrity; contributors are credited.
- **Provenance**: `harness_version` present → traceable, comparable.
- **Comparability sanity**: e.g. int4 weight must be smaller than fp16 (else it's a measurement error).
- **No fabrication**: missing values stay blank; the checker never invents data.

## Attribution policy
Contributed data means you are credited (CITATION.cff + any writeup). Data measured by someone else
is never presented as the maintainer's own. If in doubt, ask in the PR.
