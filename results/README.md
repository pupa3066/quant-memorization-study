# results/ — file guide

Every reported number in `RESULTS_multimodel.md` and `docs/PREPRINT.md` traces to a file here.
Two probes are measured per model; the file prefix tells you which. Real sample sizes are stated
below so the correct `N` is never inferred from a stray field (see caveat).

## Factuality probe (closed-book PopQA)
- **`analysis_popqa_<model>.json`** — per-precision QA accuracy + INT4-vs-FP16 McNemar test and
  bootstrap accuracy-diff CI. **Real N = `n_qa` = 100 questions** (popularity-stratified, alias-aware).
  These back the **factuality table** (the replicated null: McNemar p = 1.0 / 1.0 / 0.38).
- **`runs_popqa_<model>.jsonl`** — raw per-question run logs feeding the above.
- **`analysis_popqa.json`** — aggregate.

## Memorization probe (high-surprisal token reconstruction)
- **`analysis_mem_<model>.json`** — per-precision memorized−control reconstruction **GAP**.
  **Real N = `n_mem` = 40 passages/class** (20 memorized + 20 length-matched controls in the
  multi-model corpus). These back the **memorization GAP table**.
- **`runs_mem_<model>.jsonl`** — raw reconstruction logs.
- **`separability_<model>.json`** — detectability **AUC** (membership framing).
- **`analysis_mem.json`** — aggregate.

## Controlled same-family scale ladder (INT4, contributed replication)
- **`analysis_scale_qwen{05,15,3b}.json`**, **`separability_scale_*`**, **`runs_scale_*_int4.jsonl`**
  — Qwen2.5 0.5B/1.5B/3B held at INT4, varying only size, isolating scale as the driver of
  memorization (contributed CUDA replication; see `docs/CUDA_REPLICATION.md`).

## Cross-hardware (NVIDIA CUDA / RTX 5060)
- **`analysis_cuda_popqa_qwen05.json`**, **`analysis_cuda_mem_qwen05.json`**,
  **`runs_local_qwen05.jsonl`**, **`runs_popqa_local_qwen05.jsonl`** — the CUDA-backend replication
  of the qwen05 factuality null and memorization probe (contributor: Akshay Upadhyay, ORCID
  0009-0000-1531-3198).

## Efficiency / systems
- **`efficiency_qwen05.json`**, **`efficiency_animevlog.json`**, **`hw_m1_mlx.json`** — weight
  footprint, latency, throughput (see `docs/EFFICIENCY.md`), plus the AnimeVlog cross-modality data.
- **`tradeoff_qwen05.json`** — joins behavior × efficiency.

## Caveat: which field is the real N
Some JSONs carry a vestigial `n_mem`/`n_qa` field left from an early combined run (e.g. an `n_mem: 8`
appears inside memorization files whose real per-class N is 40, and factuality files whose real N is
`n_qa: 100`). **The authoritative N per probe is the one stated above** (factuality n_qa=100;
memorization n_mem=40/class), matching `RESULTS_multimodel.md`. The historical `n=8` value refers to
the early pilot whose false positive was caught and corrected by scaling (see `docs/PREPRINT.md` §4);
no reported result uses it.
