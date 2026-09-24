#!/usr/bin/env python3
"""run_7b_kaggle.py — scale the quant x memorization/factuality study to 7B on a FREE 16GB GPU
(Kaggle T4/P100 or Colab T4). Drives the EXISTING harness (src/harness.py) — same probes, same
from-scratch stats — so the 7B result is directly comparable to the published 6-model study.

WHY THIS FILE: the study already has an HF+CUDA backend (make_hf_backend, bitsandbytes int8/int4).
This runner just (1) fits a 7B model on 16GB by loading ONE precision at a time and freeing between,
(2) allows the one-time model DOWNLOAD (the harness defaults to offline), (3) runs PopQA factuality +
memorization probes and writes JSONL for analysis.py.

MEMORY PLAN (16GB T4):
  - int4 (~4-5GB) and int8 (~7-8GB): fit comfortably.
  - fp16 (~14-15GB for 7B): tight; load ALONE, no other precision resident. If OOM, use a 7B whose
    fp16 fits or drop fp16 and compare int8-vs-int4 (still a scale result). Qwen2.5-7B recommended
    (same family as the published 0.5/1.5/3B — clean scale extension).

USAGE (Kaggle notebook cell, GPU enabled):
    !git clone https://github.com/pupa3066/quant-memorization-study.git
    %cd quant-memorization-study
    !pip -q install "transformers>=4.44" accelerate bitsandbytes torch
    !python src/run_7b_kaggle.py --model Qwen/Qwen2.5-7B --popqa 50 \
        --precisions int4,int8,fp16 --out results/runs_7b_qwen7b.jsonl
    !python src/analysis.py results/runs_7b_qwen7b.jsonl   # same stats as the 6-model study

Then commit results/runs_7b_*.jsonl + the analysis back to the repo (evidence that flips N7/N1).
"""
from __future__ import annotations
import os, sys, gc, json, argparse, time

# allow the one-time download on Kaggle/Colab (harness defaults to offline; we need the model)
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["CCK_ALLOW_DOWNLOAD"] = "1"   # tell harness.make_hf_backend to permit network fetch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness  # the EXISTING study harness — reuse its probes + backend + stats


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_one_precision(model_id, precision, mem_items, qa_items, out_fh):
    """Load the 7B model at ONE precision, run both probes, write JSONL, then free VRAM."""
    from dataclasses import asdict
    print(f"[7b] loading {model_id} @ {precision} ...", flush=True)
    t0 = time.time()
    be = harness.make_hf_backend(model_id, precision)   # reuse the study's CUDA backend
    print(f"[7b] loaded @ {precision} in {time.time()-t0:.0f}s; running probes", flush=True)
    n = 0
    for it in mem_items:
        r = harness.mem_probe(be, it); r.precision = precision
        out_fh.write(json.dumps(asdict(r)) + "\n"); n += 1
    for it in qa_items:
        r = harness.qa_probe(be, it); r.precision = precision
        out_fh.write(json.dumps(asdict(r)) + "\n"); n += 1
    out_fh.flush()
    del be
    free_gpu()
    print(f"[7b] {precision}: wrote {n} results, VRAM freed", flush=True)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B",
                    help="7B HF model id (Qwen2.5-7B extends the published 0.5/1.5/3B family)")
    ap.add_argument("--precisions", default="int4,int8,fp16",
                    help="comma list; order matters — put fp16 LAST so it loads alone on 16GB")
    ap.add_argument("--popqa", type=int, default=50, help="PopQA items per popularity bin")
    ap.add_argument("--popqa-local", default=None, help="local PopQA path (else fetch, network on)")
    ap.add_argument("--mem-corpus", default=None, help="memorization corpus JSON (else built-in set)")
    ap.add_argument("--out", default="results/runs_7b.jsonl")
    a = ap.parse_args()

    # data: reuse the study's loaders so the 7B run uses the SAME probe sets as the 6-model study
    mem, qa = harness.default_sets()
    if a.mem_corpus:
        mc = harness.load_mem_corpus(a.mem_corpus)
        if mc: mem = mc
    if a.popqa:
        pq = (harness.load_popqa_local(a.popqa_local, n_per_bin=a.popqa) if a.popqa_local
              else harness.load_popqa(n_per_bin=a.popqa))
        if pq: qa = pq
    print(f"[7b] probes: {len(mem)} mem items, {len(qa)} qa items", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    precisions = [p.strip() for p in a.precisions.split(",") if p.strip()]
    total = 0
    with open(a.out, "w") as fh:
        for prec in precisions:
            try:
                total += run_one_precision(a.model, prec, mem, qa, fh)
            except Exception as e:
                # honest failure logging — a precision that OOMs is recorded, not hidden
                print(f"[7b] {prec} FAILED: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    print(f"[7b] DONE: {total} results across {precisions} -> {a.out}")
    print(f"[7b] next: python src/analysis.py {a.out}   (same McNemar/bootstrap as the 6-model study)")


if __name__ == "__main__":
    main()
