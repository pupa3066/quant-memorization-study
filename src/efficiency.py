#!/usr/bin/env python3
"""efficiency.py — measure the SYSTEMS-side cost of quantization (memory + latency), to pair with
the behavior results (memorization/factuality). Produces a behavior-vs-efficiency tradeoff.

Metrics (MEASURED, per precision, on the SAME model):
- weight_bytes: on-disk quantized weight size (proxy for memory footprint; the memory-bound win)
- peak_rss_mb: process resident memory after load + a forward pass (real host memory footprint)
- load_s: model load time
- decode_tok_s: generation throughput (tokens/sec) over a fixed prompt
- prefill_ms: latency of a single forward pass over a fixed-length prompt

Framing: quantization is a MEMORY-BOUND optimization (fewer bits/param -> less bandwidth). This
reports the footprint reduction and any latency effect, in the same terms as systems-efficiency work
(latency, memory footprint). No fabrication: if a model can't load, it's skipped and logged.

Usage:
  efficiency.py --models fp16=<path> int8=<path> int4=<path> --out efficiency.json
"""
from __future__ import annotations
import json, sys, time, argparse, os, resource, glob

def peak_rss_mb():
    # ru_maxrss is bytes on macOS, kilobytes on Linux
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(v / (1024*1024), 1) if sys.platform == "darwin" else round(v / 1024, 1)

def hf_cache_weight_bytes(model_path: str) -> int:
    """Best-effort: sum safetensors/gguf sizes in the HF cache for this model (memory-footprint proxy)."""
    cache = os.path.expanduser("~/.cache/huggingface/hub")
    name = "models--" + model_path.replace("/", "--")
    total = 0
    for ext in ("*.safetensors", "*.gguf", "*.npz"):
        for f in glob.glob(os.path.join(cache, name, "**", ext), recursive=True):
            try: total += os.path.getsize(f)
            except OSError: pass
    return total

PROMPT = "Explain in one sentence why quantization reduces memory usage in neural networks."

def measure_one(precision: str, model_path: str) -> dict:
    from mlx_lm import load, generate
    t0 = time.perf_counter()
    model, tok = load(model_path)
    load_s = round(time.perf_counter() - t0, 2)

    # prefill latency: one forward pass over the prompt
    import mlx.core as mx
    ids = tok.encode(PROMPT)
    t1 = time.perf_counter()
    _ = model(mx.array(ids)[None]); mx.eval(_)
    prefill_ms = round((time.perf_counter() - t1) * 1000, 1)

    # decode throughput: generate N tokens, measure tok/s
    N = 64
    t2 = time.perf_counter()
    _ = generate(model, tok, prompt=PROMPT, max_tokens=N, verbose=False)
    dt = time.perf_counter() - t2
    decode_tok_s = round(N / dt, 1) if dt > 0 else None

    return {
        "precision": precision,
        "model": model_path,
        "weight_bytes": hf_cache_weight_bytes(model_path),
        "weight_mb": round(hf_cache_weight_bytes(model_path) / (1024*1024), 1),
        "load_s": load_s,
        "prefill_ms": prefill_ms,
        "decode_tok_s": decode_tok_s,
        "peak_rss_mb": peak_rss_mb(),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="precision=path pairs, e.g. fp16=mlx-community/... int4=...")
    ap.add_argument("--out", default="efficiency.json")
    a = ap.parse_args()

    pairs = []
    for m in a.models:
        if "=" not in m: print(f"skip malformed {m}", file=sys.stderr); continue
        prec, path = m.split("=", 1); pairs.append((prec, path))

    results = []
    for prec, path in pairs:
        try:
            r = measure_one(prec, path)
            results.append(r)
            print(f"[eff] {prec}: {r['weight_mb']}MB weights, {r['decode_tok_s']} tok/s, prefill {r['prefill_ms']}ms", file=sys.stderr)
        except Exception as e:
            print(f"[eff] {prec} FAILED: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)

    # derive reductions vs fp16 baseline if present
    base = next((r for r in results if r["precision"] == "fp16"), None)
    out = {"prompt": PROMPT, "per_precision": results}
    if base and base["weight_bytes"]:
        for r in results:
            if r["weight_bytes"]:
                r["weight_reduction_vs_fp16"] = round(1 - r["weight_bytes"]/base["weight_bytes"], 4)
            if base["decode_tok_s"] and r["decode_tok_s"]:
                r["decode_speedup_vs_fp16"] = round(r["decode_tok_s"]/base["decode_tok_s"], 3)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
