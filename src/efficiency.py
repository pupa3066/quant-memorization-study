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
  efficiency.py --models fp16=<path> int8=<path> int4=<path> [--backend hf|mlx|auto] --out efficiency.json
  (hf = Transformers+CUDA on Windows/NVIDIA; mlx = Apple Silicon. LOCAL model paths, offline.)
"""
from __future__ import annotations
import json, sys, time, argparse, os, glob

def peak_rss_mb():
    """Cross-platform process peak RSS in MB. Uses psutil when available (works on Windows),
    falls back to resource.getrusage on POSIX, else None."""
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    except Exception:
        pass
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(v / (1024 * 1024), 1) if sys.platform == "darwin" else round(v / 1024, 1)
    except Exception:
        return None

def hf_cache_weight_bytes(model_path: str) -> int:
    """Best-effort weight footprint. If model_path is a local dir, sum its weight files;
    otherwise sum matching files in the local HF cache. Memory-footprint proxy."""
    exts = ("*.safetensors", "*.gguf", "*.npz", "*.bin")
    total = 0
    if os.path.isdir(model_path):
        for ext in exts:
            for f in glob.glob(os.path.join(model_path, "**", ext), recursive=True):
                try: total += os.path.getsize(f)
                except OSError: pass
        return total
    cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    cache = os.environ.get("HF_HOME_HUB", cache)
    name = "models--" + model_path.replace("/", "--")
    for ext in exts:
        for f in glob.glob(os.path.join(cache, name, "**", ext), recursive=True):
            try: total += os.path.getsize(f)
            except OSError: pass
    return total

PROMPT = "Explain in one sentence why quantization reduces memory usage in neural networks."

def measure_one_mlx(precision: str, model_path: str) -> dict:
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

def measure_one_hf(precision: str, model_path: str) -> dict:
    """CUDA path: Transformers + bitsandbytes, LOCAL ONLY. Reports GPU memory + latency."""
    allow_dl = os.environ.get("CCK_ALLOW_DOWNLOAD") == "1"
    if not allow_dl:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available for HF efficiency measurement")
    token = os.environ.get("HF_TOKEN") or None
    local_only = not allow_dl
    kwargs = {"local_files_only": local_only, "token": token, "device_map": "cuda"}
    if precision == "int8":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "int4":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    else:
        kwargs["torch_dtype"] = torch.float16

    torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only, token=token)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs); model.eval()
    torch.cuda.synchronize()
    load_s = round(time.perf_counter() - t0, 2)

    ids = tok(PROMPT, return_tensors="pt").to("cuda")
    torch.cuda.synchronize(); t1 = time.perf_counter()
    with torch.no_grad():
        _ = model(**ids)
    torch.cuda.synchronize()
    prefill_ms = round((time.perf_counter() - t1) * 1000, 1)

    N = 64
    torch.cuda.synchronize(); t2 = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(**ids, max_new_tokens=N, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t2
    decode_tok_s = round(N / dt, 1) if dt > 0 else None
    gpu_peak_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)

    return {
        "precision": precision,
        "model": model_path,
        "weight_bytes": hf_cache_weight_bytes(model_path),
        "weight_mb": round(hf_cache_weight_bytes(model_path) / (1024*1024), 1),
        "load_s": load_s,
        "prefill_ms": prefill_ms,
        "decode_tok_s": decode_tok_s,
        "peak_rss_mb": peak_rss_mb(),
        "gpu_peak_mb": gpu_peak_mb,
    }

def _select_measure(backend: str):
    if backend == "mlx":
        return measure_one_mlx
    if backend == "hf":
        return measure_one_hf
    try:
        import torch
        if torch.cuda.is_available():
            return measure_one_hf
    except Exception:
        pass
    return measure_one_mlx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="precision=path pairs, e.g. fp16=... int4=...")
    ap.add_argument("--backend", choices=("auto", "hf", "mlx"), default="auto",
                    help="hf=Transformers+CUDA (Windows/NVIDIA), mlx=Apple Silicon, auto=detect")
    ap.add_argument("--out", default="efficiency.json")
    a = ap.parse_args()
    measure_one = _select_measure(a.backend)

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
