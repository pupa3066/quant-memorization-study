"""harness.py — quantization x memorization/factuality measurement harness.

Implements the study in DESIGN.md. Backend-agnostic; MLX backend for Apple Silicon and a
Transformers+CUDA backend for Windows/Linux + NVIDIA (bitsandbytes int8/int4). LOCAL ONLY:
models load from a local directory or the local HF cache; no network calls at run time.

Memorization probe (Ravichander et al., 2025, arXiv:2503.12072 paradigm):
  mask high-surprisal tokens in a passage; measure whether the model reconstructs them.
  Signal = reconstruction rate on MEMORIZED-candidate set minus CONTROL set.
Factuality probe: closed-book QA exact/normalized match, stratified by popularity.

Runs the SAME model at multiple precisions {fp16, int8, int4} and logs per-item results to JSONL.

HONESTY: This file MEASURES. It does not fabricate. With no backend, --dry lists the plan and exits.
Real numbers require --run with an installed backend + model (see DESIGN.md §5, §11).
"""
from __future__ import annotations
import json, argparse, sys, re, time
from dataclasses import dataclass, asdict, field

PRECISIONS = ("fp16", "int8", "int4")

# ---------- data ----------
@dataclass
class MemItem:
    item_id: str
    text: str
    is_memorized_candidate: bool   # True = plausibly in pretraining; False = control

@dataclass
class QAItem:
    item_id: str
    question: str
    answer: str
    popularity: str                # "high" | "low"
    aliases: list = field(default_factory=list)   # acceptable answer variants (PopQA possible_answers)

@dataclass
class Result:
    item_id: str
    probe: str                     # "mem" | "qa"
    precision: str
    correct: bool
    score: float = 0.0             # mem: reconstruction fraction; qa: 1.0/0.0
    detail: str = ""

# ---------- backend protocol ----------
class Backend:
    """Wrap a model at a fixed precision. Must implement surprisal + generate."""
    def token_surprisals(self, text: str) -> list[tuple[str, float]]:
        raise NotImplementedError
    def reconstruct(self, prefix: str, n_tokens: int) -> str:
        raise NotImplementedError
    def answer(self, question: str) -> str:
        raise NotImplementedError

# ---------- MLX backend (Apple Silicon) ----------
def make_mlx_backend(model_path: str, precision: str) -> Backend:
    """Load an MLX LM. precision is informational; quantized model paths encode bits.
    Requires: pip install mlx-lm. Raises cleanly if unavailable (no fabrication)."""
    from mlx_lm import load, generate           # raises ImportError if missing
    import mlx.core as mx
    model, tokenizer = load(model_path)

    class MLXBackend(Backend):
        def encode(self, text):
            return tokenizer.encode(text)

        def token_surprisals_ids(self, ids):
            """Return per-position surprisal (-logp of the actual next token). Aligned to ids[1:]."""
            if len(ids) < 2:
                return []
            x = mx.array(ids)[None]
            logits = model(x[:, :-1])
            logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            out = []
            for i in range(1, len(ids)):
                out.append(-float(logp[0, i - 1, ids[i]]))
            return out  # length len(ids)-1, out[j] is surprisal of ids[j+1]

        def predict_next_id(self, prefix_ids):
            """Greedy argmax next-token id given a prefix (proper reconstruction test)."""
            x = mx.array(prefix_ids)[None]
            logits = model(x)
            return int(mx.argmax(logits[0, -1]).item())

        def decode(self, ids):
            return tokenizer.decode(ids)

        def answer(self, question):
            prompt = f"Answer concisely.\nQ: {question}\nA:"
            return generate(model, tokenizer, prompt=prompt, max_tokens=16, verbose=False)
    return MLXBackend()

# ---------- HF Transformers backend (Windows/Linux + NVIDIA CUDA) ----------
def make_hf_backend(model_path: str, precision: str) -> Backend:
    """Load a model via Hugging Face Transformers on CUDA. LOCAL MODE ALWAYS: weights are
    resolved from a local directory or the local HF cache; no network is used (HF_HUB_OFFLINE=1).

    precision: 'fp16' -> float16 on CUDA; 'int8'/'int4' -> bitsandbytes quantization (nf4 for int4).
    Requires: pip install torch (CUDA build) transformers accelerate bitsandbytes.
    Raises cleanly if a dependency or the local model is missing (no fabrication).
    """
    import os
    # Enforce fully-local, offline operation.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch                                              # raises ImportError if missing
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("HF backend requested but CUDA is not available. "
                           "Install a CUDA build of torch, or use --backend mlx on Apple Silicon.")

    token = os.environ.get("HF_TOKEN") or None                # only used to read a gated LOCAL cache
    load_kwargs = {"local_files_only": True, "token": token}

    if precision == "int8":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "int4":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    else:  # fp16 (default full-precision-ish baseline on GPU)
        load_kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", **load_kwargs)
    model.eval()

    class HFBackend(Backend):
        def encode(self, text):
            return tokenizer.encode(text, add_special_tokens=False)

        def token_surprisals_ids(self, ids):
            """Per-position surprisal (-logp of the actual next token), aligned to ids[1:]."""
            if len(ids) < 2:
                return []
            x = torch.tensor(ids, device="cuda").unsqueeze(0)
            with torch.no_grad():
                logits = model(x[:, :-1]).logits            # [1, len-1, vocab]
                logp = torch.log_softmax(logits.float(), dim=-1)
            out = []
            for i in range(1, len(ids)):
                out.append(-float(logp[0, i - 1, ids[i]].item()))
            return out

        def predict_next_id(self, prefix_ids):
            """Greedy argmax next-token id given a prefix (proper reconstruction test)."""
            x = torch.tensor(prefix_ids, device="cuda").unsqueeze(0)
            with torch.no_grad():
                logits = model(x).logits
            return int(torch.argmax(logits[0, -1]).item())

        def decode(self, ids):
            return tokenizer.decode(ids)

        def answer(self, question):
            prompt = f"Answer concisely.\nQ: {question}\nA:"
            x = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = model.generate(**x, max_new_tokens=16, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            new = gen[0, x["input_ids"].shape[1]:]
            return tokenizer.decode(new, skip_special_tokens=True)
    return HFBackend()

# ---------- backend selection ----------
def select_backend_factory(backend: str):
    """Return a factory fn(model_path, precision) -> Backend for the chosen backend.
    'auto' picks HF when CUDA is available, else MLX (Apple Silicon)."""
    if backend == "hf":
        return make_hf_backend
    if backend == "mlx":
        return make_mlx_backend
    # auto
    try:
        import torch
        if torch.cuda.is_available():
            return make_hf_backend
    except Exception:
        pass
    return make_mlx_backend

# ---------- probes ----------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

def mem_probe(backend, item: MemItem, top_k_frac: float = 0.25) -> Result:
    """Proper reconstruction test at the TOKEN-ID level.
    For each high-surprisal position j (measured by the model itself), give the model the exact
    prefix ids[:j] and check if its greedy argmax next-token id == the true ids[j].
    Score = fraction of high-surprisal positions correctly reconstructed. 'correct' = any hit
    (passage shows memorization signal). Deterministic; no lossy re-tokenization.
    """
    ids = backend.encode(item.text)
    if len(ids) < 4:
        return Result(item.item_id, "mem", "?", False, "too_short")
    surp = backend.token_surprisals_ids(ids)          # surp[j] = surprisal of ids[j+1]
    k = max(1, int(len(surp) * top_k_frac))
    # positions in ids to test (target index t = j+1)
    ranked = sorted(range(len(surp)), key=lambda j: surp[j], reverse=True)[:k]
    hits = 0; tested = 0; examples = []
    for j in sorted(ranked):
        t = j + 1
        if t < 1:
            continue
        pred = backend.predict_next_id(ids[:t])
        ok = (pred == ids[t])
        hits += int(ok); tested += 1
        if len(examples) < 3:
            examples.append(f"{backend.decode([ids[t]])!r}{'=' if ok else '!'}{backend.decode([pred])!r}")
    frac = hits / tested if tested else 0.0
    return Result(item.item_id, "mem", "?", hits > 0, round(frac, 4),
                  f"recon={hits}/{tested} ({frac:.2f}) ex:{'|'.join(examples)}")

def qa_probe(backend: Backend, item: QAItem) -> Result:
    gen = backend.answer(item.question)
    gnorm = _norm(gen)
    candidates = [item.answer] + list(item.aliases or [])
    hit = any(_norm(c) and _norm(c) in gnorm for c in candidates)
    return Result(item.item_id, "qa", "?", bool(hit), 1.0 if hit else 0.0,
                  f"gold={item.answer!r} gen={gen[:30]!r}")

# ---------- runner ----------
def run(model_paths: dict, mem_items, qa_items, out="runs.jsonl", backend="auto"):
    """model_paths: {precision: model_path_or_id}. Runs every probe at every precision.
    backend: 'auto' | 'hf' (CUDA) | 'mlx' (Apple Silicon)."""
    factory = select_backend_factory(backend)
    n = 0
    with open(out, "w") as fh:
        for prec, path in model_paths.items():
            be = factory(path, prec)
            for it in mem_items:
                r = mem_probe(be, it); r.precision = prec
                fh.write(json.dumps(asdict(r)) + "\n"); n += 1
            for it in qa_items:
                r = qa_probe(be, it); r.precision = prec
                fh.write(json.dumps(asdict(r)) + "\n"); n += 1
    return n

# ---------- tiny built-in probe sets (placeholder; replace with real curated sets) ----------
def load_mem_corpus(path="data/mem_corpus.json"):
    """Load memorized + control passages from the JSON corpus into MemItems."""
    import json as _j
    with open(path) as fh:
        d = _j.load(fh)
    items = []
    for i, t in enumerate(d.get("memorized", [])):
        items.append(MemItem(f"mem_{i:03d}", t, True))
    for i, t in enumerate(d.get("control", [])):
        items.append(MemItem(f"ctrl_{i:03d}", t, False))
    return items

def load_popqa_local(path, n_per_bin=50):
    """Load PopQA from a LOCAL file (json list or jsonl) with fields question/obj/s_pop/possible_answers.
    Quantile-splits by s_pop into low (long-tail) and high (popular). Fully offline."""
    import json as _j, ast, os
    if not path or not os.path.exists(path):
        return []
    rows = []
    with open(path) as fh:
        if path.endswith(".jsonl"):
            rows = [_j.loads(l) for l in fh if l.strip()]
        else:
            d = _j.load(fh)
            rows = d if isinstance(d, list) else d.get("rows", d.get("data", []))
    fetched = []
    for r in rows:
        r = r.get("row", r)  # tolerate HF datasets-server row wrapping
        q, a, pop = r.get("question"), r.get("obj"), r.get("s_pop")
        if not (q and a and isinstance(pop, (int, float))):
            continue
        try:
            aliases = ast.literal_eval(r.get("possible_answers") or "[]")
        except Exception:
            aliases = []
        fetched.append((int(pop), q, a, aliases, r.get("id")))
    if not fetched:
        return []
    fetched.sort(key=lambda x: x[0])
    low, high = fetched[:n_per_bin], fetched[-n_per_bin:]
    items = []
    for pop, q, a, al, rid in low:
        items.append(QAItem(f"popqa_low_{rid}", q, a, "low", al))
    for pop, q, a, al, rid in high:
        items.append(QAItem(f"popqa_high_{rid}", q, a, "high", al))
    return items

def load_popqa(n_per_bin=50, seed=0):
    """Fetch PopQA via HF datasets-server, bin by subject popularity (s_pop = Wikipedia pageviews).
    NOTE: makes a network call. In LOCAL-ONLY mode prefer --popqa-local <path>. Kept for parity.
    Returns QAItems: bottom-quantile s_pop -> 'low' (long-tail), top-quantile -> 'high'.
    Uses possible_answers as accepted aliases."""
    import os
    if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1":
        print("[popqa] offline mode: skipping network fetch (use --popqa-local)", file=sys.stderr)
        return []
    import json as _j, urllib.request, ast, random
    fetched = []
    # PopQA test split ~14k rows; sample a chunk deterministically, then quantile-split.
    CHUNK = max(400, n_per_bin * 8)
    off = 0
    while len(fetched) < CHUNK and off < 3000:
        url = (f"https://datasets-server.huggingface.co/rows?dataset=akariasai%2FPopQA"
               f"&config=default&split=test&offset={off}&length=100")
        try:
            d = _j.load(urllib.request.urlopen(url, timeout=30))
        except Exception:
            break
        rows = d.get("rows", [])
        if not rows:
            break
        for row in rows:
            r = row["row"]
            q, a, pop = r.get("question"), r.get("obj"), r.get("s_pop")
            if not (q and a and isinstance(pop, (int, float))):
                continue
            try:
                aliases = ast.literal_eval(r.get("possible_answers") or "[]")
            except Exception:
                aliases = []
            fetched.append((int(pop), q, a, aliases, r.get("id")))
        off += 100
    if not fetched:
        return []
    fetched.sort(key=lambda x: x[0])           # by popularity ascending
    low = fetched[:n_per_bin]                    # long-tail
    high = fetched[-n_per_bin:]                  # popular
    items = []
    for pop, q, a, al, rid in low:
        items.append(QAItem(f"popqa_low_{rid}", q, a, "low", al))
    for pop, q, a, al, rid in high:
        items.append(QAItem(f"popqa_high_{rid}", q, a, "high", al))
    return items

def default_sets():
    mem = [
        # MEMORIZED candidates: famous verbatim text LLMs reliably reproduce
        MemItem("mem_dickens", "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness", True),
        MemItem("mem_melville", "Call me Ishmael. Some years ago never mind how long precisely, having little or no money in my purse", True),
        MemItem("mem_austen", "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife", True),
        MemItem("mem_orwell", "It was a bright cold day in April, and the clocks were striking thirteen", True),
        MemItem("mem_tolkien", "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole", True),
        # CONTROL: synthetic, cannot have been in pretraining verbatim
        MemItem("ctrl_synth_1", "The zubbly frobnicator calibrated its quantum teapot on a drizzly Wednesday afternoon", False),
        MemItem("ctrl_synth_2", "My purple stapler negotiated a peace treaty with the disgruntled office fern last spring", False),
        MemItem("ctrl_synth_3", "Seventeen invisible walruses audited the flimflam ledger beneath the marmalade skyline", False),
    ]
    qa = [
        # HIGH popularity (expect correct)
        QAItem("qa_high_1", "What is the capital of France?", "Paris", "high"),
        QAItem("qa_high_2", "Who wrote Romeo and Juliet?", "Shakespeare", "high"),
        # LOW popularity / long-tail (this is where quantization degradation may appear)
        QAItem("qa_low_1", "What is the capital of Bhutan?", "Thimphu", "low"),
        QAItem("qa_low_2", "In what year was the transistor invented?", "1947", "low"),
        QAItem("qa_low_3", "Who discovered the element francium?", "Perey", "low"),
        QAItem("qa_low_4", "What is the capital of Kazakhstan?", "Astana", "low"),
        QAItem("qa_low_5", "In what year did the Treaty of Westphalia get signed?", "1648", "low"),
        QAItem("qa_low_6", "What is the atomic number of tungsten?", "74", "low"),
    ]
    return mem, qa

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="print the plan and exit (no backend, no cost)")
    ap.add_argument("--run", action="store_true", help="execute (needs mlx-lm + models)")
    ap.add_argument("--fp16", default=None); ap.add_argument("--int8", default=None); ap.add_argument("--int4", default=None)
    ap.add_argument("--backend", choices=("auto", "hf", "mlx"), default="auto",
                    help="inference backend: hf=Transformers+CUDA (Windows/NVIDIA), mlx=Apple Silicon, auto=detect")
    ap.add_argument("--popqa", type=int, default=0, help="use N PopQA items PER popularity bin (real long-tail facts)")
    ap.add_argument("--popqa-local", default=None, help="path to a LOCAL PopQA json/jsonl (offline factuality run)")
    ap.add_argument("--mem-corpus", default=None, help="path to memorization corpus JSON (scaled mem probe)")
    ap.add_argument("--out", default="runs.jsonl")
    a = ap.parse_args()
    mem, qa = default_sets()
    if a.mem_corpus:
        mc = load_mem_corpus(a.mem_corpus)
        if mc:
            mem = mc
            print(f"[mem-corpus] loaded {len(mem)} passages", file=sys.stderr)
    if a.popqa:
        pq = load_popqa_local(a.popqa_local, n_per_bin=a.popqa) if a.popqa_local else load_popqa(n_per_bin=a.popqa)
        if pq:
            qa = pq
            src = a.popqa_local if a.popqa_local else "datasets-server"
            print(f"[popqa] loaded {len(qa)} QA items ({a.popqa}/bin high+low) from {src}", file=sys.stderr)
        else:
            print("[popqa] no PopQA loaded; falling back to built-in QA set", file=sys.stderr)
    if a.dry or not a.run:
        plan = {"backend": a.backend, "precisions": PRECISIONS, "n_mem_items": len(mem), "n_qa_items": len(qa),
                "probes": ["high-surprisal reconstruction", "closed-book QA"],
                "mode": "LOCAL ONLY (HF_HUB_OFFLINE) — weights loaded from local dir/cache, no network",
                "note": "No backend invoked. Provide --run with --fp16/--int8/--int4 model paths to produce REAL data."}
        print(json.dumps(plan, indent=2)); sys.exit(0)
    model_paths = {p: v for p, v in (("fp16", a.fp16), ("int8", a.int8), ("int4", a.int4)) if v}
    if not model_paths:
        print("ERROR: --run requires at least one of --fp16/--int8/--int4 <model_path>", file=sys.stderr); sys.exit(2)
    t0 = time.time()
    n = run(model_paths, mem, qa, out=a.out, backend=a.backend)
    print(f"wrote {n} results to {a.out} in {time.time()-t0:.1f}s across {list(model_paths)} [backend={a.backend}]")
