#!/usr/bin/env python3
"""cross_hardware.py — combine per-hardware efficiency files (results/hw_*.json) into one
cross-hardware comparison table. Reports ONLY measured fields; missing data stays blank.
No fabrication. Flags provenance (harness_version) and attribution (measured_by).

Usage: cross_hardware.py   (reads results/hw_*.json, writes docs/CROSS_HARDWARE_RESULTS.md)
"""
import json, glob, os
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
DOCS = os.path.join(os.path.dirname(HERE), "docs")

def load(p):
    try: return json.load(open(p))
    except Exception: return None

def main():
    files = sorted(glob.glob(os.path.join(RESULTS, "hw_*.json")))
    hws = [load(f) for f in files]
    hws = [h for h in hws if h]
    lines = ["# Cross-Hardware Efficiency Results",
             "",
             "> Auto-generated from results/hw_*.json. Each row = one hardware/backend, MEASURED with",
             "> a comparable harness. Blank = not measured. Contributors credited per `measured_by`.",
             ""]
    if not hws:
        lines += ["_No hardware data files present yet. Add results/hw_<label>.json per",
                  "docs/CROSS_HARDWARE_SCHEMA.md (RTX + peer + M1)._"]
        open(os.path.join(DOCS, "CROSS_HARDWARE_RESULTS.md"), "w").write("\n".join(lines))
        print("no hw_*.json yet — wrote placeholder"); return

    # provenance + attribution table
    lines += ["## Provenance & attribution",
              "| hardware | backend | fused int4? | measured_by | harness |",
              "|---|---|---|---|---|"]
    for h in hws:
        lines.append(f"| {h.get('hardware','?')} | {h.get('backend','?')} | "
                     f"{h.get('fused_int4_kernel','?')} | {h.get('measured_by','?')} | "
                     f"{h.get('harness_version','?')} |")

    # int4-vs-fp16 latency + memory per hardware (the cross-hardware finding)
    lines += ["", "## INT4 vs FP16 per hardware",
              "| hardware | fused int4? | int4 mem reduction | int4 decode speedup vs fp16 |",
              "|---|---|---|---|"]
    for h in hws:
        pp = {r["precision"]: r for r in h.get("per_precision", [])}
        fp16, i4 = pp.get("fp16"), pp.get("int4")
        memred = spd = "—"
        if fp16 and i4 and fp16.get("weight_mb"):
            memred = f"{round(1 - i4['weight_mb']/fp16['weight_mb'], 3)}"
        if fp16 and i4 and fp16.get("decode_tok_s") and i4.get("decode_tok_s"):
            spd = f"{round(i4['decode_tok_s']/fp16['decode_tok_s'], 2)}×"
        lines.append(f"| {h.get('hardware','?')} | {h.get('fused_int4_kernel','?')} | {memred} | {spd} |")

    lines += ["", "## Read honestly",
              "- Memory reduction should be ~constant across hardware (precision, not device, sets it).",
              "- Decode speedup should track `fused int4?`: >1× where fused kernel exists, <1× where not.",
              "- State the cross-hardware claim ONLY if the measured pattern actually holds. Report",
              "  contradictions if they appear — that's a finding too."]
    open(os.path.join(DOCS, "CROSS_HARDWARE_RESULTS.md"), "w").write("\n".join(lines))
    print(f"wrote docs/CROSS_HARDWARE_RESULTS.md from {len(hws)} hardware file(s)")

if __name__ == "__main__":
    main()
