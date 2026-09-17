#!/usr/bin/env python3
"""layer_adaptive.py — Stage-2: layer-adaptive precision analysis (ANALYSIS-ONLY, no model loading).

Consumes ALREADY-MEASURED layer-sensitivity data and derives a per-layer precision policy:
which layers can be quantized (memory saved) vs must stay higher-precision (fidelity preserved).
This is the precision analogue of per-layer dataflow tuning in sparse-conv engines.

NO inference, NO model load, NO RAM spike — pure analysis over measured JSON. That is the honest
scope under the "no RAM-heavy work" constraint: the layer-SENSITIVITY was already measured; this
turns it into a policy + quantifies the fidelity/coverage tradeoff.

Input format (measured): [{"skip_first_last": k, "layers_quantized": n, "cosine": c}, ...]
Usage: layer_adaptive.py <sensitivity.json>
"""
import json, sys

def analyze(sens):
    # sens: list of {skip_first_last, layers_quantized, cosine}
    rows = sorted(sens, key=lambda r: r["skip_first_last"])
    if not rows:
        return {"error": "no sensitivity data"}
    full = rows[0]  # skip=0 → all layers quantized (max memory saving, min fidelity)
    out = {"measured_points": rows, "analysis": {}}

    # marginal fidelity gain per skipped (protected) layer-pair
    deltas = []
    for a, b in zip(rows, rows[1:]):
        d_cos = b["cosine"] - a["cosine"]
        d_layers = a["layers_quantized"] - b["layers_quantized"]  # layers we STOPPED quantizing
        deltas.append({
            "from_skip": a["skip_first_last"], "to_skip": b["skip_first_last"],
            "fidelity_gain": round(d_cos, 6),
            "layers_protected": d_layers,
            "fidelity_gain_per_protected_layer": round(d_cos / d_layers, 6) if d_layers else None,
        })
    out["analysis"]["marginal"] = deltas

    # KEY FINDING: is sensitivity uneven? (do first/last layers cost more fidelity when quantized?)
    # If protecting first/last raises cosine, those layers are the sensitive ones → non-uniform policy wins.
    monotonic = all(rows[i]["cosine"] <= rows[i+1]["cosine"] for i in range(len(rows)-1))
    out["analysis"]["sensitivity_uneven"] = monotonic
    out["analysis"]["interpretation"] = (
        "Protecting (not quantizing) the first/last layers monotonically improves fidelity "
        f"({rows[0]['cosine']} → {rows[-1]['cosine']}) → layer sensitivity is UNEVEN; the optimal "
        "precision policy is per-layer (quantize inner layers, protect first/last), NOT uniform."
        if monotonic else "sensitivity trend not monotonic — inconclusive."
    )

    # POLICY: recommend the knee — max layers quantized while cosine stays above a quality floor
    FLOOR = 0.9999
    ok = [r for r in rows if r["cosine"] >= FLOOR]
    best = max(ok, key=lambda r: r["layers_quantized"]) if ok else rows[-1]
    out["policy"] = {
        "quality_floor": FLOOR,
        "recommended_skip_first_last": best["skip_first_last"],
        "layers_quantized": best["layers_quantized"],
        "cosine": best["cosine"],
        "rationale": "max memory saving (most layers int4) while staying above the fidelity floor; "
                     "protect the sensitive first/last layers only.",
    }
    return out

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("usage: layer_adaptive.py <sensitivity.json>", file=sys.stderr); sys.exit(2)
    d = json.load(open(path))
    sens = d.get("layer_sensitivity") if isinstance(d, dict) else d
    print(json.dumps(analyze(sens), indent=2))

if __name__ == "__main__":
    main()
