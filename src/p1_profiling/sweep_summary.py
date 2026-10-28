#!/usr/bin/env python3
"""Chunk time and stage shares for every NPU config, from the stage runs (standard library only, host).

Chunk = 3 x (vision + connector) + prefix + 10 x denoise, from results/<cfg>/torchsim_e2e/<stage>/timing/
(the one-session e2e run matches this within 0.2% on every config where both exist). Also reports per-stage DRAM
traffic, and the roofline-ideal shares: each stage at max(FLOPs / peak, compulsory bytes / bandwidth), with the
FLOPs and compulsory bytes from roofline_stages.csv of the reference config.

  python3 src/p1_profiling/sweep_summary.py  -> results/sweep_summary.csv (one row per config)
                                               results/sweep_stages.csv  (one row per config and stage, for the plots)
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from parse_e2e import stats
from parse_togsim import hardware

RESULTS = Path(__file__).resolve().parents[2] / "results"
STAGES = {"vision": 3, "connector": 3, "prefix": 1, "denoise": 10}
CONFIGS = {p.parent.name: p for p in sorted(RESULTS.glob("*/torchsim_e2e"))}
REF = RESULTS / "edge_64x64_lpddr5" / "torchsim_e2e" / "roofline_stages.csv"   # FLOPs/bytes are config-independent


def main():
    ideal = {r["phase"]: (float(r["gflop"]) * 1e9, float(r["compulsory_mb"]) * 1e6)
             for r in csv.DictReader(REF.open())} if REF.exists() else {}
    rows, stage_rows = [], []
    for name, root in CONFIGS.items():
        logs = {s: root / s / "timing" / "stdout.log" for s in STAGES}
        if not all(l.exists() and "Total execution cycles" in l.read_text(errors="replace") for l in logs.values()):
            continue
        st = {s: stats(l) for s, l in logs.items()}
        hw = hardware(logs["vision"])
        ms = {s: n * st[s]["ms"] for s, n in STAGES.items()}
        chunk = sum(ms.values())
        row = {"config": name, "array": f"{hw['n_sa']}x{hw['sa_dim']}x{hw['sa_dim']}",
               "peak_tflops": round(hw["peak_flops_per_cycle"] * hw["freq_mhz"] / 1e6, 2),
               "dram_gbs": round(hw["dram_gbs"], 1), "chunk_ms": round(chunk, 1)}
        for s in STAGES:
            row[f"{s}_pct"] = round(100 * ms[s] / chunk, 1)
        row["dram_gb_per_chunk"] = round(sum(n * st[s]["dram_mb"] for s, n in STAGES.items()) / 1e3, 2)
        peak_flops, bw = hw["peak_flops_per_cycle"] * hw["freq_mhz"] * 1e6, hw["dram_gbs"] * 1e9
        for s, n in STAGES.items():
            stage_rows.append({"config": name, "array": row["array"], "peak_tflops": row["peak_tflops"],
                               "dram_gbs": row["dram_gbs"], "stage": s, "multiplier": n, "ms": st[s]["ms"],
                               "dram_mb": st[s]["dram_mb"], "sa_util_pct": st[s]["sa_util_pct"],
                               "gflop": ideal[s][0] / 1e9 if ideal else "",
                               "compulsory_mb": ideal[s][1] / 1e6 if ideal else "",
                               "ideal_ms": round(max(ideal[s][0] / peak_flops, ideal[s][1] / bw) * 1e3, 3) if ideal else ""})
        if ideal:
            t = {s: n * max(ideal[s][0] / (hw["peak_flops_per_cycle"] * hw["freq_mhz"] * 1e6),
                            ideal[s][1] / (hw["dram_gbs"] * 1e9)) * 1e3 for s, n in STAGES.items()}
            tot = sum(t.values())
            row["ideal_chunk_ms"] = round(tot, 1)
            for s in STAGES:
                row[f"ideal_{s}_pct"] = round(100 * t[s] / tot, 1)
        rows.append(row)
    rows.sort(key=lambda r: (r["array"], r["dram_gbs"]))
    out = RESULTS / "sweep_summary.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    stage_rows.sort(key=lambda r: (r["array"], r["dram_gbs"]))
    with (RESULTS / "sweep_stages.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stage_rows[0]))
        w.writeheader()
        w.writerows(stage_rows)
    print(f"{'config':26s} {'array':>12s} {'TFLOP/s':>7s} {'GB/s':>6s} {'chunk ms':>9s}  vis/con/pre/den %      "
          f"ideal ms  ideal vis/con/pre/den %")
    for r in rows:
        sim = "/".join(f"{r[f'{s}_pct']:.0f}" for s in STAGES)
        idl = "/".join(f"{r.get(f'ideal_{s}_pct', 0):.0f}" for s in STAGES)
        print(f"{r['config']:26s} {r['array']:>12s} {r['peak_tflops']:7.2f} {r['dram_gbs']:6.1f} {r['chunk_ms']:9.1f}  "
              f"{sim:20s} {r.get('ideal_chunk_ms', 0):8.1f}  {idl}")
    print(f"wrote {out} and {RESULTS / 'sweep_stages.csv'}")


if __name__ == "__main__":
    main()
