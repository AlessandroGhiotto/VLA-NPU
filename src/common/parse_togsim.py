#!/usr/bin/env python3
"""Aggregate TOGSim logs under results/<config>/torchsim/<phase>/<mode>/ into phase_summary.csv.

Each phase may produce several logs (one per compiled kernel unless wrapped in one
TOGSimulator run); cycles and instruction counts are summed, utilizations are
cycle-weighted. `total_cycles_full_inference` applies the per-phase multiplier from meta.json.
"""
import argparse
import csv
import json
import re
from pathlib import Path

SA = re.compile(r"Core \[\d+\] : Systolic array \[\d+\] utilization\(%\): ([\d.]+), active_cycles: (\d+), idle_cycles: (\d+)")
VEC = re.compile(r"Core \[\d+\] : Vector unit utilization\(%\): ([\d.]+), active cycle: (\d+)")
COMP = re.compile(r"COMP\s+inst_count: (\d+) \(GEMM: (\d+), Vector: (\d+)\)")
BW = re.compile(r"channels 0\.\.\d+ combined \| ([\d.]+) GB/s aggregate, ([\d.]+)% of utilization")
TOTAL = re.compile(r"Total execution cycles: (\d+)")
DMA = re.compile(r"DMA active_cycles: (\d+), DMA idle_cycles: (\d+)")
CORE_CFG = re.compile(r"\[Config/Core\] Core 0: core_freq_mhz: (\d+), systolic_arrays_per_core: (\d+)")
DRAM_CFG = re.compile(r"\[Config/DRAM\] Total bandwidth ([\d.]+) GB/s, \d+ MHz, (\d+) channels, (\d+) bytes per request")
LANES = re.compile(r"^vpu_num_lanes: (\d+)", re.M)
KSUM = re.compile(r"Kernel \d+ execution summary - Started at: \d+ cycles, Total compute time: (\d+) cycles")


def hardware(path):
    """Simulated NPU parameters as TOGSim printed them at startup (the array is vpu_num_lanes x vpu_num_lanes)."""
    t = Path(path).read_text(errors="replace")
    freq, n_sa = map(int, CORE_CFG.search(t).groups())
    gbs, ch, req = DRAM_CFG.search(t).groups()
    sa_dim = int(LANES.search(t).group(1))
    return {"freq_mhz": freq, "n_sa": n_sa, "sa_dim": sa_dim, "dram_gbs": float(gbs), "dram_channels": int(ch),
            "req_bytes": int(req), "peak_flops_per_cycle": n_sa * sa_dim * sa_dim * 2,
            "peak_bytes_per_cycle": float(gbs) * 1e3 / freq}


def parse_log(path):
    t = full = Path(path).read_text(errors="replace")
    # TOGSim prints cumulative stats every core_stats_print_period_cycles; only the final
    # block (after the last "Instructions count" header) holds the end-of-run totals.
    cycles = int(TOTAL.findall(t)[-1]) if TOTAL.search(t) else 0
    t = t[t.rfind("===== Instructions count ====="):] if "===== Instructions count =====" in t else t
    sa = SA.findall(t)
    sa_active = sum(int(a) for _, a, _ in sa)
    sa_total = sum(int(a) + int(i) for _, a, i in sa)
    vec = VEC.findall(t)
    comp = COMP.findall(t)
    bw = BW.findall(full)
    dma = DMA.findall(t)
    return {
        "cycles": cycles,
        "sa_active": sa_active,
        "sa_total": sa_total,
        "vec_active": sum(int(a) for _, a in vec),
        "gemm_insts": sum(int(g) for _, g, _ in comp),
        "vec_insts": sum(int(v) for _, _, v in comp),
        "dma_active": sum(int(a) for a, _ in dma),
        "bw_gbs_x_cycles": float(bw[-1][0]) * cycles if bw else 0.0,
        "dram_util_x_cycles": float(bw[-1][1]) * cycles if bw else 0.0,
        # kernels timed at <= 1 cycle are not physical (autotune artifact, see configs/timing_autotune.yml)
        "zero_cycle_kernels": sum(int(d) <= 1 for d in KSUM.findall(full)),
    }


def main():
    here = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(here / "results" / "timing_heuristic" / "torchsim"))
    ap.add_argument("--mode", default="timing")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    rows = []
    for meta_path in sorted(root.glob(f"*/{a.mode}/meta.json")):
        meta = json.loads(meta_path.read_text())
        logs = sorted(meta_path.parent.glob("*.log"))
        agg = {}
        for lg in logs:
            for k, v in parse_log(lg).items():
                agg[k] = agg.get(k, 0) + v
        if not agg or not agg["cycles"]:
            continue
        cyc = agg["cycles"]
        rows.append({
            "phase": meta["phase"],
            "n_logs": len(logs),
            "cycles_per_instance": cyc,
            "multiplier": meta["multiplier"],
            "total_cycles_full_inference": cyc * meta["multiplier"],
            "systolic_util_pct": round(100 * agg["sa_active"] / agg["sa_total"], 2) if agg["sa_total"] else 0,
            "gemm_insts": agg["gemm_insts"],
            "vector_insts": agg["vec_insts"],
            "dram_bw_gbs_avg": round(agg["bw_gbs_x_cycles"] / cyc, 2),
            "dram_util_pct_avg": round(agg["dram_util_x_cycles"] / cyc, 2),
            "zero_cycle_kernels": agg["zero_cycle_kernels"],
        })
    out = Path(a.out) if a.out else root / "phase_summary.csv"
    if rows:
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    total = sum(r["total_cycles_full_inference"] for r in rows)
    for r in rows:
        if r["zero_cycle_kernels"]:
            print(f"WARNING {r['phase']}: {r['zero_cycle_kernels']} kernels timed at <= 1 cycle; cycles are underestimated")
    for r in rows:
        share = 100 * r["total_cycles_full_inference"] / total if total else 0
        print(f"{r['phase']:20s} {r['cycles_per_instance']:>12,} x{r['multiplier']:<3} = {r['total_cycles_full_inference']:>14,}  "
              f"({share:5.1f}%)  SA util {r['systolic_util_pct']:5.1f}%  DRAM {r['dram_bw_gbs_avg']:7.1f} GB/s")
    print(f"\ntotal (sum of phases, serial) = {total:,} cycles -> wrote {out}")


if __name__ == "__main__":
    main()
