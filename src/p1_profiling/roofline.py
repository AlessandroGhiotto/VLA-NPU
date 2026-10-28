#!/usr/bin/env python3
"""Roofline analysis of the simulated SmolVLA phases.

Analytic side (needs torch + the checkpoint, run in the container): GEMM/conv FLOPs
from torch's FlopCounterMode on one phase instance, and compulsory DRAM traffic
(weights + inputs + outputs, each touched once). Measured side: cycles and DRAM
requests from <root>/<phase>/timing/stdout.log. Writes <root>/roofline.csv (or roofline_stages.csv).

Peaks come from the hardware TOGSim reports in each log (parse_togsim.hardware): arrays x lanes^2 MACs
per cycle, and the Ramulator-derived DRAM bandwidth per core cycle.

  python src/p1_profiling/roofline.py [--root results/<hw>/torchsim]  |  --stages [--root results/<hw>/torchsim_e2e]
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src" / "common")]
from parse_togsim import parse_log, hardware  # noqa: E402

RW = re.compile(r"channels 0\.\.\d+ combined .*\| (\d+) reads, (\d+) writes")


def numel(x):
    if hasattr(x, "numel"):
        return x.numel()
    if isinstance(x, (tuple, list)):
        return sum(numel(i) for i in x)
    return 0


def analytic(phase, bpe, build=None):
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    import smolvla_phases as sp
    policy = sp.load_policy()
    mod, args, mult = (build or sp.build)(phase, policy)
    del policy
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        out = mod(*args)
    flops = fc.get_total_flops()
    params = sum(p.numel() for p in mod.parameters())
    acts_in, acts_out = sum(numel(a) for a in args), numel(out)
    return {
        "phase": phase, "multiplier": mult, "flops": flops,
        "param_elems": params, "in_elems": acts_in, "out_elems": acts_out,
        "compulsory_bytes": (params + acts_in + acts_out) * bpe,
    }


def measured(phase, root):
    log = root / phase / "timing" / "stdout.log"
    if not log.exists():
        return None
    text = log.read_text(errors="replace")
    m = RW.findall(text)
    reads, writes = (int(m[-1][0]), int(m[-1][1])) if m else (0, 0)
    p, hw = parse_log(log), hardware(log)
    return {"hw": hw, "cycles": p["cycles"], "dram_bytes": (reads + writes) * hw["req_bytes"],
            "sa_util_pct": round(100 * p["sa_active"] / p["sa_total"], 1) if p["sa_total"] else 0,
            "sa_active_cycles": p["sa_active"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "results" / "timing_heuristic" / "torchsim"))
    ap.add_argument("--bytes-per-elem", type=int, default=4, help="4=fp32, 2=bf16, 1=int8")
    ap.add_argument("--phases", nargs="*")
    ap.add_argument("--stages", action="store_true",
                    help="analyse whole stages (smolvla_e2e) from results/torchsim_e2e instead of single layers")
    a = ap.parse_args()
    import smolvla_phases as sp
    build, names = sp.build, sp.PHASES
    root = Path(a.root)
    if a.stages:
        import smolvla_e2e as e2e
        build, names = e2e.build, e2e.STAGES
        if a.root == ap.get_default("root"):
            root = REPO / "results" / "timing_heuristic" / "torchsim_e2e"
    rows = []
    for phase in a.phases or names:
        m = measured(phase, root)
        if m is None:
            print(f"skip {phase}: no timing log", file=sys.stderr)
            continue
        an = analytic(phase, a.bytes_per_elem, build)
        hw = m["hw"]
        t_comp = an["flops"] / hw["peak_flops_per_cycle"]
        t_mem = an["compulsory_bytes"] / hw["peak_bytes_per_cycle"]
        bound = max(t_comp, t_mem)
        rows.append({
            "phase": phase, "multiplier": an["multiplier"],
            "gflop": round(an["flops"] / 1e9, 3),
            "compulsory_mb": round(an["compulsory_bytes"] / 1e6, 2),
            "ai_flop_per_byte": round(an["flops"] / an["compulsory_bytes"], 1),
            "measured_mb": round(m["dram_bytes"] / 1e6, 2),
            "traffic_ratio": round(m["dram_bytes"] / an["compulsory_bytes"], 2),
            "measured_ai": round(an["flops"] / m["dram_bytes"], 1) if m["dram_bytes"] else 0,
            "t_compute_cyc": round(t_comp), "t_mem_cyc": round(t_mem),
            "regime": "compute" if t_comp >= t_mem else "memory",
            "roofline_cyc": round(bound), "measured_cyc": m["cycles"],
            "efficiency_pct": round(100 * bound / m["cycles"], 1),
            "sa_util_pct": m["sa_util_pct"],
            # useful MACs per busy array-cycle: 100% means every PE did a MAC whenever the array was busy
            "pe_occupancy_pct": round(100 * an["flops"] / (m["sa_active_cycles"] * hw["sa_dim"] ** 2 * 2), 1)
            if m["sa_active_cycles"] else 0,
        })
    out = root / ("roofline_stages.csv" if a.stages else "roofline.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    hw = measured(rows[0]["phase"], root)["hw"]
    print(f"{hw['n_sa']}x {hw['sa_dim']}x{hw['sa_dim']} arrays @ {hw['freq_mhz']} MHz: "
          f"peak {hw['peak_flops_per_cycle'] * hw['freq_mhz'] / 1e6:.2f} TFLOP/s, {hw['dram_gbs']:.1f} GB/s, "
          f"ridge {hw['peak_flops_per_cycle'] / hw['peak_bytes_per_cycle']:.0f} FLOP/B ({a.bytes_per_elem} B/elem)")
    for r in rows:
        print(f"{r['phase']:19s} AI {r['ai_flop_per_byte']:7.1f} ({r['regime']:7s}) "
              f"roofline {r['roofline_cyc']:>9,} vs {r['measured_cyc']:>9,} cyc = {r['efficiency_pct']:5.1f}%  "
              f"traffic x{r['traffic_ratio']:.2f}  SA busy {r['sa_util_pct']:4.1f}%  PE occ {r['pe_occupancy_pct']:5.1f}%")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
