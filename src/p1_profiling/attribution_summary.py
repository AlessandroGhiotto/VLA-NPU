#!/usr/bin/env python3
"""Where the time and the DRAM traffic of each stage go, from the instruction-level replays (host, standard library).

Reads results/<cfg>/torchsim_e2e/<stage>/timing/inst_kernels.csv (written by replay_trace.py) for every config
where it exists, and the roofline ideal of results/sweep_stages.csv (FLOPs / peak, compulsory bytes / bandwidth).
Times are ms per stage instance; traffic is MB per stage instance. array_ms is the time the array is occupied
(any matmul/preload instruction in flight, fill and drain included); sa_streaming_ms is TOGSim's own "Systolic array
active_cycles" averaged over the arrays, which leaves out the fill/drain cycles not overlapped with the previous
instruction (Core::sa_cycle bubbles), i.e. the time the array streams operands.

  python3 src/p1_profiling/attribution_summary.py
  -> results/attribution_stages.csv   one row per config and stage: time categories, unit busy times, ideal, traffic
     results/attribution_classes.csv  the same per kernel class (GEMM, attention, softmax, other vector)
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from parse_togsim import hardware, parse_log

RESULTS = Path(__file__).resolve().parents[2] / "results"
STAGES = {"vision": 3, "connector": 3, "prefix": 1, "denoise": 10}
TIME = ["cycles", "array", "vector", "exposed_mem", "other", "sa_busy", "vu_busy", "dma_busy", "dma_hidden"]
TRAFFIC = ["rd_compulsory", "rd_intermediate", "rd_cross_kernel", "rd_partial_sum", "rd_tiling_external",
           "rd_tiling_produced", "rd_misaligned", "wr_intermediate", "wr_output", "wr_rewrites", "wr_misaligned"]


def main():
    ideal = {(r["config"], r["stage"]): r for r in csv.DictReader((RESULTS / "sweep_stages.csv").open())}
    stage_rows, class_rows = [], []
    for kfile in sorted(RESULTS.glob("*/torchsim_e2e/*/timing/inst_kernels.csv")):
        run = kfile.parent
        cfg, stage = run.parents[2].name, run.parent.name
        if stage not in STAGES or (cfg, stage) not in ideal:
            continue
        hw = hardware(run / "stdout.log")
        ms = lambda c: c / (hw["freq_mhz"] * 1e3)
        mb = lambda n: n * hw["req_bytes"] / 1e6
        tot = defaultdict(lambda: defaultdict(float))
        for r in csv.DictReader(kfile.open()):
            for key in ("all", r["class"]):
                for n in TIME + TRAFFIC:
                    tot[key][n] += float(r[n])
        p = parse_log(run / "replay.log")
        i = ideal[(cfg, stage)]
        peak, bw = float(i["peak_tflops"]), float(i["dram_gbs"])
        for key, t in tot.items():
            row = {"config": cfg, "stage": stage, "multiplier": STAGES[stage]}
            if key != "all":
                row["class"] = key
            row.update({("ms" if n == "cycles" else f"{n}_ms"): round(ms(t[n]), 4) for n in TIME})
            if key == "all":
                row.update({"ideal_compute_ms": round(float(i["gflop"]) / peak, 4),
                            "ideal_mem_ms": round(float(i["compulsory_mb"]) / bw, 4),
                            "ideal_ms": float(i["ideal_ms"]), "compulsory_mb_roofline": float(i["compulsory_mb"]),
                            "sa_streaming_ms": round(ms(p["sa_active"] / hw["n_sa"]), 4)})
            row.update({f"{n}_mb": round(mb(t[n]), 3) for n in TRAFFIC})
            row["dram_mb"] = round(sum(row[f"{n}_mb"] for n in TRAFFIC), 2)
            (stage_rows if key == "all" else class_rows).append(row)
    for name, rows in (("attribution_stages.csv", stage_rows), ("attribution_classes.csv", class_rows)):
        with (RESULTS / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"{'config':24s} {'stage':9s} {'ms':>8s} {'array':>6s} {'vector':>6s} {'mem':>6s} {'other':>6s} "
          f"{'ideal':>7s} {'x ideal':>7s} {'MB':>8s} {'x comp':>6s} {'stream/ideal compute':>20s}")
    for r in stage_rows:
        p = lambda n: 100 * r[f"{n}_ms"] / r["ms"]
        print(f"{r['config']:24s} {r['stage']:9s} {r['ms']:8.2f} {p('array'):5.1f}% {p('vector'):5.1f}% "
              f"{p('exposed_mem'):5.1f}% {p('other'):5.1f}% {r['ideal_ms']:7.2f} {r['ms'] / r['ideal_ms']:7.1f} "
              f"{r['dram_mb']:8.1f} {r['dram_mb'] / r['compulsory_mb_roofline']:6.1f} "
              f"{r['sa_streaming_ms'] / r['ideal_compute_ms']:20.2f}")
    print(f"wrote {RESULTS / 'attribution_stages.csv'} and attribution_classes.csv")


if __name__ == "__main__":
    main()
