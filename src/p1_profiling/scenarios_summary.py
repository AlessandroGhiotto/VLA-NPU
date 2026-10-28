#!/usr/bin/env python3
"""Action-chunk time and stage shares under four tiling/fusion scenarios on the 64x64 NPU (host, standard library).

Answers: does the action expert's denoise loop dominate a low-bandwidth edge NPU once the compiler stops wasting
traffic? For 12.8 and 51.2 GB/s:
  heuristic   PyTorchSim's heuristic tiling, simulated (sweep_bw_64x64_ch1, edge_64x64_lpddr5)
  forced      traffic-minimising GEMM tiles (make_mapping.py), simulated (extmap_64x64_ch1, extmap_64x64_ch4)
  fused (est) forced tiling minus the kernel-boundary round trips (data written by one kernel and read back by the
              next), removed at the stage's achieved DRAM bandwidth: an estimate of on-chip chaining of kernels,
              which TOGSim cannot simulate (nothing stays in the scratchpad between kernels). Needs the replays
              (inst_kernels.csv) of the forced-tiling runs.
  ideal       roofline: max(FLOPs / peak, compulsory bytes / bandwidth) per stage

  python3 src/p1_profiling/scenarios_summary.py -> results/scenarios.csv (one row per bandwidth, scenario and stage)
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from parse_e2e import stats
from parse_togsim import hardware

RESULTS = Path(__file__).resolve().parents[2] / "results"
STAGES = {"vision": 3, "connector": 3, "prefix": 1, "denoise": 10}
BW = {12.8: ("sweep_bw_64x64_ch1", "extmap_64x64_ch1"), 51.2: ("edge_64x64_lpddr5", "extmap_64x64_ch4")}


def run(cfg, stage):
    return RESULTS / cfg / "torchsim_e2e" / stage / "timing"


def roundtrip_mb(cfg, stage):
    """MB of kernel-boundary round trips (intermediate writes + their reads) from the replay, or None."""
    f = run(cfg, stage) / "inst_kernels.csv"
    if not f.exists():
        return None
    req = hardware(run(cfg, stage) / "stdout.log")["req_bytes"]
    rows = list(csv.DictReader(f.open()))
    return sum(int(r["rd_intermediate"]) + int(r["wr_intermediate"]) for r in rows) * req / 1e6


def main():
    ideal = {(r["config"], r["stage"]): float(r["ideal_ms"]) for r in csv.DictReader((RESULTS / "sweep_stages.csv").open())}
    out = []
    for bw, (heur, forced) in BW.items():
        sc = {"heuristic": {}, "forced": {}, "fused (est)": {}, "ideal": {}}
        for s in STAGES:
            h = stats(run(heur, s) / "stdout.log")
            sc["heuristic"][s] = h["ms"]
            sc["ideal"][s] = ideal[(heur, s)]
            log = run(forced, s) / "stdout.log"
            if not (log.exists() and "Total execution cycles" in log.read_text(errors="replace")):
                continue
            f = stats(log)
            sc["forced"][s] = f["ms"]
            rt = roundtrip_mb(forced, s)
            if rt is not None:
                # remove the round-trip bytes at the stage's achieved bandwidth, never below the ideal
                sc["fused (est)"][s] = max(f["ms"] - rt / f["dram_gbs"], sc["ideal"][s])
        for name, t in sc.items():
            if len(t) < len(STAGES):
                print(f"{bw:5.1f} GB/s {name:12s}: missing stages {sorted(set(STAGES) - set(t))}")
                continue
            chunk = sum(n * t[s] for s, n in STAGES.items())
            for s, n in STAGES.items():
                out.append({"dram_gbs": bw, "scenario": name, "stage": s, "ms": round(t[s], 3), "multiplier": n,
                            "chunk_ms": round(chunk, 1), "share_pct": round(100 * n * t[s] / chunk, 1)})
            print(f"{bw:5.1f} GB/s {name:12s} chunk {chunk:8.1f} ms  " +
                  "  ".join(f"{s} {100 * n * t[s] / chunk:4.1f}%" for s, n in STAGES.items()))
    with (RESULTS / "scenarios.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {RESULTS / 'scenarios.csv'}")


if __name__ == "__main__":
    main()
