#!/usr/bin/env python3
"""PyTorchSim vs SCALE-Sim cycles for the cross-check GEMMs (standard library only, host).

Both sides model one 64x64 weight-stationary array and are compute-bound (PyTorchSim: configs/pytorchsim/
check_64x64_hbm2.yml via torchsim_gemm.py; SCALE-Sim v3: configs/scalesim/ws64_check.cfg, stall-free). SCALE-Sim's
"Total Cycles" excludes the initial SRAM prefetch. Ideal = M*N*K / (64*64) cycles.

  python3 src/p1_profiling/compare_scalesim.py  -> results/scalesim_check/comparison.csv
"""
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKLOAD = REPO / "configs" / "scalesim" / "gemms_check.csv"
SCALESIM = REPO / "results" / "scalesim_check" / "COMPUTE_REPORT.csv"
TORCHSIM = REPO / "results" / "check_64x64_hbm2" / "gemm" / "gemm_cycles.csv"
PES = 64 * 64


def rows(path):
    """csv rows with stripped keys and values (SCALE-Sim writes ', ' separators and a trailing comma)."""
    with path.open() as f:
        return [{k.strip(): v.strip() for k, v in r.items() if k and k.strip()} for r in csv.DictReader(f)]


def main():
    layers = rows(WORKLOAD)
    ss = rows(SCALESIM)
    ts = {r["shape"]: int(r["cycles"]) for r in rows(TORCHSIM)}
    out = []
    for L, s in zip(layers, ss):
        m, n, k = int(L["M"]), int(L["N"]), int(L["K"])
        ideal = m * n * k / PES
        t, c = ts[f"{m}x{n}x{k}"], int(s["Total Cycles"])
        out.append({"gemm": L["Layer Name"], "M": m, "N": n, "K": k, "ideal_cycles": int(ideal),
                    "pytorchsim_cycles": t, "scalesim_cycles": c,
                    "pytorchsim_util_pct": round(100 * ideal / t, 1), "scalesim_util_pct": round(100 * ideal / c, 1),
                    "scalesim_over_pytorchsim": round(c / t, 2)})
    dest = SCALESIM.parent / "comparison.csv"
    with dest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(f"{r['gemm']:16s} {r['M']:>5}x{r['N']}x{r['K']:<5} ideal {r['ideal_cycles']:>8,}  PyTorchSim "
              f"{r['pytorchsim_cycles']:>8,} ({r['pytorchsim_util_pct']:4.1f}%)  SCALE-Sim {r['scalesim_cycles']:>8,} "
              f"({r['scalesim_util_pct']:4.1f}%)  ratio {r['scalesim_over_pytorchsim']:.2f}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
