#!/usr/bin/env python3
"""Traffic-minimising GEMM tile sizes for PyTorchSim's external mapping (host, standard library).

PyTorchSim's heuristic (PyTorchSimFrontend/mlir/mlir_template.py gemm_combination_mapping) picks, among the tiles
that fit half the scratchpad, the one using the *most scratchpad*; it ignores DRAM traffic. For the VLM MLP
(241 x 2560 x 960) on a 64x64 NPU it picks TILE_M = 64, so every weight tile is read once per 64-row slice of M
(4x). This script enumerates the same candidates with the same feasibility check (tile multiples of the lane count
dividing the padded dims; double-buffered scratchpad, total and per lane) and picks, per GEMM shape, the tile with
the least modelled DRAM traffic:
    W * ceil(M / TILE_M) + X * ceil(N / TILE_N) + Y            (output-stationary over K, as the MLIR template)
ties broken by fewer K steps. The epilogue/prologue buffers are not known here, so an extra output-sized buffer is
reserved (n_extra_node = 2, more than the 0-1 of these kernels).

The result is the JSON read by mlir_gemm_template.py select_tile under codegen_mapping_strategy
"external-then-heuristic" (keys "M_N_K"; shapes absent from it fall back to the heuristic). Only mm/addmm use it:
the attention bmm and flash-SDPA templates keep their own tiling.

  python3 src/p1_profiling/make_mapping.py configs/pytorchsim/<config>.yml -> configs/pytorchsim/mapping_<stem>.json
"""
import argparse
import csv
import json
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
B = 4                                                       # fp32
N_EXTRA = 2


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def pad(x, lanes):
    f = lanes if x > lanes else 8
    return (x + f - 1) // f * f


def candidates(M, N, K, lanes, spad_per_lane):
    """Feasible (TILE_M, TILE_N, TILE_K) as in gemm_combination_mapping (fp32, no prologue)."""
    spad, half_lane = spad_per_lane * lanes // 2, spad_per_lane // 2
    Mp, Np, Kp = pad(M, lanes), pad(N, lanes), pad(max(K, 8), lanes)
    rng = lambda X, Xp: [d * lanes for d in divisors(Xp // lanes)] if X > lanes else [Xp]
    per_lane = lambda a, b: max(a * ((b + lanes - 1) // lanes), 2)
    for tk in rng(K, Kp):
        for tm in rng(M, Mp):
            for tn in rng(N, Np):
                used = (tm * tk + tk * tn + tm * tn * (1 + N_EXTRA)) * B
                lane = (per_lane(tk, tn) + per_lane(tm, tk) + per_lane(tm * (1 + N_EXTRA), tn)) * B
                if used < spad and lane < half_lane:
                    yield tm, tn, tk


def traffic(M, N, K, t):
    tm, tn, _ = t
    return B * (K * N * math.ceil(M / tm) + M * K * math.ceil(N / tn) + M * N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--shapes", default=str(REPO / "results" / "gemm_shapes.csv"))
    a = ap.parse_args()
    cfg = Path(a.config).read_text()
    lanes = int(re.search(r"^vpu_num_lanes:\s*(\d+)", cfg, re.M).group(1))
    spad_per_lane = int(re.search(r"^vpu_spad_size_kb_per_lane:\s*(\d+)", cfg, re.M).group(1)) << 10
    shapes = sorted({(int(r["M"]), int(r["N"]), int(r["K"])) for r in csv.DictReader(open(a.shapes))
                     if r["op"] in ("mm", "addmm", "conv(im2col)") and int(r["M"]) > 1})
    mapping = {}
    print(f"{lanes} lanes, {spad_per_lane >> 10} KB/lane; tiles (M,N,K) and modelled traffic vs the minimum (W+X+Y)")
    for M, N, K in shapes:
        c = list(candidates(M, N, K, lanes, spad_per_lane))
        best = min(c, key=lambda t: (traffic(M, N, K, t), -t[2], -t[0] * t[1]))
        mapping[f"{M}_{N}_{K}"] = {"TILE_M": best[0], "TILE_N": best[1], "TILE_K": best[2]}
        floor = B * (K * N + M * K + M * N)
        print(f"  {M:5d} x {N:5d} x {K:5d}: {best}  {traffic(M, N, K, best) / floor:.2f}x")
    out = Path(a.config).parent / f"mapping_{Path(a.config).stem}.json"
    out.write_text(json.dumps(mapping, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
