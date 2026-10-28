#!/usr/bin/env python3
"""Lone GEMMs on PyTorchSim, for cross-checking SCALE-Sim (inside the torchsim container).

  python src/p1_profiling/torchsim_gemm.py --config configs/pytorchsim/check_64x64_hbm2.yml 1024x768x768 50x2048x720 ...

Each MxNxK runs as C[M,N] = A[M,K] @ B[K,N] (fp32, B is the stationary operand like a Linear weight), one
launch per shape in one TOGSim session with a barrier after each. Output: results/<config stem>/gemm/stdout.log
and gemm_cycles.csv (shape, number of kernels, summed kernel cycles from the per-kernel timeline).
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
import torchsim_common as C  # noqa: E402

KSUM = re.compile(r"Kernel (\d+) execution summary - Started at: \d+ cycles, Total compute time: (\d+) cycles")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shapes", nargs="+", help="MxNxK")
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    a.config = str(Path(a.config).resolve())
    out = C.RESULTS / Path(a.config).stem / "gemm"
    out.mkdir(parents=True, exist_ok=True)
    C.tee_output(out / "stdout.log")
    C.setup(out, a.config)

    import torch
    from Simulator.simulator import TOGSimulator

    dev = torch.device("npu:0")
    g = torch.Generator().manual_seed(0)
    shapes = [tuple(int(x) for x in s.split("x")) for s in a.shapes]
    mm = torch.compile(dynamic=False)(lambda x, w: x @ w)
    launches = []
    with torch.no_grad(), TOGSimulator(config_path=a.config) as sim:
        for m, n, k in shapes:
            x = torch.randn(m, k, generator=g).to(dev)
            w = torch.randn(k, n, generator=g).to(dev)
            launches.append({"shape": f"{m}x{n}x{k}", "first_kernel": sim._next_kernel_id})
            torch.npu.launch_model(mm, x, w, stream_index=0, timestamp=0)
            torch.npu.synchronize()
    C.require_stats(out / "stdout.log")

    text = (out / "stdout.log").read_text(errors="replace")
    cyc = {int(k): int(c) for k, c in KSUM.findall(text)}
    rows = []
    for i, L in enumerate(launches):
        end = launches[i + 1]["first_kernel"] if i + 1 < len(launches) else max(cyc) + 1
        ks = [k for k in cyc if L["first_kernel"] <= k < end]
        rows.append({"shape": L["shape"], "kernels": len(ks), "cycles": sum(cyc[k] for k in ks)})
    (out / "meta.json").write_text(json.dumps({"config": a.config, "launches": launches}, indent=2))
    with (out / "gemm_cycles.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["shape", "kernels", "cycles"])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['shape']:>16s}  {r['kernels']} kernel(s)  {r['cycles']:>10,} cycles", flush=True)


if __name__ == "__main__":
    main()
