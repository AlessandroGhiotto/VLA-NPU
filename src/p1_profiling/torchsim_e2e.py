#!/usr/bin/env python3
"""Timing simulation of whole SmolVLA stages (or a full action chunk) on PyTorchSim.

  python src/p1_profiling/torchsim_e2e.py vision --launches 1     # one stage, N launches in one TOGSim session
  python src/p1_profiling/torchsim_e2e.py e2e                     # 3x (vision, connector), prefix, 10x denoise, one session

Run inside the torchsim container with the memory-safe settings from CLAUDE.md. Output goes to
<outdir>/<name>/timing/stdout.log where name is the stage (suffixed _xN for N>1 launches) or "e2e".
meta.json lists each launch with the TOGSim kernel ids it enqueued, so parse_e2e.py can split the
session's per-kernel timeline by stage. Timing-only config: values are not computed, only
addresses/shapes drive the cycle count.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src" / "common")]
import torchsim_common as C  # noqa: E402

DEFAULT_CONFIG = str(C.CONFIGS / "timing_heuristic.yml")


def e2e_sequence():
    seq = []
    for _ in range(3):
        seq += ["vision", "connector"]
    return seq + ["prefix"] + ["denoise"] * 10


def main():
    import smolvla_e2e as E
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=E.STAGES + ["e2e"])
    ap.add_argument("--launches", type=int, default=1)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--no-sync", action="store_true",
                    help="queue all launches without a barrier (the simulator then loads every tile graph up "
                         "front: ~3 GB after 10 minutes for a full chunk, which exhausts the 6 GB container cap)")
    ap.add_argument("--outdir", help="default: results/<config stem>/torchsim_e2e")
    a = ap.parse_args()
    a.config = str(Path(a.config).resolve())        # TOGSim resolves relative paths against its own dir
    a.outdir = a.outdir or str(C.RESULTS / Path(a.config).stem / "torchsim_e2e")

    if a.target == "e2e":
        name, seq = "e2e", e2e_sequence()
    else:
        name = a.target if a.launches == 1 else f"{a.target}_x{a.launches}"
        seq = [a.target] * a.launches
    out = Path(a.outdir) / name / "timing"
    out.mkdir(parents=True, exist_ok=True)
    C.tee_output(out / "stdout.log")
    C.setup(out, a.config)

    import torch
    import smolvla_phases as P
    from Simulator.simulator import TOGSimulator

    policy = P.load_policy()
    device = torch.device("npu:0")
    stages = {}
    for st in dict.fromkeys(seq):                    # build each stage once
        mod, args, _ = E.build(st, policy)
        mod = mod.eval().to(device)
        args = tuple(x.to(device) for x in args)
        stages[st] = (torch.compile(dynamic=False)(mod), args)
    del policy
    gc.collect()

    launches = []
    with torch.no_grad(), P.npu_context(), TOGSimulator(config_path=a.config) as sim:
        for i, st in enumerate(seq):
            opt, args = stages[st]
            # ids are assigned on the Python side, one per command; launch_model returns before the device stream
            # has sent all its kernels, so only the first id is meaningful (parse_e2e assigns the rest)
            launches.append({"launch": i, "stage": st, "first_kernel": sim._next_kernel_id})
            torch.npu.launch_model(opt, *args, stream_index=0, timestamp=0)
            if not a.no_sync:
                torch.npu.synchronize()   # stage dependency barrier; also keeps simulator memory bounded
            print(f"[{name}] launched {i + 1}/{len(seq)}: {st}", flush=True)
        torch.npu.synchronize()
    C.require_stats(out / "stdout.log")
    print(f"[{name}] timing run done, logs in {out}", flush=True)
    (out / "meta.json").write_text(json.dumps(
        {"name": name, "mode": "timing", "dtype": "fp32", "config": a.config,
         "mapping": C.mapping_strategy(a.config), "sdpa": "flash (vision)",
         "sequence": seq, "launches": launches}, indent=2))


if __name__ == "__main__":
    main()
