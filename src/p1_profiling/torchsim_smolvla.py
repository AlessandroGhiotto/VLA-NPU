#!/usr/bin/env python3
"""Run one SmolVLA phase on PyTorchSim (inside the torchsim container).

  python src/p1_profiling/torchsim_smolvla.py vlm_layer --mode check    # NPU vs CPU, functional (Spike)
  python src/p1_profiling/torchsim_smolvla.py vlm_layer --mode timing   # cycles only (timing-only config)

Output (console and TOGSim stats) goes to results/<config stem>/torchsim/<phase>/<mode>/stdout.log (fp32) or
results/<config stem>/torchsim_<dtype>/... The CPU reference is always computed in fp32.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src" / "common")]
import torchsim_common as C  # noqa: E402

CONFIGS = {"check": C.CONFIGS / "functional_heuristic.yml", "timing": C.CONFIGS / "timing_heuristic.yml"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase")
    ap.add_argument("--mode", choices=list(CONFIGS), default="timing")
    ap.add_argument("--config", help="override TOGSim YAML")
    # bf16 is not offered: llc cannot split bf16 vectors, and Spike outputs are read back as fp16
    # (TORCH_TO_NUMPY in Simulator/simulator.py). fp16 works only for layers without norms.
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--outdir", help="default: results/<config stem>/torchsim (fp32) or .../torchsim_<dtype>")
    a = ap.parse_args()

    config = str(Path(a.config or CONFIGS[a.mode]).resolve())   # TOGSim resolves relative paths against its own dir
    if a.outdir is None:
        a.outdir = str(C.RESULTS / Path(config).stem / ("torchsim" if a.dtype == "fp32" else f"torchsim_{a.dtype}"))
    out = Path(a.outdir) / a.phase / a.mode
    out.mkdir(parents=True, exist_ok=True)
    C.tee_output(out / "stdout.log")
    C.setup(out, config)

    import torch
    import smolvla_phases as P
    from Simulator.simulator import TOGSimulator

    policy = P.load_policy()
    mod, args, mult = P.build(a.phase, policy)
    mod = mod.eval()
    del policy  # the phase module holds its own copy of the layer weights
    import gc
    gc.collect()
    with torch.no_grad():
        ref = mod(*args)
    ref = ref[0] if isinstance(ref, tuple) else ref

    device = torch.device("npu:0")
    dt = {"fp32": torch.float32, "fp16": torch.float16}[a.dtype]
    mod = mod.to(dt).to(device)  # one dtype per module: the backend rejects mixed dtypes
    args = tuple(x.to(dt).to(device) if x.is_floating_point() else x.to(device) for x in args)
    opt = torch.compile(dynamic=False)(mod)

    with torch.no_grad(), P.npu_context():
        if a.mode == "check":
            res = opt(*args)
            res = (res[0] if isinstance(res, tuple) else res).cpu().float()
            err = (res - ref).abs()
            # random inputs through real weights give large activations, so judge error relative to scale
            # fp16 has ~3 significant digits, so its rounding error alone exceeds the fp32 thresholds
            tmax, tmean = (2e-2, 1e-2) if a.dtype == "fp32" else (1e-1, 3e-2)
            ok = bool(err.max() <= tmax * ref.abs().max() and err.mean() <= tmean * ref.abs().mean())
            print(f"[{a.phase}] NPU vs CPU: {'PASS' if ok else 'FAIL'} (max abs err {err.max().item():.3e}, "
                  f"mean abs err {err.mean().item():.3e}, ref max |x| {ref.abs().max().item():.3e}, "
                  f"ref mean |x| {ref.abs().mean().item():.3e}, nan {bool(torch.isnan(res).any())})", flush=True)
            if not ok:
                sys.exit(1)
        else:
            with TOGSimulator(config_path=config):
                torch.npu.launch_model(opt, *args, stream_index=0, timestamp=0)
                torch.npu.synchronize()
            C.require_stats(out / "stdout.log")
            print(f"[{a.phase}] timing run done, logs in {out}", flush=True)

    (out / "meta.json").write_text(json.dumps(
        {"phase": a.phase, "mode": a.mode, "dtype": a.dtype, "config": config,
         "mapping": C.mapping_strategy(config), "sdpa": "flash" if a.phase == "vision_layer" else "none", "multiplier": mult,
         "input_shapes": [list(x.shape) for x in args]}, indent=2))


if __name__ == "__main__":
    main()
