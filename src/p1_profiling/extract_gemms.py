#!/usr/bin/env python3
"""Exact GEMM list of one SmolVLA action chunk, recorded from the real lerobot model.

Runs lerobot's SmolVLAFlowMatching.sample_actions (embed_prefix, VLM prefix pass with KV fill, 10 denoise
steps) on CPU in fp32 under a TorchDispatchMode. Every matrix op that reaches the aten dispatcher
(mm, addmm, bmm, baddbmm, convolution, CPU SDPA) is recorded once, with:
  phase   vision | connector | vlm | expert | heads, from the innermost nn.Module's qualified name
  step    denoise step (0-9) for ops inside denoise_step, empty otherwise
  M, N, K, batch: C[M,N] = A[M,K] @ B[K,N], `batch` independent copies (bmm batch, or SDPA batch x heads)
  flops = 2*M*N*K*batch;  bytes = fp32 operand + result bytes, each once (per-op compulsory traffic)

Convolutions are reported as their im2col GEMM (M = N*OH*OW, N = out channels, K = C*kh*kw). SDPA is
split into its two GEMMs (QK^T and PV). Nothing is double counted: aten.linear and aten.matmul are
composite ops that the dispatcher has already decomposed into mm/addmm/bmm when the mode sees them.

Writes results/gemms_exact.csv (one row per op), results/gemm_shapes.csv (unique shapes per phase with
counts: the SCALE-Sim workload) and results/gemms_by_phase.csv. Run in the container:
  python src/p1_profiling/extract_gemms.py
"""
import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src" / "common")]
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

import smolvla_phases as P  # noqa: E402

OUT = REPO / "results"
aten = torch.ops.aten
SDPA = {aten._scaled_dot_product_flash_attention_for_cpu.default, aten._scaled_dot_product_attention_math.default}


def phase_of(name, in_denoise):
    if "vision_model" in name:
        return "vision"
    if "connector" in name:
        return "connector"
    if "lm_expert" in name:
        return "expert"
    if "text_model" in name:
        return "vlm"
    if name in ("", "vlm_with_expert"):
        # attention matmuls run in vlm_with_expert.forward itself, which lerobot calls as .forward()
        # (no module hooks fire), so they appear outside any module
        return "expert" if in_denoise else "vlm"
    return "heads"           # state_proj, action_in_proj, action_time_mlp_*, action_out_proj


def gemm_dims(func, args, out):
    """[(M, N, K, batch, label)] for a matrix op, or [] for anything else."""
    if func in (aten.mm.default,):
        (m, k), n = args[0].shape, args[1].shape[1]
        return [(m, n, k, 1, "mm")]
    if func in (aten.addmm.default,):
        (m, k), n = args[1].shape, args[2].shape[1]
        return [(m, n, k, 1, "addmm")]
    if func in (aten.bmm.default,):
        (b, m, k), n = args[0].shape, args[1].shape[2]
        return [(m, n, k, b, "bmm")]
    if func in (aten.baddbmm.default,):
        (b, m, k), n = args[1].shape, args[2].shape[2]
        return [(m, n, k, b, "baddbmm")]
    if func in (aten.convolution.default,):
        w, o = args[1], out
        groups = args[8]
        k = w.shape[1] * w.shape[2] * w.shape[3]
        return [(o.shape[0] * o.shape[2] * o.shape[3], w.shape[0] // groups, k, groups, "conv(im2col)")]
    if func in SDPA:
        q, k_ = args[0], args[1]                         # (B, H, L, E), (B, Hkv, S, E)
        b, h, l, e = q.shape
        s = k_.shape[2]
        return [(l, s, e, b * h, "sdpa QK^T"), (l, e, s, b * h, "sdpa PV")]
    return []


class Recorder(TorchDispatchMode):
    def __init__(self, stack, step):
        super().__init__()
        self.stack, self.step, self.rows = stack, step, []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for m, n, k, b, label in gemm_dims(func, args, out):
            mod = self.stack[-1] if self.stack else ""
            self.rows.append({
                "idx": len(self.rows), "phase": phase_of(mod, self.step[0] is not None), "step": "" if self.step[0] is None else self.step[0],
                "module": mod, "op": label, "batch": b, "M": m, "N": n, "K": k,
                "flops": 2 * m * n * k * b, "bytes": 4 * b * (m * k + k * n + m * n),
            })
        return out


def main():
    torch.manual_seed(0)
    policy = P.load_policy()
    model = policy.model

    # innermost module name, from pre/post forward hooks on every submodule
    stack = []
    for name, mod in model.named_modules():
        mod.register_forward_pre_hook(lambda m, a, n=name: stack.append(n))
        mod.register_forward_hook(lambda m, a, o: stack.pop() and None)   # a non-None return replaces the output
    step = [None]
    denoise = model.denoise_step

    def counted_denoise(*a, **kw):
        step[0] = 0 if step[0] is None else step[0] + 1
        return denoise(*a, **kw)
    model.denoise_step = counted_denoise

    # inputs as after lerobot's preprocessing: 3 cameras resized/padded to 512x512 in [-1, 1],
    # 48 language tokens (padded to max length), state padded to 32
    images = [torch.rand(1, 3, 512, 512) * 2 - 1 for _ in range(P.N_CAMERAS)]
    img_masks = [torch.ones(1, dtype=torch.bool) for _ in range(P.N_CAMERAS)]
    lang = torch.randint(0, 1000, (1, P.LANG_TOKENS))
    lang_masks = torch.ones(1, P.LANG_TOKENS, dtype=torch.bool)
    state = torch.randn(1, 32)

    rec = Recorder(stack, step)
    with torch.no_grad(), rec:
        actions = model.sample_actions(images, img_masks, lang, lang_masks, state)
    assert actions.shape == (1, P.CHUNK, 32), actions.shape
    assert step[0] == P.NUM_STEPS - 1, step[0]
    rows = rec.rows

    OUT.mkdir(exist_ok=True)
    with (OUT / "gemms_exact.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    shapes = Counter((r["phase"], r["op"], r["batch"], r["M"], r["N"], r["K"]) for r in rows)
    with (OUT / "gemm_shapes.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "op", "batch", "M", "N", "K", "count", "gflop_total"])
        for (ph, op, b, m, n, k), c in sorted(shapes.items()):
            w.writerow([ph, op, b, m, n, k, c, round(2 * m * n * k * b * c / 1e9, 4)])

    by = defaultdict(lambda: {"ops": 0, "flops": 0, "bytes": 0})
    for r in rows:
        by[r["phase"]]["ops"] += 1
        by[r["phase"]]["flops"] += r["flops"]
        by[r["phase"]]["bytes"] += r["bytes"]
    total = sum(v["flops"] for v in by.values())
    with (OUT / "gemms_by_phase.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "gemm_ops", "gflop", "flop_share_pct", "operand_mb"])
        for ph, v in sorted(by.items(), key=lambda x: -x[1]["flops"]):
            w.writerow([ph, v["ops"], round(v["flops"] / 1e9, 3), round(100 * v["flops"] / total, 1),
                        round(v["bytes"] / 1e6, 1)])
            print(f"{ph:10s} {v['ops']:5d} ops {v['flops'] / 1e9:9.2f} GFLOP ({100 * v['flops'] / total:4.1f}%)")
    print(f"total {len(rows)} GEMMs, {total / 1e9:.2f} GFLOP, {len(shapes)} unique shapes; wrote {OUT}/gemms_exact.csv, "
          f"gemm_shapes.csv, gemms_by_phase.csv")


if __name__ == "__main__":
    main()
