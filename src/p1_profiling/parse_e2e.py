#!/usr/bin/env python3
"""Summarise the end-to-end SmolVLA runs into <root>/torchsim_e2e/e2e_summary.csv.

Reads <root>/torchsim_e2e/<name>/timing/stdout.log (stages and the full chunk), compares the chunk
with the stage sums and with the isolated single-layer runs in <root>/torchsim/ (same config and
mapping), and adds an analytic host<->NPU transfer estimate. Standard library only.

The e2e run's per-kernel timeline ("Kernel N execution summary" lines, matched to launches through the
kernel ids in meta.json) is written to e2e_kernels.csv and gives per-stage cycles inside the one session.

  python3 src/p1_profiling/parse_e2e.py [--root results/<config stem>] [--link-gbs 16]
"""
import argparse
import bisect
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from parse_togsim import parse_log, hardware, TOTAL

RESULTS = Path(__file__).resolve().parents[2] / "results"
B = 4                                      # fp32 bytes
RW = re.compile(r"channels 0\.\.\d+ combined .*\| (\d+) reads, (\d+) writes")
ENQ = re.compile(r"Enqueued kernel_id: (\d+), tog_path: \S+, operation: (\S+),")
KSUM = re.compile(r"Kernel (\d+) execution summary - Started at: (\d+) cycles, Total compute time: (\d+) cycles")


def kernel_timeline(run_dir):
    """Rows (kernel, launch, stage, operation, start, cycles) for a run whose meta.json lists its launches."""
    meta = json.loads((run_dir / "meta.json").read_text())
    if "launches" not in meta:
        return []
    text = (run_dir / "stdout.log").read_text(errors="replace")
    ops = {int(k): op for k, op in ENQ.findall(text)}
    # launch_model returns before the device stream has sent all its kernels, so only each launch's first
    # id is known; with a barrier after every launch, a kernel belongs to the last launch starting at or before it
    firsts = sorted((L["first_kernel"], L) for L in meta["launches"])
    rows = []
    for k, start, dur in KSUM.findall(text):
        k = int(k)
        L = firsts[bisect.bisect_right([f for f, _ in firsts], k) - 1][1]
        rows.append({"kernel": k, "launch": L["launch"], "stage": L["stage"], "operation": ops.get(k, ""),
                     "start": int(start), "cycles": int(dur)})
    return rows


def kernel_class(op):
    """Coarse kernel type from the Inductor kernel name (operation field of the TOGSim log)."""
    if "scaled_dot_product" in op:
        return "attention (fused flash)"
    if op.startswith("bmm"):
        return "attention matmul (QK^T, PV)"
    if "amax" in op or "exp" in op:
        return "attention softmax"
    if op.startswith(("addmm", "mm")):
        return "GEMM (linear layers)"
    return "other vector (norms, RoPE, act., copies)"


def launch_spans(rows):
    """launch -> (stage, cycles from its first kernel's start to its last kernel's end)."""
    spans = {}
    for r in rows:
        s, e = r["start"], r["start"] + r["cycles"]
        st, s0, e0 = spans.get(r["launch"], (r["stage"], s, e))
        spans[r["launch"]] = (st, min(s0, s), max(e0, e))
    return {k: (st, e - s) for k, (st, s, e) in spans.items()}
# launches per chunk and (isolated-layer parts, multiplier) per stage, from smolvla_phases.PHASES
STAGES = {"vision": 3, "connector": 3, "prefix": 1, "denoise": 10}


def stats(path):
    t = Path(path).read_text(errors="replace")
    p = parse_log(path)
    rw = RW.findall(t)
    reads, writes = (int(rw[-1][0]), int(rw[-1][1])) if rw else (0, 0)
    cyc, hw = p["cycles"], hardware(path)
    req, f = hw["req_bytes"], hw["freq_mhz"]
    return {"cycles": cyc, "sa_util_pct": round(100 * p["sa_active"] / p["sa_total"], 1) if p["sa_total"] else 0,
            "ms": round(cyc / (f * 1e3), 2),
            "dram_mb": round((reads + writes) * req / 1e6, 1),
            "dram_gbs": round((reads + writes) * req / (cyc / (f * 1e6)) / 1e9, 1) if cyc else 0,
            "gemm_insts": p["gemm_insts"], "vec_insts": p["vec_insts"]}


def cyc(path):
    f = Path(path)
    return int(TOTAL.findall(f.read_text(errors="replace"))[-1]) if f.exists() else 0


def host_bytes():
    """Bytes crossing the host/NPU boundary per chunk (what PyTorchSim does not model)."""
    return {
        "patchified images in (3 cams, 1024x768)": 3 * 1024 * 768 * B,
        "connector out -> host embeds (3 x 64x960)": 3 * 64 * 960 * B,
        "prefix embeddings in (241x960)": 241 * 960 * B,
        "denoise inputs, 10 steps (x_t 50x32 + time emb 50x720)": 10 * (50 * 32 + 50 * 720) * B,
        "denoise outputs, 10 steps (v_t 50x32)": 10 * 50 * 32 * B,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(RESULTS / "timing_heuristic"), help="directory holding torchsim/ and torchsim_e2e/")
    ap.add_argument("--link-gbs", type=float, default=16.0, help="assumed host<->NPU bandwidth, GB/s")
    a = ap.parse_args()
    ROOT = Path(a.root)
    e2e = ROOT / "torchsim_e2e"
    rows = []
    for name in ["vision", "connector", "prefix", "denoise", "denoise_x4", "e2e"]:
        log = e2e / name / "timing" / "stdout.log"
        if log.exists():
            rows.append({"name": name, **stats(log)})
    by = {r["name"]: r for r in rows}

    single = ROOT / "torchsim"
    iso = {p: cyc(single / p / "timing" / "stdout.log")
           for p in ["patch_embed", "vision_layer", "vlm_layer", "expert_self_layer", "expert_cross_layer"]}
    iso_stage = {"vision": iso["patch_embed"] + 12 * iso["vision_layer"], "prefix": 16 * iso["vlm_layer"],
                 "denoise": 8 * (iso["expert_self_layer"] + iso["expert_cross_layer"])}
    out = ROOT / "torchsim_e2e" / "e2e_summary.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + ["isolated_layers_cycles", "stack_vs_isolated_pct"])
        w.writeheader()
        for r in rows:
            i = iso_stage.get(r["name"], 0) if all(iso.values()) else 0
            r["isolated_layers_cycles"] = i or ""
            r["stack_vs_isolated_pct"] = round(100 * (r["cycles"] / i - 1), 1) if i else ""
            w.writerow(r)

    print(f"{'run':11s} {'cycles':>12s} {'ms':>8s} {'SA busy':>8s} {'DRAM MB':>9s} {'GB/s':>7s}")
    for r in rows:
        print(f"{r['name']:11s} {r['cycles']:>12,} {r['ms']:8.2f} {r['sa_util_pct']:7.1f}% "
              f"{r['dram_mb']:9.1f} {r['dram_gbs']:7.1f}")
    if "e2e" in by and all(s in by for s in STAGES):
        chunk = by["e2e"]["cycles"]
        stage_sum = sum(n * by[s]["cycles"] for s, n in STAGES.items())
        print(f"\nchunk (one session)      {chunk:>13,} cycles = {by['e2e']['ms']:.1f} ms")
        print(f"sum of stage runs        {stage_sum:>13,}  (chunk is {100 * (chunk / stage_sum - 1):+.1f}%)")
        if all(iso.values()):                        # single-layer runs exist for this config
            iso_sum = 3 * (iso_stage["vision"] + by["connector"]["cycles"]) + iso_stage["prefix"] + 10 * iso_stage["denoise"]
            print(f"isolated layers x mult   {iso_sum:>13,}  (chunk is {100 * (chunk / iso_sum - 1):+.1f}%)")
        for s, n in STAGES.items():
            share = 100 * n * by[s]["cycles"] / stage_sum
            print(f"  {s:10s} x{n:<2d} {n * by[s]['cycles']:>12,} ({share:4.1f}% of stage sum)")

    run = e2e / "e2e" / "timing"
    if (run / "meta.json").exists():
        rows = kernel_timeline(run)
        if rows:
            kout = e2e / "e2e_kernels.csv"
            with kout.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            per_stage = {}
            for st, c in launch_spans(rows).values():
                per_stage[st] = per_stage.get(st, 0) + c
            inside = sum(per_stage.values())
            print(f"\ninside the e2e session ({len(rows)} kernels, wrote {kout.name}):")
            for st, c in per_stage.items():
                print(f"  {st:10s} {c:>13,} cycles ({100 * c / inside:4.1f}%)")
            by_class = {}
            for r in rows:
                key = (r["stage"], kernel_class(r["operation"]))
                by_class[key] = by_class.get(key, 0) + r["cycles"]
            print("  by kernel type (cycles, share of the chunk):")
            for (st, cl), c in sorted(by_class.items(), key=lambda x: (x[0][0], -x[1])):
                print(f"    {st:10s} {cl:42s} {c:>13,} ({100 * c / inside:4.1f}%)")
            print(f"  sum of launch spans {inside:,} vs session total {by['e2e']['cycles']:,} "
                  f"(gap {by['e2e']['cycles'] - inside:,} cycles: syncs and inter-launch idle)")
        else:
            print("\nno per-launch kernel ids in e2e meta.json (run predates them): no per-kernel timeline")
    hb = host_bytes()
    tot = sum(hb.values())
    freq = hardware(next(e2e.glob("*/timing/stdout.log")))["freq_mhz"]
    link_cycles = tot / (a.link_gbs * 1e9) * freq * 1e6
    print(f"\nhost<->NPU traffic per chunk (NOT simulated): {tot / 1e6:.1f} MB")
    for k, v in hb.items():
        print(f"  {k:58s} {v / 1e6:7.2f} MB")
    print(f"at {a.link_gbs:g} GB/s, no overlap: {link_cycles:,.0f} cycles ({tot / (a.link_gbs * 1e9) * 1e3:.2f} ms)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
