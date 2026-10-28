#!/usr/bin/env python3
"""DRAM-bandwidth sweep on the 64x64 edge NPU (LPDDR5 x1..x16): chunk time and stage shares, simulated vs ideal.

  python3 src/p1_profiling/plots/plot_bandwidth.py  -> figures/plots/p1/bandwidth_{time,shares}.{pdf,png}
"""
import matplotlib.pyplot as plt

import style as S


def main():
    sm = S.summary()
    cfgs = [c for c in S.BW_SWEEP if c in sm]
    bw = [sm[c]["dram_gbs"] for c in cfgs]

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    sim = [sm[c]["chunk_ms"] for c in cfgs]
    ideal = [sm[c]["ideal_chunk_ms"] for c in cfgs]
    ax.plot(bw, sim, "-o", color=S.INK, ms=5, label="simulated (PyTorchSim, heuristic tiling)")
    ax.plot(bw, ideal, "--s", color=S.MUTED, ms=5, label="ideal (each stage at its roofline bound)")
    for x, y in zip(bw, sim):
        ax.annotate(f"{y:,.0f}", (x, y), textcoords="offset points", xytext=(4, 5), fontsize=7.5, color=S.INK2)
    for x, y in zip(bw, ideal):
        ax.annotate(f"{y:,.0f}", (x, y), textcoords="offset points", xytext=(4, -11), fontsize=7.5, color=S.INK2)
    ax.set_xlim(bw[0] / 1.25, bw[-1] * 1.4)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(bw, [f"{b:g}" for b in bw])
    ax.minorticks_off()
    ax.set_xlabel("DRAM bandwidth (GB/s), 64×64 array, 8.2 TFLOP/s fp32")
    ax.set_ylabel("action-chunk time (ms)")
    ax.set_title("Chunk time vs DRAM bandwidth", loc="left")
    ax.legend(loc="upper right")
    S.save(fig, "bandwidth_time")

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    labels = [f"{b:g} GB/s" for b in bw]
    S.share_bars(axs[0], labels, [{s: sm[c][f"{s}_pct"] for s in S.STAGES} for c in cfgs],
                 "Simulated (PyTorchSim)")
    S.share_bars(axs[1], labels, [{s: sm[c][f"ideal_{s}_pct"] for s in S.STAGES} for c in cfgs],
                 "Ideal (perfect tiling)")
    S.stage_legend(fig)
    fig.suptitle("Stage shares vs DRAM bandwidth (64×64 array)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    S.save(fig, "bandwidth_shares")


if __name__ == "__main__":
    main()
