#!/usr/bin/env python3
"""Roofline plots: per stage on four NPUs (one row of small multiples), and per single layer on the big NPU.

Filled marker = simulated (arithmetic intensity from the DRAM traffic TOGSim measured, throughput from the cycles).
Hollow marker = the same stage at its compulsory traffic, sitting on the roof (what perfect tiling could reach).
The arrow between them is the cost of the extra DRAM traffic plus the compute inefficiency.

  python3 src/p1_profiling/plots/plot_roofline.py  -> figures/plots/p1/roofline_{stages,layers}.{pdf,png}
"""
import matplotlib.pyplot as plt
import numpy as np

import style as S

# label offsets (points) for the single-layer plot, where several layers sit close together
OFFSET = {"vision_layer": (6, 4), "patch_embed": (6, -10), "vlm_layer": (-52, 4), "connector": (6, -4),
          "expert_cross_layer": (6, -2), "expert_self_layer": (6, -10)}
PANELS = ["timing_heuristic", "sweep_sa_128x128_spad4m", "edge_64x64_lpddr5", "edge_32x32_lpddr5"]


def roof(ax, peak_tflops, gbs):
    ai = np.logspace(-0.5, 3.6, 200)
    ax.plot(ai, np.minimum(peak_tflops, ai * gbs / 1e3), color=S.INK2, lw=1.2, zorder=1)
    ridge = peak_tflops * 1e3 / gbs
    ax.axvline(ridge, color=S.GRID, lw=1, ls=":", zorder=0)
    ax.text(ridge * 1.1, peak_tflops / 250, f"ridge\n{ridge:.0f} FLOP/B", fontsize=7, color=S.MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10 ** -0.5, 10 ** 3.6)
    ax.set_ylim(peak_tflops / 300, peak_tflops * 2.5)


def main():
    sm, st = S.summary(), S.stages()
    fig, axs = plt.subplots(1, 4, figsize=(9.6, 3.0))
    for ax, c in zip(axs, PANELS):
        row = sm[c]
        roof(ax, row["peak_tflops"], row["dram_gbs"])
        for s in S.STAGES:
            r = st[(c, s)]
            ai_sim = r["gflop"] * 1e3 / r["dram_mb"]
            perf_sim = r["gflop"] / r["ms"]                     # GFLOP/ms = TFLOP/s
            ai_cmp = r["gflop"] * 1e3 / r["compulsory_mb"]
            perf_cmp = r["gflop"] / r["ideal_ms"]
            ax.annotate("", (ai_sim, perf_sim), (ai_cmp, perf_cmp),
                        arrowprops=dict(arrowstyle="->", color=S.STAGE_COLOR[s], lw=0.9, alpha=0.7))
            ax.plot(ai_cmp, perf_cmp, "o", mfc="white", mec=S.STAGE_COLOR[s], mew=1.5, ms=7, zorder=3)
            ax.plot(ai_sim, perf_sim, "o", color=S.STAGE_COLOR[s], mec="white", mew=1, ms=8, zorder=4)
        ax.set_title(S.CONFIG_LABEL[c].replace(", ", ",\n", 1) + f", {row['peak_tflops']:g} TFLOP/s", loc="left",
                     fontsize=8)
        ax.set_xlabel("FLOP / DRAM byte", fontsize=8)
    axs[0].set_ylabel("throughput (TFLOP/s)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=S.STAGE_COLOR[s], ms=7) for s in S.STAGES]
    handles += [plt.Line2D([], [], marker="o", ls="", color=S.INK2, ms=7),
                plt.Line2D([], [], marker="o", ls="", mfc="white", mec=S.INK2, ms=7)]
    labels = [S.STAGE_LABEL[s] for s in S.STAGES] + ["simulated", "compulsory traffic (ideal)"]
    fig.legend(handles, labels, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.01), fontsize=7.5,
               handletextpad=0.2, columnspacing=1)
    fig.suptitle("Roofline per stage (fp32)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    S.save(fig, "roofline_stages")

    # single layers on the big NPU (results/timing_heuristic/torchsim/roofline.csv)
    big = sm["timing_heuristic"]
    f = S.freq_ghz(big)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    roof(ax, big["peak_tflops"], big["dram_gbs"])
    for r in S.read("timing_heuristic/torchsim/roofline.csv"):
        g, cyc = float(r["gflop"]), float(r["measured_cyc"])
        perf = g / (cyc / (f * 1e9)) / 1e3
        ai_sim, ai_cmp = float(r["measured_ai"]), float(r["ai_flop_per_byte"])
        perf_cmp = g / (float(r["roofline_cyc"]) / (f * 1e9)) / 1e3
        ax.annotate("", (ai_sim, perf), (ai_cmp, perf_cmp), arrowprops=dict(arrowstyle="->", color=S.MUTED, lw=0.8))
        ax.plot(ai_cmp, perf_cmp, "o", mfc="white", mec=S.INK2, mew=1.3, ms=6, zorder=3)
        ax.plot(ai_sim, perf, "o", color=S.INK, mec="white", ms=7, zorder=4)
        ax.annotate(r["phase"], (ai_sim, perf), textcoords="offset points", xytext=OFFSET.get(r["phase"], (5, -9)), fontsize=7.5, color=S.INK)
    ax.set_xlabel("arithmetic intensity (FLOP / DRAM byte)")
    ax.set_ylabel("throughput (TFLOP/s)")
    ax.set_title("Single layers on the big NPU (2×128×128, HBM2 481 GB/s)", loc="left")
    ax.legend([plt.Line2D([], [], marker="o", ls="", color=S.INK, ms=7),
               plt.Line2D([], [], marker="o", ls="", mfc="white", mec=S.INK2, ms=6)],
              ["simulated", "compulsory traffic (ideal)"], loc="upper left")
    S.save(fig, "roofline_layers")


if __name__ == "__main__":
    main()
