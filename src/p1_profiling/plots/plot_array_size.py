#!/usr/bin/env python3
"""Array-size sweep at fixed bandwidth (32/64/128 lanes, LPDDR5 x4 = 51.2 GB/s) plus the big NPU reference.

  python3 src/p1_profiling/plots/plot_array_size.py  -> figures/plots/p1/array_{time,shares}.{pdf,png}
"""
import matplotlib.pyplot as plt

import style as S


def main():
    sm = S.summary()
    cfgs = [c for c in S.ARRAY_SWEEP + ["timing_heuristic"] if c in sm]
    labels = [S.CONFIG_LABEL[c] for c in cfgs]

    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    y = range(len(cfgs))
    h = 0.36
    sim = [sm[c]["chunk_ms"] for c in cfgs]
    ideal = [sm[c]["ideal_chunk_ms"] for c in cfgs]
    ax.barh([i - h / 2 for i in y], sim, height=h, color=S.INK2, label="simulated (PyTorchSim)")
    ax.barh([i + h / 2 for i in y], ideal, height=h, color="#c9c8c2", label="ideal (roofline bound per stage)")
    for i, (a, b) in enumerate(zip(sim, ideal)):
        ax.text(a, i - h / 2, f" {a:,.0f} ms", va="center", fontsize=7.5, color=S.INK)
        ax.text(b, i + h / 2, f" {b:,.0f} ms", va="center", fontsize=7.5, color=S.INK2)
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(sim) * 1.22)
    ax.set_xlabel("action-chunk time (ms)")
    ax.set_title("Chunk time vs array size (51.2 GB/s LPDDR5; big NPU for reference)", loc="left")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    S.save(fig, "array_time")

    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.6), sharey=True)
    S.share_bars(axs[0], labels, [{s: sm[c][f"{s}_pct"] for s in S.STAGES} for c in cfgs], "Simulated (PyTorchSim)")
    S.share_bars(axs[1], labels, [{s: sm[c][f"ideal_{s}_pct"] for s in S.STAGES} for c in cfgs],
                 "Ideal (perfect tiling)")
    S.stage_legend(fig)
    fig.suptitle("Stage shares vs array size", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    S.save(fig, "array_shares")


if __name__ == "__main__":
    main()
