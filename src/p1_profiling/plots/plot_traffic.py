#!/usr/bin/env python3
"""DRAM traffic simulated by PyTorchSim relative to the compulsory traffic (weights + inputs + outputs once).

  python3 src/p1_profiling/plots/plot_traffic.py  -> figures/plots/p1/traffic_ratio.{pdf,png}
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

import style as S

# sequential blue ramp, light (near 1x) to dark
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"])


def main():
    sm, st = S.summary(), S.stages()
    cfgs = [c for c in S.CONFIG_LABEL if c in sm]
    cols = S.STAGES + ["chunk"]
    m = np.zeros((len(cfgs), len(cols)))
    for i, c in enumerate(cfgs):
        for j, s in enumerate(S.STAGES):
            m[i, j] = st[(c, s)]["dram_mb"] / st[(c, s)]["compulsory_mb"]
        sim = sum(S.MULT[s] * st[(c, s)]["dram_mb"] for s in S.STAGES)
        cmp_ = sum(S.MULT[s] * st[(c, s)]["compulsory_mb"] for s in S.STAGES)
        m[i, -1] = sim / cmp_

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    norm = LogNorm(1, 25)
    im = ax.imshow(m, cmap=BLUES, norm=norm, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            ax.text(j, i, f"{v:.1f}×", ha="center", va="center", fontsize=8,
                    color="white" if norm(v) > 0.55 else S.INK, fontweight="bold" if j == len(cols) - 1 else None)
    ax.set_xticks(range(len(cols)), [S.STAGE_LABEL.get(c, "whole chunk") for c in cols])
    ax.set_yticks(range(len(cfgs)), [S.CONFIG_LABEL[c] for c in cfgs])
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axvline(len(cols) - 1.5, color="white", lw=3)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("simulated / compulsory DRAM bytes")
    cb.outline.set_visible(False)
    ax.set_title("DRAM traffic inflation from PyTorchSim's heuristic tiling", loc="left")
    S.save(fig, "traffic_ratio")


if __name__ == "__main__":
    main()
