#!/usr/bin/env python3
"""Where the cycles go inside the one-session action-chunk runs, by stage and kernel type, from e2e_kernels.csv.

  python3 src/p1_profiling/plots/plot_kernels.py  -> figures/plots/p1/kernel_classes.{pdf,png}, kernel_timeline.{pdf,png}
"""
import matplotlib.pyplot as plt

import style as S
from parse_e2e import kernel_class

CONFIGS = ["timing_heuristic", "edge_64x64_lpddr5", "edge_32x32_lpddr5"]


def kernels(cfg):
    rows = S.read(f"{cfg}/torchsim_e2e/e2e_kernels.csv")
    for r in rows:
        r["class"] = kernel_class(r["operation"])
        r["start"], r["cycles"] = int(r["start"]), int(r["cycles"])
    return rows


def main():
    sm = S.summary()
    fig, axs = plt.subplots(1, len(CONFIGS), figsize=(7.6, 2.9), sharey=True)
    for ax, cfg in zip(axs, CONFIGS):
        f = S.freq_ghz(sm[cfg]) * 1e6                              # cycles per ms
        tot = {}
        for r in kernels(cfg):
            tot[(r["stage"], r["class"])] = tot.get((r["stage"], r["class"]), 0) + r["cycles"] / f
        for i, s in enumerate(S.STAGES):
            left = 0
            for c in S.CLASSES:
                v = tot.get((s, c), 0)
                ax.barh(i, v, left=left, height=0.62, color=S.CLASS_COLOR[c], edgecolor="white", linewidth=0.8)
                left += v
            ax.text(left, i, f" {left:,.0f}", va="center", fontsize=7.5, color=S.INK2)
        ax.set_xlim(0, max(sum(v for (s2, _), v in tot.items() if s2 == s) for s in S.STAGES) * 1.3)
        ax.set_title(S.CONFIG_LABEL[cfg], loc="left", fontsize=9)
        ax.set_xlabel("ms per action chunk")
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(range(len(S.STAGES)), [S.STAGE_LABEL[s] for s in S.STAGES])
    axs[0].invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=S.CLASS_COLOR[c]) for c in S.CLASSES]
    fig.legend(handles, [S.CLASS_LABEL[c] for c in S.CLASSES], ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), fontsize=7.5, columnspacing=1)
    fig.suptitle("Kernel time per stage in the one-session chunk run", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    S.save(fig, "kernel_classes")

    # timeline of one chunk on the 64x64 edge NPU: one row per stage, kernels coloured by type
    cfg = "edge_64x64_lpddr5"
    f = S.freq_ghz(sm[cfg]) * 1e6
    rows = kernels(cfg)
    fig, ax = plt.subplots(figsize=(7.6, 2.4))
    for i, s in enumerate(S.STAGES):
        for c in S.CLASSES:
            spans = [(r["start"] / f, max(r["cycles"] / f, 0.05)) for r in rows if r["stage"] == s and r["class"] == c]
            if spans:
                ax.broken_barh(spans, (i - 0.32, 0.64), facecolors=S.CLASS_COLOR[c], linewidth=0)
    ax.set_yticks(range(len(S.STAGES)), [S.STAGE_LABEL[s] for s in S.STAGES])
    ax.invert_yaxis()
    ax.set_xlim(0, max(r["start"] + r["cycles"] for r in rows) / f * 1.01)
    ax.set_xlabel("time in the session (ms)")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=S.CLASS_COLOR[c]) for c in S.CLASSES]
    ax.legend(handles, [S.CLASS_LABEL[c] for c in S.CLASSES], ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.32), fontsize=7.5, columnspacing=1)
    ax.set_title(f"Kernel timeline of one action chunk, {S.CONFIG_LABEL[cfg]}", loc="left")
    S.save(fig, "kernel_timeline")


if __name__ == "__main__":
    main()
