#!/usr/bin/env python3
"""PyTorchSim vs SCALE-Sim on lone GEMMs (64x64 weight-stationary array, compute-bound), from comparison.csv.

  python3 src/p1_profiling/plots/plot_scalesim.py  -> figures/plots/p1/scalesim_check.{pdf,png}
"""
import matplotlib.pyplot as plt

import style as S

TOOLS = [("pytorchsim_util_pct", "PyTorchSim", "#2a78d6"), ("scalesim_util_pct", "SCALE-Sim v3", "#eb6834")]


def main():
    rows = S.read("scalesim_check/comparison.csv")
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    h = 0.38
    for k, (col, name, color) in enumerate(TOOLS):
        for i, r in enumerate(rows):
            v = float(r[col])
            ax.barh(i + (k - 0.5) * h, v, height=h, color=color, edgecolor="white", linewidth=1,
                    label=name if i == 0 else None)
            ax.text(v, i + (k - 0.5) * h, f" {v:.0f}%", va="center", fontsize=7.5, color=S.INK2)
    ax.set_yticks(range(len(rows)), [f"{r['gemm']}  {r['M']}×{r['N']}×{r['K']}" for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xlabel("array utilisation = ideal cycles (M·N·K / 4096) / simulated cycles (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("Cross-check on single GEMMs, 64×64 weight-stationary array", loc="left")
    S.save(fig, "scalesim_check")


if __name__ == "__main__":
    main()
