#!/usr/bin/env python3
"""Stage shares of the action chunk on the 64x64 NPU under the tiling/fusion scenarios of scenarios_summary.py.

  python3 src/p1_profiling/plots/plot_scenarios.py -> figures/plots/p1/scenarios.{pdf,png}
"""
import matplotlib.pyplot as plt

import style as S

ORDER = ["heuristic", "forced", "fused (est)", "ideal"]
LABEL = {"heuristic": "heuristic tiling (sim.)", "forced": "forced tiling (sim.)",
         "fused (est)": "forced + no kernel round trips (est.)", "ideal": "roofline ideal"}


def main():
    rows = S.read("scenarios.csv")
    bws = sorted({float(r["dram_gbs"]) for r in rows})
    fig, axs = plt.subplots(1, len(bws), figsize=(9.6, 2.5), sharey=True)
    for ax, bw in zip(axs, bws):
        sel = [r for r in rows if float(r["dram_gbs"]) == bw]
        scen = [s for s in ORDER if any(r["scenario"] == s for r in sel)]
        shares = [{r["stage"]: float(r["share_pct"]) for r in sel if r["scenario"] == s} for s in scen]
        S.share_bars(ax, [LABEL[s] for s in scen], shares, f"64×64 NPU, {bw:g} GB/s")
        for i, s in enumerate(scen):
            chunk = next(float(r["chunk_ms"]) for r in sel if r["scenario"] == s)
            ax.text(101, i, f"{chunk:,.0f} ms", va="center", fontsize=7.5, color=S.INK2)
    S.stage_legend(fig, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 0.97, 1))
    S.save(fig, "scenarios")


if __name__ == "__main__":
    main()
