#!/usr/bin/env python3
"""GEMM workload of one action chunk (lerobot sample_actions, from extract_gemms.py): FLOPs per phase and the
unique GEMM shapes.

  python3 src/p1_profiling/plots/plot_gemms.py  -> figures/plots/p1/gemm_{phases,shapes}.{pdf,png}
"""
import matplotlib.pyplot as plt

import style as S

PHASE_COLOR = {"vision": S.STAGE_COLOR["vision"], "connector": S.STAGE_COLOR["connector"],
               "vlm": S.STAGE_COLOR["prefix"], "expert": S.STAGE_COLOR["denoise"], "heads": S.MUTED}
PHASE_LABEL = {"vision": "vision encoder", "connector": "connector", "vlm": "VLM (prefix)",
               "expert": "action expert (10 steps)", "heads": "projections / heads"}
PHASE_MARKER = {"vision": "o", "connector": "D", "vlm": "s", "expert": "^", "heads": "v"}


def main():
    rows = S.read("gemms_by_phase.csv")
    fig, ax = plt.subplots(figsize=(5.4, 2.4))
    for i, r in enumerate(rows):
        g = float(r["gflop"])
        ax.barh(i, g, height=0.6, color=PHASE_COLOR[r["phase"]])
        ax.text(g, i, f"  {g:,.1f} GFLOP ({float(r['flop_share_pct']):.1f}%), {int(r['gemm_ops'])} GEMMs",
                va="center", fontsize=7.5, color=S.INK2)
    ax.set_yticks(range(len(rows)), [PHASE_LABEL[r["phase"]] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, max(float(r["gflop"]) for r in rows) * 1.8)
    ax.set_xlabel("GFLOP per action chunk")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"GEMM FLOPs per phase ({sum(float(r['gflop']) for r in rows):,.0f} GFLOP per chunk)", loc="left")
    S.save(fig, "gemm_phases")

    shapes = S.read("gemm_shapes.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for ph in PHASE_COLOR:
        sel = [r for r in shapes if r["phase"] == ph]
        if not sel:
            continue
        ax.scatter([int(r["M"]) for r in sel], [int(r["N"]) * int(r["K"]) for r in sel],
                   s=[12 + 3 * int(r["count"]) ** 0.5 * 6 for r in sel], marker=PHASE_MARKER[ph],
                   color=PHASE_COLOR[ph], edgecolor="white", linewidth=0.8, alpha=0.9, label=PHASE_LABEL[ph])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("M (rows streamed through the array: tokens)")
    ax.set_ylabel("N × K (weight / stationary operand size)")
    ax.set_title("Unique GEMM shapes (marker area ~ √count)", loc="left")
    handles = [plt.Line2D([], [], marker=PHASE_MARKER[ph], ls="", color=PHASE_COLOR[ph], ms=7) for ph in PHASE_COLOR]
    ax.legend(handles, [PHASE_LABEL[ph] for ph in PHASE_COLOR], loc="upper left", bbox_to_anchor=(1.01, 1))
    S.save(fig, "gemm_shapes")


if __name__ == "__main__":
    main()
