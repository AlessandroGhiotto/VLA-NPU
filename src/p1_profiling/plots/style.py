"""Shared look, data loading and output for the part-1 plots (matplotlib, host).

Every plot reads only the committed summary CSVs under results/, so the figures can be regenerated from a clean
checkout without the raw TOGSim logs. Figures go to figures/plots/p1/<name>.{pdf,png}.

Colours: the stage and kernel-class slots below are a fixed categorical order (validated for adjacent-pair
colour-vision-deficiency separation); a colour always means the same stage / class in every figure.
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RESULTS = REPO / "results"
OUT = REPO / "figures" / "plots" / "p1"
sys.path[:0] = [str(HERE.parent), str(REPO / "src" / "common")]

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
STAGES = ["vision", "connector", "prefix", "denoise"]
MULT = {"vision": 3, "connector": 3, "prefix": 1, "denoise": 10}
STAGE_COLOR = dict(zip(STAGES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
STAGE_LABEL = {"vision": "vision ×3", "connector": "connector ×3", "prefix": "prefix (VLM)", "denoise": "denoise ×10"}
CLASSES = ["GEMM (linear layers)", "attention (fused flash)", "attention matmul (QK^T, PV)", "attention softmax",
           "other vector (norms, RoPE, act., copies)"]
CLASS_COLOR = dict(zip(CLASSES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
CLASS_LABEL = dict(zip(CLASSES, ["GEMM (linear)", "flash attention (vision)", "attention matmul (QKᵀ, PV)",
                                 "attention softmax", "other vector"]))

# config stem -> short label; the order is the order in multi-config figures
CONFIG_LABEL = {
    "edge_32x32_lpddr5": "32×32, 2 MB, 51 GB/s",
    "sweep_sa_32x32_spad4m": "32×32, 4 MB, 51 GB/s",
    "sweep_bw_64x64_ch1": "64×64, 12.8 GB/s",
    "sweep_bw_64x64_ch2": "64×64, 25.6 GB/s",
    "edge_64x64_lpddr5": "64×64, 51 GB/s",
    "sweep_bw_64x64_ch8": "64×64, 102 GB/s",
    "sweep_bw_64x64_ch16": "64×64, 205 GB/s",
    "extmap_64x64_ch1": "64×64, 12.8 GB/s, forced tiling",
    "extmap_64x64_ch4": "64×64, 51 GB/s, forced tiling",
    "sweep_sa_128x128_spad4m": "128×128, 51 GB/s",
    "timing_heuristic": "big: 2×128×128, 481 GB/s",
}
BW_SWEEP = ["sweep_bw_64x64_ch1", "sweep_bw_64x64_ch2", "edge_64x64_lpddr5", "sweep_bw_64x64_ch8", "sweep_bw_64x64_ch16"]
ARRAY_SWEEP = ["edge_32x32_lpddr5", "edge_64x64_lpddr5", "sweep_sa_128x128_spad4m"]

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 150,
    "savefig.bbox": "tight", "font.family": "DejaVu Sans", "lines.linewidth": 2,
})


def read(rel):
    with (RESULTS / rel).open() as f:
        return list(csv.DictReader(f))


def summary():
    """config -> row of results/sweep_summary.csv (numbers converted)."""
    return {r["config"]: {k: _num(v) for k, v in r.items()} for r in read("sweep_summary.csv")}


def stages():
    """(config, stage) -> row of results/sweep_stages.csv."""
    return {(r["config"], r["stage"]): {k: _num(v) for k, v in r.items()} for r in read("sweep_stages.csv")}


def freq_ghz(row):
    """Core clock from a sweep_summary row: peak = arrays x dim^2 MACs x 2 FLOP per cycle."""
    n, d, _ = (int(x) for x in row["array"].split("x"))
    return row["peak_tflops"] * 1e3 / (2 * n * d * d)


def _num(v):
    try:
        return float(v)
    except ValueError:
        return v


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote {OUT.relative_to(REPO)}/{name}.{{pdf,png}}")


def share_bars(ax, labels, shares, title):
    """Horizontal 100% stacked bars, one per label; shares = list of {stage: pct}. Labels >= 8% are printed."""
    for i, sh in enumerate(shares):
        left = 0
        for s in STAGES:
            ax.barh(i, sh[s], left=left, height=0.62, color=STAGE_COLOR[s], edgecolor="white", linewidth=1)
            if sh[s] >= 8:
                ax.text(left + sh[s] / 2, i, f"{sh[s]:.0f}", ha="center", va="center", fontsize=7.5, color=INK)
            left += sh[s]
    ax.set_yticks(range(len(labels)), labels)
    if not ax.yaxis_inverted():                  # axes sharing y are inverted once, by the first call
        ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of the action-chunk time (%)")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)


def stage_legend(fig, **kw):
    handles = [plt.Rectangle((0, 0), 1, 1, color=STAGE_COLOR[s]) for s in STAGES]
    fig.legend(handles, [STAGE_LABEL[s] for s in STAGES], ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               **kw)
