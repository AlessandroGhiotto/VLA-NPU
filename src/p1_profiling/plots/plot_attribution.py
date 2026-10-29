#!/usr/bin/env python3
"""Where the time and the DRAM traffic of each stage go (instruction-level replays, attribution_summary.py).

  python3 src/p1_profiling/plots/plot_attribution.py
  -> figures/plots/p1/attribution_time.{pdf,png}     per stage and config: time split into array busy / vector only /
                                                      exposed memory / other, in units of the stage's roofline ideal
     figures/plots/p1/attribution_traffic.{pdf,png}  per stage and config: DRAM traffic by cause, in units of the
                                                      compulsory traffic
     figures/plots/p1/attribution_kernels.{pdf,png}  edge 64x64: time per kernel class and category, per chunk
"""
import matplotlib.pyplot as plt

import style as S

CONFIGS = ["timing_heuristic", "sweep_sa_128x128_spad4m", "edge_64x64_lpddr5", "extmap_64x64_ch4", "edge_32x32_lpddr5",
           "sweep_bw_64x64_ch16", "sweep_bw_64x64_ch1", "extmap_64x64_ch1"]
TIME = [("array", "array busy"), ("vector", "vector unit only"), ("exposed_mem", "waiting on DRAM only"),
        ("other", "other (sync, latency)")]
TIME_COLOR = dict(zip([k for k, _ in TIME], ["#2a78d6", "#e87ba4", "#eda100", S.MUTED]))
STREAM_FILL = "#9cc0ec"                        # lighter step of the array blue: the non-streaming part of array time
TRAFFIC = [  # (label, columns)
    ("compulsory (incl. padding)", ["rd_compulsory", "wr_output"]),
    ("kernel-boundary round trips", ["rd_intermediate", "wr_intermediate"]),
    ("weight / input re-reads", ["rd_tiling_external", "rd_cross_kernel"]),
    ("activation re-reads, partial sums", ["rd_tiling_produced", "rd_partial_sum", "wr_rewrites"]),
    ("TOGSim address misalignment", ["rd_misaligned", "wr_misaligned"]),
]
TRAFFIC_COLOR = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", S.MUTED]


def rows(name):
    return [{k: S._num(v) for k, v in r.items()} for r in S.read(name)]


def stacked(ax, i, parts, colors, total_label=None):
    left = 0
    for v, c in zip(parts, colors):
        ax.barh(i, v, left=left, height=0.62, color=c, edgecolor="white", linewidth=0.8)
        left += v
    if total_label:
        ax.text(left, i, " " + total_label, va="center", fontsize=7, color=S.INK2)
    return left


def panels(data, value, colors, xlabel, title, name, ref_label, legend):
    """One panel per stage, one bar per config; value(row) -> (parts, end label, ideal-marker x or None)."""
    fig, axs = plt.subplots(1, 4, figsize=(9.6, 3.6), sharey=True)
    cfgs = [c for c in CONFIGS if any((c, s) in data for s in S.STAGES)]
    for ax, s in zip(axs, S.STAGES):
        xmax = 1
        for i, c in enumerate(cfgs):
            r = data.get((c, s))
            if not r:
                continue
            parts, lab, mark = value(r)
            xmax = max(xmax, stacked(ax, i, parts, colors, lab))
            if mark is not None:
                ax.plot([mark, mark], [i - 0.38, i + 0.38], color=S.INK, lw=1.2)
        ax.axvline(1, color=S.INK, lw=0.8, ls=(0, (3, 2)))
        ax.set_xlim(0, xmax * 1.35)
        ax.set_title(S.STAGE_LABEL[s].split(" ×")[0], loc="left", fontsize=9)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(xlabel, fontsize=8)
    axs[0].set_yticks(range(len(cfgs)), [S.CONFIG_LABEL[c] for c in cfgs])
    axs[0].invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors] + \
        [plt.Line2D([], [], color=S.INK, lw=0.8, ls=(0, (3, 2)))]
    if ref_label[1]:
        handles.append(plt.Line2D([], [], color=S.INK, lw=1.2))
    fig.legend(handles, legend + [l for l in ref_label if l], ncol=3 if len(handles) <= 6 else 4, loc="lower center",
               bbox_to_anchor=(0.5, -0.06), fontsize=7.5, columnspacing=1)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    S.save(fig, name)


def main():
    st = {(r["config"], r["stage"]): r for r in rows("attribution_stages.csv")}

    def t(r):
        ideal = r["ideal_ms"]
        stream = min(r["sa_streaming_ms"], r["array_ms"])
        parts = [stream / ideal, (r["array_ms"] - stream) / ideal] + [r[f"{k}_ms"] / ideal for k, _ in TIME[1:]]
        return parts, f"{r['ms']:.3g} ms", None
    panels(st, t, [TIME_COLOR["array"], STREAM_FILL] + [TIME_COLOR[k] for k, _ in TIME[1:]], "time / roofline ideal",
           "Where each stage's time goes (one instance), relative to its roofline ideal", "attribution_time",
           ["ideal (= 1)", None],
           ["array streaming", "array fill / drain"] + [l for _, l in TIME[1:]])

    def q(r):
        comp = r["compulsory_mb_roofline"]
        parts = [sum(r[f"{c}_mb"] for c in cols) / comp for _, cols in TRAFFIC]
        return parts, f"{r['dram_mb']:,.0f} MB", None
    panels(st, q, TRAFFIC_COLOR, "DRAM traffic / compulsory",
           "Where each stage's DRAM traffic comes from (one instance), relative to the compulsory traffic",
           "attribution_traffic", ["compulsory (= 1)", None], [l for l, _ in TRAFFIC])

    # per kernel class on the 64x64 edge NPU, per chunk
    cfg = "edge_64x64_lpddr5"
    cl = [r for r in rows("attribution_classes.csv") if r["config"] == cfg]
    fig, axs = plt.subplots(1, 4, figsize=(9.6, 2.6), sharey=True)
    for ax, s in zip(axs, S.STAGES):
        by = {r["class"]: r for r in cl if r["stage"] == s}
        xmax = 0
        for i, c in enumerate(S.CLASSES):
            r = by.get(c)
            if r:
                parts = [r[f"{k}_ms"] * S.MULT[s] for k, _ in TIME]
                xmax = max(xmax, stacked(ax, i, parts, [TIME_COLOR[k] for k, _ in TIME], f"{sum(parts):,.0f}"))
        ax.set_xlim(0, max(xmax, 1) * 1.3)
        ax.set_title(S.STAGE_LABEL[s], loc="left", fontsize=9)
        ax.set_xlabel("ms per action chunk", fontsize=8)
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(range(len(S.CLASSES)), [S.CLASS_LABEL[c] for c in S.CLASSES])
    axs[0].invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=TIME_COLOR[k]) for k, _ in TIME]
    fig.legend(handles, [l for _, l in TIME], ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.06), fontsize=7.5)
    fig.suptitle(f"Time per kernel type and what it waits on, {S.CONFIG_LABEL[cfg]}", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    S.save(fig, "attribution_kernels")


if __name__ == "__main__":
    main()
