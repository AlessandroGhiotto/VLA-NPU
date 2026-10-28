#!/usr/bin/env python3
"""Regenerate every part-1 figure from the committed summary CSVs (host, matplotlib).

  python3 src/p1_profiling/plots/make_all.py  -> figures/plots/p1/*.{pdf,png}

To refresh the CSVs themselves after new simulations: src/p1_profiling/sweep_summary.py and compare_scalesim.py
(host), parse_e2e.py --root results/<config stem> (host), roofline.py --stages (container), attribution_summary.py
and scenarios_summary.py (host, after replay_trace.py in the container).
"""
import plot_array_size
import plot_attribution
import plot_bandwidth
import plot_gemms
import plot_kernels
import plot_roofline
import plot_scalesim
import plot_scenarios
import plot_traffic

for m in (plot_bandwidth, plot_array_size, plot_roofline, plot_traffic, plot_kernels, plot_gemms, plot_scalesim,
          plot_attribution, plot_scenarios):
    m.main()
