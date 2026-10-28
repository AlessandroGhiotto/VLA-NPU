#!/usr/bin/env python3
"""Replay a finished TOGSim run at instruction level and attribute its time and DRAM traffic (run in the container).

Every session run saved its launch list as <run>/<timestamp>.trace. TOGSim replays it standalone from the compiled
tile graphs in /workspace/PyTorchSim/outputs_<config stem>/ (TOGSim/src/main.cc process_trace_file), cycle for cycle,
and with --log_level trace it logs every instruction (TOGSim/src/Core.cc, CoreTraceLog.cc):
  [cycle][Core c][INST_ISSUED|INST_FINISHED|ASYNC_DMA_ISSUE|DRAM_RESP_DONE|INST_SKIP][INST_ID=i] OPCODE (details)
MOVIN/MOVOUT carry tensor (addr_name), DRAM base, tile size/stride and element bits; COMP its unit
(compute_type 0 vector, 1 matmul, 2 preload) and compute cycles. The log (GBs for vision) is parsed from a pipe.

Time, per kernel: busy intervals of the systolic arrays ([max(issue, finish - compute_cycle), finish) of matmul and
preload COMPs), of the vector unit (vector COMPs) and of the DMA (issue -> last DRAM response, MOVIN and MOVOUT).
Every cycle is classified in priority order: array busy > vector only > exposed memory (only DMA in flight) > other.
A kernel's window runs from its start to the next kernel's start: TOGSim reports a kernel complete before all of its
tiles have issued (late tiles and the final MOVOUTs overlap the next kernel), so the windows, not TOGSim's kernel
durations, tile the run.

Traffic, per kernel and tensor: the DRAM lines (dram_req_size bytes) of each DMA instruction. The `dram=` field is
the tensor's byte pointer plus the tile's offset in *elements* (TileGraphParser.cc:424), and
Instruction::get_dram_address computes (dram + offset_in_bits + 7) >> 3, so tiles overlap in TOGSim's address space
although each instruction's request count is right. Requests are therefore counted with TOGSim's formula (they match
Ramulator's), and data identity uses element addresses dram + element offset (distinct per element and tensor, since
tensors are at least 4 B per element apart). Instruction ids come from one counter as each launched tile graph is
built (Instruction.cc), so each kernel owns a contiguous id range: ids are labelled by the window in which their
MOVIN/COMP issued, a suffix minimum fixes late tiles, and every DMA (also late MOVOUTs) is assigned by id.
A line read by a kernel is
  compulsory    first read of data not produced in this run (weights, run inputs)
  intermediate  first read in this kernel of data written by an earlier kernel (unfused kernel boundary)
  cross-kernel  data not produced in this run but already read by an earlier kernel
  partial-sum   data written earlier by the same kernel (K-split accumulation)
  tiling        further reads of a line inside one kernel, split by external (weights) / produced (activations)
and written lines are intermediate (read by a later kernel), output (never read) or rewrites.

  python src/p1_profiling/replay_trace.py results/<config stem>/torchsim_e2e/<stage>/timing [--config yml]
  -> <run>/replay.log (non-trace lines), <run>/inst_kernels.csv, <run>/inst_traffic.csv
"""
import argparse
from array import array
import csv
import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src" / "common")]
from parse_togsim import parse_log, TOTAL                     # noqa: E402
from parse_e2e import RW, kernel_class                         # noqa: E402
from torchsim_common import CONFIGS, TORCHSIM_DIR             # noqa: E402

INST = re.compile(r"\[(\d+)\]\[Core \d+\]\[(\w+)\s*\]\[INST_ID=(\d+)\] (\w+) \((.*)\)")
FIELD = {k: re.compile(k + r"=(\S+?)[ )\]]") for k in ("addr_name", "dram", "elem_bits", "async")}
SIZE = re.compile(r"size=\[([\d,]+)\]")
STRIDE = re.compile(r"stride=\[([\d,]+)\]")
CTYPE = re.compile(r"compute_type=(\d) compute_cycle=(\d+)")
ENQ = re.compile(r"Enqueued kernel_id: (\d+), tog_path: \S+, operation: (\S+),")
DONE = re.compile(r"Kernel (\d+) has completed .* finished at cycle (\d+)")
KSUM = re.compile(r"Kernel (\d+) execution summary - Started at: (\d+) cycles")
REQ = re.compile(r"(\d+) bytes per request")
CATS = ["array", "vector", "exposed_mem", "other"]
TRAFFIC = ["rd_requests", "rd_compulsory", "rd_intermediate", "rd_cross_kernel", "rd_partial_sum", "rd_tiling_external",
           "rd_tiling_produced", "rd_misaligned", "wr_requests", "wr_intermediate", "wr_output", "wr_rewrites",
           "wr_misaligned"]


class Offsets:
    """Sorted unique element offsets of a DMA tile (size, stride in elements), cached per shape."""
    def __init__(self):
        self.cache = {}

    def __call__(self, size, stride):
        key = (size, stride)
        o = self.cache.get(key)
        if o is None:
            o = np.zeros(1, np.int64)
            for n, s in zip(size, stride):
                o = (o[:, None] + np.arange(n, dtype=np.int64)[None, :] * s).ravel()
            o = np.unique(o)
            if len(self.cache) > 256:
                self.cache.clear()
            self.cache[key] = o
        return o


def lines_of(dram, offs, bits, req):
    """Unique lines (sorted) of the elements dram + offs: offs is sorted, so the line ids are too."""
    l = (dram + offs) * bits // (8 * req)
    return l[np.concatenate(([True], l[1:] != l[:-1]))]


def togsim_requests(dram, offs, bits, req):
    """Request count of Instruction::get_dram_address: lines of ((dram + offs * bits + 7) >> 3)."""
    l = (((dram + 7) >> 3) + offs * bits // 8) // req
    return 1 + int(np.count_nonzero(l[1:] != l[:-1]))


class Traffic:
    """Per-kernel line classification against what earlier kernels of this run wrote and read."""
    def __init__(self):
        self.req = 0                             # bytes per DRAM request, from TOGSim's config printout
        self.written = np.empty(0, np.int64)     # lines written by finished kernels
        self.seen = np.empty(0, np.int64)        # external lines already read
        self.inter_read = []                     # lines read after being written (-> intermediate writes)
        self.kwrites = []                        # (kernel, unique written lines, total written requests)
        self.rows = []                           # per kernel x tensor
        self.kern = {}                           # per kernel classified line counts

    def kernel(self, k, reads, writes):
        """reads/writes: lists of (tensor, lines, togsim_requests, tile)."""
        c = defaultdict(int)
        w_all = np.concatenate([l for _, l, _, _ in writes]) if writes else np.empty(0, np.int64)
        w_u = np.unique(w_all)
        if reads:
            r_all = np.concatenate([l for _, l, _, _ in reads])
            u, cnt = np.unique(r_all, return_counts=True)
            inter = np.isin(u, self.written, assume_unique=True)
            seen = np.isin(u, self.seen, assume_unique=True)
            own = np.isin(u, w_u, assume_unique=True)
            c["rd_compulsory"] = int(np.count_nonzero(~inter & ~seen & ~own))
            c["rd_intermediate"] = int(np.count_nonzero(inter))
            c["rd_cross_kernel"] = int(np.count_nonzero(~inter & seen))
            c["rd_partial_sum"] = int(np.count_nonzero(~inter & ~seen & own))
            extra = cnt - 1
            c["rd_tiling_external"] = int(extra[~inter].sum())
            c["rd_tiling_produced"] = int(extra[inter].sum())
            c["rd_requests"] = sum(n for _, _, n, _ in reads)
            c["rd_misaligned"] = c["rd_requests"] - len(r_all)      # extra requests of TOGSim's shifted addresses
            self.seen = np.union1d(self.seen, u[~inter & ~own])
            self.inter_read.append(u[inter])
        if writes:
            c["wr_rewrites"] = int(len(w_all) - len(w_u))
            c["wr_requests"] = sum(n for _, _, n, _ in writes)
            c["wr_misaligned"] = c["wr_requests"] - len(w_all)
            self.kwrites.append((k, w_u))
            self.written = np.union1d(self.written, w_u)
        self.kern[k] = c
        for dirn, recs in (("read", reads), ("write", writes)):
            per = defaultdict(lambda: [0, [], 0, defaultdict(int)])
            for t, l, n, tile in recs:
                p = per[t]
                p[0] += len(l)
                p[1].append(l)
                p[2] += 1
                p[3][tile] += 1
            for t, (tot, ls, ninst, tiles) in per.items():
                uniq = len(np.unique(np.concatenate(ls)))
                self.rows.append({"kernel": k, "tensor": t, "dir": dirn, "insts": ninst,
                                  "mb": round(tot * self.req / 1e6, 4), "unique_mb": round(uniq * self.req / 1e6, 4),
                                  "reload": round(tot / uniq, 2), "tile": max(tiles, key=tiles.get)})

    def finish(self):
        ir = np.unique(np.concatenate(self.inter_read)) if self.inter_read else np.empty(0, np.int64)
        for k, w_u in self.kwrites:
            n_inter = int(np.count_nonzero(np.isin(w_u, ir, assume_unique=True)))
            self.kern[k]["wr_intermediate"] = n_inter
            self.kern[k]["wr_output"] = len(w_u) - n_inter


def busy(starts, ends, t):
    """For the elementary segments [t[i], t[i+1]): is at least one interval [s, e) active?"""
    c = np.zeros(len(t) + 1, np.int64)
    np.add.at(c, np.searchsorted(t, starts), 1)
    np.add.at(c, np.searchsorted(t, ends), -1)
    return np.cumsum(c)[:len(t) - 1] > 0


def attribute_time(iv, ks, end):
    """Cycles per kernel window [ks[i], ks[i+1]) (last one to `end`) and category, from the unit busy intervals."""
    ke = np.append(ks[1:], end)
    arr = {u: (np.frombuffer(iv[u][0], np.int64), np.frombuffer(iv[u][1], np.int64)) for u in ("sa", "vu", "dma")}
    t = np.unique(np.concatenate([ks, ke] + [a for se in arr.values() for a in se]))
    seg = np.diff(t)
    sa, vu, dm = (busy(*arr[u], t) for u in ("sa", "vu", "dma"))
    cat = np.where(sa, 0, np.where(vu, 1, np.where(dm, 2, 3)))
    k = np.searchsorted(ks, t[:-1], side="right") - 1
    inside = (k >= 0) & (t[:-1] < end)
    nk = len(ks)
    out = {"cat": np.bincount(k[inside] * 4 + cat[inside], weights=seg[inside], minlength=4 * nk).reshape(nk, 4)}
    for name, w in [("sa", sa), ("vu", vu), ("dma", dm), ("hidden", dm & (sa | vu))]:
        m = inside & w
        out[name] = np.bincount(k[m], weights=seg[m], minlength=nk)
    return out


def kernel_of_ids(lab_ids, lab_k, ids):
    """Kernel index of each id in `ids`, from (id, window kernel) labels of MOVIN/COMP issues: ids are contiguous per
    kernel, so a label is lowered to the smallest label of any larger id (a late tile issues in a later window)."""
    o = np.argsort(lab_ids)
    li, lk = lab_ids[o], lab_k[o]
    lk = np.minimum.accumulate(lk[::-1])[::-1]
    first = np.array([li[lk >= k][0] if np.any(lk >= k) else np.iinfo(np.int64).max for k in range(lk.max() + 1)])
    return np.searchsorted(first, ids, side="right") - 1


def remap_trace(trace, config):
    """The launch list with every tile graph path pointing into the config's compile cache. Runs made before the
    caches were named outputs_<config stem> (big-NPU runs of 2026-10-22: outputs_heuristic) point to a renamed
    directory; TOGSim would replay them as empty kernels without an error."""
    cache = f"{TORCHSIM_DIR}/outputs_{config.stem}"
    out = []
    for line in trace.read_text().splitlines():
        f = line.split(",")
        if f[0] == "LAUNCH_KERNEL":
            for i in (4, 5):
                f[i] = re.sub(rf"^{re.escape(TORCHSIM_DIR)}/outputs[^/]*", cache, f[i])
            if not Path(f[4]).exists():
                sys.exit(f"missing tile graph {f[4]}: the compile cache no longer holds this run's kernels")
        out.append(",".join(f))
    tmp = tempfile.NamedTemporaryFile("w", suffix=".trace", delete=False)
    tmp.write("\n".join(out) + "\n")
    tmp.close()
    return Path(tmp.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run dir with stdout.log, meta.json and the .trace file")
    ap.add_argument("--config", help="TOGSim config (default: configs/pytorchsim/<stem of meta.json config>.yml)")
    a = ap.parse_args()
    run = Path(a.run).resolve()
    meta = json.loads((run / "meta.json").read_text())
    config = Path(a.config).resolve() if a.config else CONFIGS / (Path(meta["config"]).stem + ".yml")
    trace = remap_trace(sorted(run.glob("*.trace"))[-1], config)
    sim = Path(TORCHSIM_DIR) / "TOGSim"
    cmd = [str(sim / "build/bin/Simulator"), "--config", str(config), "--models_list", str(trace), "--log_level", "trace"]
    print(" ".join(cmd), flush=True)

    issue = {}                                   # COMP id -> issue cycle
    dma_open = {}                                # DMA id -> issue cycle, closed by FINISHED / DRAM_RESP_DONE
    iv = {u: (array("q"), array("q")) for u in ("sa", "vu", "dma")}       # busy intervals (start, end) per unit
    lab_ids, lab_cyc = array("q"), array("q")    # MOVIN/COMP issues (id, cycle): kernel id ranges
    dmas = []                                    # (id, is_write, tensor, dram, (size, stride), bits)
    ops, kstart = {}, {}
    req, skipped = 0, 0

    proc = subprocess.Popen(cmd, cwd=sim, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1 << 20)
    with (run / "replay.log").open("w") as log:
        for line in proc.stdout:
            if "[trace]" not in line:
                log.write(line)
                if "bytes per request" in line:
                    req = int(REQ.search(line).group(1))
                elif "Enqueued kernel_id" in line:
                    k, op = ENQ.search(line).groups()
                    ops[int(k)] = op
                elif "execution summary" in line:
                    k, st = map(int, KSUM.search(line).groups())
                    kstart[k] = st
                continue
            if "INST_ID=" not in line or "[TOGParser]" in line:
                continue
            m = INST.search(line)
            if not m:
                continue
            cyc, tag, iid, opc, det = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4), m.group(5)
            if opc == "COMP":
                if tag == "INST_ISSUED":
                    issue[iid] = cyc
                    lab_ids.append(iid)
                    lab_cyc.append(cyc)
                elif tag == "INST_FINISHED":
                    ct, cc = map(int, CTYPE.search(det).groups())
                    u = iv["vu"] if ct == 0 else iv["sa"]
                    u[0].append(max(issue.pop(iid, cyc - cc), cyc - cc))
                    u[1].append(cyc)
            elif opc in ("MOVIN", "MOVOUT"):
                if tag == "INST_ISSUED":
                    shape = (tuple(map(int, SIZE.search(det).group(1).split(","))),
                             tuple(map(int, STRIDE.search(det).group(1).split(","))))
                    dmas.append((iid, opc == "MOVOUT", FIELD["addr_name"].search(det).group(1),
                                 int(FIELD["dram"].search(det).group(1), 16), shape,
                                 int(FIELD["elem_bits"].search(det).group(1))))
                    dma_open[iid] = cyc
                    if opc == "MOVIN":
                        lab_ids.append(iid)
                        lab_cyc.append(cyc)
                elif tag in ("INST_FINISHED", "DRAM_RESP_DONE") and iid in dma_open:
                    if tag == "INST_FINISHED" and "ASYNC" in det:
                        continue
                    iv["dma"][0].append(dma_open.pop(iid))
                    iv["dma"][1].append(cyc)
                elif tag == "INST_SKIP":
                    skipped += 1
    if proc.wait() != 0:
        sys.exit(f"TOGSim replay failed (exit {proc.returncode})")
    trace.unlink()
    if not lab_ids or not dmas:
        sys.exit("no instruction events in the replay: was TOGSim built without trace logging?")

    rlog = run / "replay.log"
    orig = int(TOTAL.findall((run / "stdout.log").read_text(errors="replace"))[-1])
    p = parse_log(rlog)
    rd, wr = map(int, RW.findall(rlog.read_text(errors="replace"))[-1])
    kids = sorted(kstart, key=kstart.get)                     # kernels in execution order
    ks = np.array([kstart[k] for k in kids], np.int64)
    t = attribute_time(iv, ks, p["cycles"])

    # traffic, kernel by kernel in id order
    lab_k = np.searchsorted(ks, np.frombuffer(lab_cyc, np.int64), side="right") - 1
    dmas.sort(key=lambda d: d[0])
    kof = kernel_of_ids(np.frombuffer(lab_ids, np.int64), lab_k, np.array([d[0] for d in dmas], np.int64))
    offsets = Offsets()
    tr = Traffic()
    tr.req = req
    togsim_req = [0, 0]
    j = 0
    for ki, k in enumerate(kids):
        reads, writes = [], []
        while j < len(dmas) and kof[j] <= ki:
            _, w, name, dram, shape, bits = dmas[j]
            offs = offsets(*shape)
            n = togsim_requests(dram, offs, bits, req)
            togsim_req[w] += n
            (writes if w else reads).append((name, lines_of(dram, offs, bits, req), n, "x".join(map(str, shape[0]))))
            j += 1
        tr.kernel(k, reads, writes)
    tr.finish()

    rows = []
    for i, k in enumerate(kids):
        c = tr.kern.get(k, {})
        end = ks[i + 1] if i + 1 < len(ks) else p["cycles"]
        r = {"kernel": k, "operation": ops.get(k, ""), "class": kernel_class(ops.get(k, "")), "start": int(ks[i]),
             "cycles": int(end - ks[i])}
        r.update({n: int(t["cat"][i][j]) for j, n in enumerate(CATS)})
        r.update({"sa_busy": int(t["sa"][i]), "vu_busy": int(t["vu"][i]), "dma_busy": int(t["dma"][i]),
                  "dma_hidden": int(t["hidden"][i])})
        r.update({n: c.get(n, 0) for n in TRAFFIC})
        rows.append(r)
    with (run / "inst_kernels.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (run / "inst_traffic.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tr.rows[0]))
        w.writeheader()
        w.writerows(tr.rows)

    tot = lambda n: sum(r[n] for r in rows)
    cyc = p["cycles"]
    print(f"cycles: replay {cyc:,} vs original {orig:,} ({'identical' if cyc == orig else 'DIFFERENT'})")
    print(f"DRAM requests: Ramulator {rd:,} reads / {wr:,} writes; DMA instructions {togsim_req[0]:,} / {togsim_req[1]:,}"
          f" ({100 * (togsim_req[0] + togsim_req[1]) / max(rd + wr, 1) - 100:+.2f}%); skipped MOVINs {skipped:,}")
    print(f"array busy: {tot('sa_busy'):,} (TOGSim SA active {p['sa_active']:,}, summed over arrays)")
    print(f"vector busy: {tot('vu_busy'):,} (TOGSim {p['vec_active']:,}); DMA busy: {tot('dma_busy'):,} (TOGSim {p['dma_active']:,})")
    print("time: " + ", ".join(f"{n} {100 * tot(n) / tot('cycles'):.1f}%" for n in CATS) + f" of {tot('cycles'):,} cycles")
    rq = tot("rd_requests") + tot("wr_requests")
    for n in TRAFFIC:
        if not n.endswith("requests"):
            print(f"  {n:20s} {tot(n) * req / 1e6:10.1f} MB ({100 * tot(n) / max(rq, 1):5.1f}%)")
    print(f"wrote {run / 'inst_kernels.csv'} and inst_traffic.csv")
    if cyc != orig:
        sys.exit(1)


if __name__ == "__main__":
    main()
