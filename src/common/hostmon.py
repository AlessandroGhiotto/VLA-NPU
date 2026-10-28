#!/usr/bin/env python3
"""Host monitor for crash diagnosis: every 10 s append memory, swap, load, temperatures, fans and the top
processes to logs/hostmon.log, fsync'd so the last lines survive a hard freeze (journald syncs only every ~5 min)."""
import os, subprocess, time
from pathlib import Path

LOG = Path(__file__).with_name("hostmon.log")

def meminfo():
    m = {l.split(":")[0]: int(l.split()[1]) // 1024 for l in open("/proc/meminfo")}
    return f"avail={m['MemAvailable']}MB swapused={m['SwapTotal'] - m['SwapFree']}MB dirty={m['Dirty']}MB"

def temps():
    out = []
    for z in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            out.append(f"{(z / 'type').read_text().strip()}={int((z / 'temp').read_text()) // 1000}")
        except OSError:
            pass
    for f in Path("/sys/class/hwmon").glob("hwmon*/fan*_input"):
        try:
            out.append(f"{f.parent.name}/{f.name}={f.read_text().strip()}")
        except OSError:
            pass
    return " ".join(out)

with LOG.open("a") as fh:
    while True:
        top = subprocess.run(["ps", "-eo", "rss,pcpu,comm", "--sort=-rss"], capture_output=True, text=True).stdout.splitlines()[1:5]
        top = "; ".join(" ".join(l.split()) for l in top)
        fh.write(f"{time.strftime('%F %T')} load={os.getloadavg()[0]:.1f} {meminfo()} {temps()} | {top}\n")
        fh.flush()
        os.fsync(fh.fileno())
        time.sleep(10)
