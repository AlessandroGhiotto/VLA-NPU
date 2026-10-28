"""Shared setup for the PyTorchSim runners (src/p1_profiling/torchsim_*.py). Run inside the container."""
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIGS = REPO / "configs" / "pytorchsim"
RESULTS = REPO / "results"
TORCHSIM_DIR = os.environ.get("TORCHSIM_DIR", "/workspace/PyTorchSim")


def tee_output(path):
    """Send this process's stdout/stderr, and those of every child (the TOGSim process), to `path` and the console.

    In session mode (with TOGSimulator) TOGSim writes its stats only to the inherited stdout and no .log file
    (Simulator/simulator.py leaves stdout=PIPE commented out), so the stats exist only if fd 1 is captured.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    tee = subprocess.Popen(["tee", str(path)], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), 1)
    os.dup2(tee.stdin.fileno(), 2)
    tee.stdin.close()


def setup(out, config):
    """Environment PyTorchSim reads at import / first compile. Call before importing torch."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ["TORCHSIM_LOG_PATH"] = str(out)
    os.environ["TOGSIM_CONFIG"] = str(config)
    # One compile cache per config file: the Inductor FX graph cache under TORCHSIM_DUMP_PATH is not keyed by the
    # TOGSim config, so a shared cache replays kernels generated for another mapping strategy or lane/scratchpad
    # size (seen 2026-10-22: an autotune e2e run reusing heuristic prefix/denoise kernels).
    os.environ.setdefault("TORCHSIM_DUMP_PATH", f"{TORCHSIM_DIR}/outputs_{Path(config).stem}")
    # Autotune compiles every candidate (mlir-opt/llc + gem5) in Inductor's AsyncCompile pool, sized
    # min(32, cpu_count) by default; that pool froze the laptop. 2 workers (1 would crash: the backend calls
    # .result() on the submit() return value).
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")
    sys.path.insert(0, TORCHSIM_DIR)


def mapping_strategy(config):
    m = re.search(r"^codegen_mapping_strategy:\s*(\S+)", Path(config).read_text(), re.M)
    return m.group(1) if m else "unknown"


def require_stats(log):
    """Exit 1 unless TOGSim's final stats reached the log. When TOGSim dies (e.g. OOM-killed by the container cap)
    the error is raised on the device stream thread and the Python process would otherwise exit 0."""
    if "Total execution cycles" not in Path(log).read_text(errors="replace"):
        print(f"ERROR: no 'Total execution cycles' in {log}: TOGSim did not finish", flush=True)
        sys.exit(1)
