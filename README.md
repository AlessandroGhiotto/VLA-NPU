# Multidisciplinary Project

### MSc in High Performance Computing Engineering - PoliMi

## Workload-Driven Exploration of Edge NPU Architectures for Vision-Language-Action (VLA) Models

### 1. Presentations (`presentations/`)

### 2. Continuous Report (`report/`)
- Latest compiled report: [report.pdf](https://github.com/AlessandroGhiotto/VLA-NPU/releases/latest/download/report.pdf) (PDFs are attached to the [releases](https://github.com/AlessandroGhiotto/VLA-NPU/releases), not versioned; `p1`: part 1, SmolVLA profiling on PyTorchSim).

### 3. Shared Figures (`figures/`)

### 4. Code (`src/`), organised per project phase
- `src/common/`: phase-independent helpers (TOGSim log parser, PyTorchSim runner setup, host monitor).
- `src/p1_profiling/`: part 1, end-to-end profiling of SmolVLA on PyTorchSim (model re-implementation, runners, analysis) and `plots/` (all part-1 figures, from the summary CSVs).

### 5. Configs (`configs/`)
- `configs/pytorchsim/`: simulated NPUs (big TPUv3-like, edge 32x32/64x64, bandwidth and array sweeps) and `jobs/` lists.
- `configs/scalesim/`: SCALE-Sim cross-check.

### 6. Results
- `results/`: summary CSVs only (raw simulator logs are not versioned), one directory per NPU config.
- `figures/plots/p1/`: part-1 figures (PDF for the report, PNG for markdown).
- `ENVIRONMENT.md`: container setup.
