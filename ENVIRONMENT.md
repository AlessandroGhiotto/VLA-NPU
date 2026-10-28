## SWAPFILE

```bash
sudo swapoff /swapfile 2>/dev/null
sudo rm -f /swapfile
sudo touch /swapfile
sudo chattr +C /swapfile

sudo dd if=/dev/zero of=/swapfile bs=1M count=16384 status=progress
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# MAKE IT PERMANENT
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## DOCKER

```bash
docker pull ghcr.io/psal-postech/torchsim-ci:v1.1.0

# Run the Docker container
# docker run -it \
#   --ipc=host \
#   --name torchsim \
#   -w /workspace/PyTorchSim \
#   ghcr.io/psal-postech/torchsim-ci:v1.1.0 bash


### Mounting 3-MDP folder
# (first install nvidia-container toolkit)
docker run -it \
  --ipc=host \
  --gpus all \
  --name torchsim \
  -v $(pwd):/workspace/3-MDP \
  -w /workspace/PyTorchSim \
  ghcr.io/psal-postech/torchsim-ci:v1.1.0 bash

pip install "lerobot[smolvla]"

docker start -ai torchsim
```

- `-v $(pwd):/workspace/PyTorchSim`: Connects your host directory ($(pwd)) to /workspace/PyTorchSim inside the container.
- `-w /workspace/PyTorchSim`: Sets your starting directory inside the container.
- `--name torchsim`: Assigns a custom name to this container instance so you can easily re-enter it later.
- `--ipc=host`: Shares host IPC memory (crucial for PyTorch / multi-processing).

### Container lifecycle

- `docker start -ai my_env`: Boots up your stopped container and opens an interactive shell (equivalent to apptainer shell).
- `exit`: Stops the container and returns to your host terminal.
- `docker exec -it my_env bash`: Opens a second shell inside an already running container (great for debugging while a script runs).
- `docker stop my_env`: Safely stops the running container from the host.

### Image & Download Management

- `docker pull <image_name>`: Downloads image layers with resume support if the network drops.
- `docker images`: Shows all downloaded image blueprints saved on your machine.
- `docker ps`: Shows currently running containers.
- `docker ps -a`: Shows all containers, including stopped ones.

### Cleanup

- `docker rm my_env`: Removes the container instance (your local code on the host remains safe).
- `docker rmi <image_name>`: Removes the cached image blueprint to free up disk space.

## Running SmolVLA phases on PyTorchSim

```bash
docker update --memory 9g --memory-swap 12g torchsim   # protect the host from OOM
docker start torchsim
docker exec -it torchsim bash -lc 'cd /workspace/3-MDP && python src/p1_profiling/torchsim_smolvla.py vlm_layer --mode timing'
python3 src/common/parse_togsim.py          # -> results/timing_heuristic/torchsim/phase_summary.csv
```

Phases: `patch_embed connector vision_layer vlm_layer expert_self_layer expert_cross_layer`. Use `--mode check` for NPU-vs-CPU validation (not for `vision_layer`, see `CLAUDE.md`). Run one phase at a time.

### core dump freeze

```bash
sudo mkdir -p /etc/systemd/coredump.conf.d && printf '[Coredump]\nStorage=none\nProcessSizeMax=0\n' | sudo tee /etc/systemd/coredump.conf.d/no-dumps.conf
```

What it does:

- It creates /etc/systemd/coredump.conf.d/no-dumps.conf with Storage=none and ProcessSizeMax=0.
- When a process crashes, systemd-coredump will only log the crash and won't read, compress or store the core image.
- A dozen simultaneous crashes then cost almost nothing, instead of a dozen multi-GB dumps running on the host outside Docker's memory cap.
- systemd-coredump is started fresh for each crash, so it takes effect right away; no reboot or service restart is needed.

Side effect:

- No core dumps are saved for any program on the laptop, including desktop apps. coredumpctl will still list crashes but have no core to debug.
- For this project that doesn't matter.

To undo: `sudo rm /etc/systemd/coredump.conf.d/no-dumps.conf`
