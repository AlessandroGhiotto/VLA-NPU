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

docker run -it \
  --ipc=host \
  --name torchsim \
  -v $(pwd):/workspace/PyTorchSim \
  -w /workspace/PyTorchSim \
  ghcr.io/psal-postech/torchsim-ci:v1.1.0 bash

# Run the Docker container
docker run -it \
  --ipc=host \
  --name torchsim \
  -w /workspace/PyTorchSim \
  ghcr.io/psal-postech/torchsim-ci:v1.1.0 bash

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

## APPTAINER

```bash
apptainer build --sandbox torchsim-sandbox \
 docker://ghcr.io/psal-postech/torchsim-ci:v1.1.0

apptainer exec --writable \
 /home/ghi/Documents/HPC-PoliMi/torchsim-sandbox \
 mkdir -p /workspace/3-MDP

apptainer shell --nv \
 --bind /home/ghi/Documents/HPC-PoliMi/3-MDP:/workspace/ \
 --pwd /workspace/PyTorchSim \
 /home/ghi/Documents/HPC-PoliMi/torchsim-sandbox
```
