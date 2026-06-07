# OpticalMoE Linux Server Setup

This guide applies to the current single-task four-expert experiment and the
separate MNIST + FashionMNIST multitask experiment.

## 1. Is The Code Linux Compatible?

Yes. The current training path is platform independent:

- file paths use `pathlib` or relative paths;
- CUDA selection uses `torch.cuda.is_available()`;
- optical propagation uses `torch.fft`;
- plotting uses the non-interactive Matplotlib `Agg` backend;
- no PowerShell, Windows drive letter, or Windows GUI is required at runtime.

Always run commands from the repository root, the directory containing
`pyproject.toml`, `configs/`, `scripts/`, and `src/`.

Linux filenames are case-sensitive. Do not change capitalization in config,
script, or checkpoint paths.

## 2. Recommended Rental Server

Recommended baseline:

- OS: Ubuntu 22.04 LTS
- GPU: one NVIDIA GPU
- GPU memory: 24 GB minimum
- system memory: 32 GB minimum, 64 GB preferred
- disk: at least 50 GB
- CPU: 8 cores or more
- Python: 3.10

Practical GPU choices:

- RTX 4090 24 GB: recommended price/performance starting point
- RTX A5000 24 GB: usable, generally slower than RTX 4090
- L40S 48 GB: preferred for larger batches and multitask experiments

The pinned CUDA 12.6 command below targets these Ada/Ampere-generation choices.
If you rent a newer Blackwell GPU such as RTX 5090 or B200, use the provider's
preinstalled PyTorch image or select a CUDA 12.8/13.0 wheel from the official
PyTorch installer instead of forcing the CUDA 12.6 command.

Avoid renting an 8 GB GPU for the 700 x 700 four-expert model. The model uses
complex64 fields and repeated FFT propagation, so its activation memory is much
larger than that of an ordinary MNIST classifier.

Start with:

- single-task 24 GB GPU: `batch_size: 2`
- multitask 24 GB GPU: `batch_size: 1` for each task
- 48 GB GPU: try single-task batch 4 and multitask batch 2 only after smoke tests

Increasing batch size does not guarantee higher accuracy. It mainly improves
throughput when memory permits.

## 3. Preferred Server Image

When renting the server, prefer an image that already contains:

- Ubuntu 22.04
- NVIDIA driver
- CUDA 12.6 or CUDA 12.8 compatible environment
- PyTorch 2.x

The NVIDIA driver is the important host component. PyTorch pip wheels include
their CUDA runtime libraries, so manually installing a full CUDA toolkit is
usually unnecessary.

After connecting, check the GPU:

```bash
nvidia-smi
```

The command must display the GPU model, GPU memory, and driver version. If
`nvidia-smi` fails, change the rental image or ask the provider to install the
NVIDIA driver before configuring Python.

## 4. Connect To The Server

From Windows PowerShell:

```powershell
ssh root@SERVER_IP
```

If the provider specifies a port:

```powershell
ssh -p SERVER_PORT root@SERVER_IP
```

Replace `SERVER_IP` and `SERVER_PORT` with the values shown by the provider.

## 5. Upload The Repository

### Method A: SCP

Run this command in Windows PowerShell from the directory that contains the
`opticalmoe` folder:

```powershell
scp -r opticalmoe root@SERVER_IP:/root/
```

With a custom SSH port:

```powershell
scp -P SERVER_PORT -r opticalmoe root@SERVER_IP:/root/
```

The local `runs/` directory may contain large results. It is usually better to
remove old run outputs from the upload copy or transfer only the checkpoints
needed for continued training.

### Method B: Provider File Manager

Upload the complete `opticalmoe` directory to `/root/opticalmoe`. Confirm that
the uploaded directory contains:

```text
pyproject.toml
configs/
scripts/
src/
tests/
```

## 6. Install Miniconda

Skip this section if the server image already provides Conda.

```bash
cd /root
```

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

```bash
bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3
```

```bash
/root/miniconda3/bin/conda init bash
```

```bash
source /root/.bashrc
```

Confirm:

```bash
conda --version
```

## 7. Create The Python Environment

Use Python 3.10 rather than copying the old Windows Python 3.8 environment.

```bash
conda create -n opticalmoe python=3.10 pip -y
```

```bash
conda activate opticalmoe
```

```bash
python --version
```

The output should start with `Python 3.10`.

## 8. Install PyTorch With CUDA

Recommended reproducible installation:

```bash
python -m pip install --upgrade pip
```

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu126
```

The project does not use `torchaudio`, so it is not required.

If the server provider supplies a tested PyTorch image, first inspect it:

```bash
python -c "import torch, torchvision; print('torch=', torch.__version__); print('torchvision=', torchvision.__version__); print('cuda_runtime=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available())"
```

If CUDA is available and `torchvision` imports correctly, keep the provider's
working PyTorch installation instead of reinstalling it.

## 9. Install OpticalMoE

Enter the repository:

```bash
cd /root/opticalmoe
```

Install the remaining pinned dependencies:

```bash
python -m pip install -r requirements-linux.txt
```

Install the local package without replacing the already installed CUDA PyTorch:

```bash
python -m pip install -e . --no-deps
```

## 10. Verify CUDA And FFT

```bash
python -c "import torch; print('torch=', torch.__version__); print('cuda_runtime=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('vram_GB=', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1) if torch.cuda.is_available() else 0)"
```

Test a complex FFT on the GPU:

```bash
python -c "import torch; assert torch.cuda.is_available(); x=torch.randn(1,700,700,device='cuda',dtype=torch.complex64); y=torch.fft.ifft2(torch.fft.fft2(x)); print(y.shape, y.dtype, y.device)"
```

Expected output includes:

```text
torch.Size([1, 700, 700]) torch.complex64 cuda:0
```

## 11. Run Unit Tests

```bash
cd /root/opticalmoe
```

```bash
python -m pytest -q
```

Run only the new four-expert tests when diagnosing:

```bash
python -m pytest tests/test_four_expert_moe_v2.py tests/test_four_expert_multitask_moe.py -q
```

## 12. Run A Single-Task Smoke Test

```bash
python scripts/train_four_expert_moe_v2.py --config configs/four_expert_moe_mnist.yaml --run_name linux_single_smoke --epochs 1 --smoke_test
```

Check that CUDA is being used:

```text
device: cuda
```

Check the output:

```bash
find runs/linux_single_smoke -maxdepth 2 -type f | sort
```

The run should contain `initial_state/`, `architecture_report.json`,
`metrics.csv`, `best.pt`, `last.pt`, and `summary.json`.

## 13. Run A Multitask Smoke Test

```bash
python scripts/train_four_expert_multitask_moe.py --config configs/four_expert_moe_multitask_mnist_fashion.yaml --run_name linux_multitask_smoke --epochs 1 --smoke_test
```

Check:

```bash
find runs/linux_multitask_smoke -maxdepth 3 -type f | sort
```

The run should contain separate initial-state directories for `mnist` and
`fashionmnist`, multitask metric files, checkpoints, and
`task_switching_eval.csv`.

## 14. Start Formal Training

Single-task MNIST:

```bash
python scripts/train_four_expert_moe_v2.py --config configs/four_expert_moe_mnist.yaml --run_name four_expert_mnist_server
```

Single-task FashionMNIST:

```bash
python scripts/train_four_expert_moe_v2.py --config configs/four_expert_moe_fashionmnist.yaml --run_name four_expert_fashion_server
```

Multitask MNIST + FashionMNIST:

```bash
python scripts/train_four_expert_multitask_moe.py --config configs/four_expert_moe_multitask_mnist_fashion.yaml --run_name four_expert_multitask_server
```

## 15. Keep Training Running After SSH Disconnects

Install `tmux`:

```bash
sudo apt-get update && sudo apt-get install -y tmux
```

Create a session:

```bash
tmux new -s opticalmoe
```

Activate the environment and start training inside tmux:

```bash
conda activate opticalmoe
```

```bash
cd /root/opticalmoe
```

```bash
python scripts/train_four_expert_moe_v2.py --config configs/four_expert_moe_mnist.yaml --run_name four_expert_mnist_server
```

Detach without stopping training by pressing `Ctrl+B`, then `D`.

Reconnect later:

```bash
tmux attach -t opticalmoe
```

## 16. Monitor GPU And Output

GPU utilization:

```bash
watch -n 1 nvidia-smi
```

Follow metrics:

```bash
tail -f runs/four_expert_mnist_server/metrics.csv
```

Check disk space:

```bash
df -h
```

Check run directory size:

```bash
du -sh runs/*
```

## 17. Handling CUDA Out Of Memory

Edit the YAML configuration:

```yaml
dataset:
  batch_size: 1
```

For multitask training, set `batch_size: 1` inside both task dataset blocks.

Before restarting a failed process, check that no old Python process still
occupies GPU memory:

```bash
nvidia-smi
```

List training processes:

```bash
ps aux | grep train_four_expert
```

Do not kill a process unless it is an abandoned duplicate. Stopping the active
training process discards progress since the last saved checkpoint.

## 18. Download Results To Windows

Run from Windows PowerShell:

```powershell
scp -r root@SERVER_IP:/root/opticalmoe/runs/four_expert_mnist_server ./
```

With a custom port:

```powershell
scp -P SERVER_PORT -r root@SERVER_IP:/root/opticalmoe/runs/four_expert_mnist_server ./
```

For large run folders, archive them on the server first:

```bash
cd /root/opticalmoe
```

```bash
tar -czf four_expert_mnist_server.tar.gz runs/four_expert_mnist_server
```

Then download the archive:

```powershell
scp root@SERVER_IP:/root/opticalmoe/four_expert_mnist_server.tar.gz ./
```

## 19. Common Problems

### `torch.cuda.is_available()` is `False`

- confirm `nvidia-smi` works;
- confirm a CUDA wheel, not a CPU wheel, was installed;
- reinstall the matching PyTorch CUDA wheel;
- restart the server if the provider changed the driver.

### Dataset download fails

The first run downloads MNIST or FashionMNIST. Check outbound network access.
Alternatively upload the local `data/` directory to `/root/opticalmoe/data/`.

### Process stops when SSH closes

Run training inside `tmux`.

### Training is unexpectedly on CPU

Set `device: auto` or `device: cuda` in the YAML and verify the startup output
contains `device: cuda`.

### Matplotlib display error

The training scripts already use the `Agg` backend and do not require a desktop
environment. Do not install X11 only for plotting.
