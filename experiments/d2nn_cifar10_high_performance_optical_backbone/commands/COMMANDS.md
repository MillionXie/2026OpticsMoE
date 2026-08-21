# 服务器命令

在仓库任意位置均可调用脚本。GPU 使用 `nvidia-smi` 中的物理序号，并在脚本内转换成 UUID，避免 CUDA 可见序号错位。

```bash
bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/01_prepare_data.sh
PHYSICAL_GPU_INDEX=3 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/02_smoke_test.sh
PHYSICAL_GPU_INDEX=3 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/03_train_a01.sh
PHYSICAL_GPU_INDEX=3 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/04_evaluate_a01.sh
bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/05_aggregate_a01.sh
PHYSICAL_GPU_INDEX=2 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/06_train_a02_pool16.sh
PHYSICAL_GPU_INDEX=2 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/07_pretrain_a03_cifar100.sh
PHYSICAL_GPU_INDEX=2 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/08_finetune_a04_cifar10.sh
PHYSICAL_GPU_INDEX=1 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/09_refine_a05_from_a01.sh
```

长任务后台启动示例：

```bash
mkdir -p experiments/d2nn_cifar10_high_performance_optical_backbone/runs/main_rgb8
nohup env PHYSICAL_GPU_INDEX=3 \
  bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/03_train_a01.sh \
  > experiments/d2nn_cifar10_high_performance_optical_backbone/runs/main_rgb8/train.log 2>&1 &
```

默认会从 `latest.pt` 自动续训。只有明确希望丢弃同配置断点时才直接在 Python 命令追加 `--force`。

`08_finetune_a04_cifar10.sh` 依赖 A03 的 `best.pt`，必须等 `07_pretrain_a03_cifar100.sh` 正常结束后启动。

`09_refine_a05_from_a01.sh` 依赖 A01 的 `best.pt`，用于低学习率连续精修；它不改变架构。

A05 只有在明确重置该输出目录的训练状态时使用 `FORCE_RESTART=1`，否则默认从自己的 `latest.pt` 续训：

```bash
FORCE_RESTART=1 PHYSICAL_GPU_INDEX=1 \
  bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/09_refine_a05_from_a01.sh
```

正式四组实验必须先生成共同 head-warmup checkpoint，再运行四组。每个方法脚本会依次补齐配置中的 seed 2026、2027、2028，并自动复用已经完成的结果：

```bash
PHYSICAL_GPU_INDEX=1 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/10_prepare_common_formal.sh
PHYSICAL_GPU_INDEX=1 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/11_run_noft_pilot.sh
PHYSICAL_GPU_INDEX=1 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/12_run_bp_pilot.sh
PHYSICAL_GPU_INDEX=2 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/13_run_fa_pretrained_pilot.sh
PHYSICAL_GPU_INDEX=5 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/14_run_fa_random_pilot.sh
bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/15_compare_formal_pilot.sh
```

正式长跑前，先用真实 A03 source 和限量 batch 跑通四组全链路（仅工程检查，不作为实验结果）：

```bash
PHYSICAL_GPU_INDEX=1 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/16_smoke_formal.sh
```

11-14 必须等待 10 正常结束。12-14 可以在不同 GPU 并行，但不得修改 config 或共同 checkpoint。

A07 是正式四组之外的 BP 骨干优化，只测试把光学残差权重硬下限从 0.35 提高到 0.50；它不作为第五种正式方法：

```bash
PHYSICAL_GPU_INDEX=2 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/17_train_a07_high_optical.sh
```

A08–A10 是高光学骨干的电子残差筛选，不增加正式 fixed-feedback 方法数量。三个候选共享 A03 source、A07 的 50-epoch 预算和 `main_min=0.50`，可在不同 GPU 并行：

```bash
mkdir -p experiments/d2nn_cifar10_high_performance_optical_backbone/runs/a08_pointwise_electronic_residual
mkdir -p experiments/d2nn_cifar10_high_performance_optical_backbone/runs/a09_depthwise_electronic_residual
mkdir -p experiments/d2nn_cifar10_high_performance_optical_backbone/runs/a10_depthwise_unet_skips

nohup env PHYSICAL_GPU_INDEX=2 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/18_train_a08_pointwise_residual.sh \
  > experiments/d2nn_cifar10_high_performance_optical_backbone/runs/a08_pointwise_electronic_residual/train.log 2>&1 &
nohup env PHYSICAL_GPU_INDEX=4 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/19_train_a09_depthwise_residual.sh \
  > experiments/d2nn_cifar10_high_performance_optical_backbone/runs/a09_depthwise_electronic_residual/train.log 2>&1 &
nohup env PHYSICAL_GPU_INDEX=5 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/20_train_a10_unet_skips.sh \
  > experiments/d2nn_cifar10_high_performance_optical_backbone/runs/a10_depthwise_unet_skips/train.log 2>&1 &
```

三个脚本都会在 validation-best checkpoint 上自动运行六项测试：normal、optical-off、phase-random、phase-shuffle、electronic-skip-off 和 long-skip-off。先选出 accuracy–optical-dependence Pareto 候选，再对读出头做第二轮受控比较。

第二轮把第一轮胜出的 A08 pointwise bypass 固定，只比较读出头：

```bash
PHYSICAL_GPU_INDEX=2 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/21_train_a11_conv_readout.sh
PHYSICAL_GPU_INDEX=5 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/22_train_a12_dual_pool_readout.sh
```

筛选候选若按预先声明的规则提前停止，保留其 validation-best checkpoint 后用统一脚本补齐六项消融：

```bash
CONFIG_PATH=experiments/d2nn_cifar10_high_performance_optical_backbone/configs/a09_depthwise_electronic_residual.yaml \
PHYSICAL_GPU_INDEX=4 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/23_evaluate_candidate.sh
```

在实验室允许的 1–2M 电子参数预算内，A13 使用约 0.31M 的低分辨率电子残差；加读出头后总电子参数约 0.42M：

```bash
PHYSICAL_GPU_INDEX=4 bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/24_train_a13_lowres_residual.sh
```

A13 架构锁定后，不再改结构或超参数，只补 seed 2026/2027 复现。两个进程写入同一实验目录下
互不重叠的 `seed_<seed>/`，并且显式 `--seed` 时不会在训练结束时抢写 aggregate：

```bash
mkdir -p experiments/d2nn_cifar10_high_performance_optical_backbone/runs/a13_lowres_electronic_residual
nohup env PHYSICAL_GPU_INDEX=4 RUN_SEED=2026 \
  bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/25_train_a13_replica.sh \
  > experiments/d2nn_cifar10_high_performance_optical_backbone/runs/a13_lowres_electronic_residual/train_seed_2026.log 2>&1 &
nohup env PHYSICAL_GPU_INDEX=5 RUN_SEED=2027 \
  bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/25_train_a13_replica.sh \
  > experiments/d2nn_cifar10_high_performance_optical_backbone/runs/a13_lowres_electronic_residual/train_seed_2027.log 2>&1 &
```

两组都生成 `evaluation.json` 后，在 CPU 上汇总三个 seeds：

```bash
bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/26_aggregate_a13_replicas.sh
```

若另有低占用卡，可增加正式种子体系中的 seed 2028 作为确认性第四 seed。它不改变前三 seed
预注册通过规则：

```bash
nohup env PHYSICAL_GPU_INDEX=5 \
  bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/27_train_a13_confirmatory_seed2028.sh \
  > experiments/d2nn_cifar10_high_performance_optical_backbone/runs/a13_lowres_electronic_residual/train_seed_2028.log 2>&1 &
```

仅当 A13 三 seed 复验通过后，准备 P02 共同起点：

```bash
PHYSICAL_GPU_INDEX=<空闲卡> bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/28_prepare_common_a13_formal.sh
```

共同起点冻结并登记 SHA-256 后，每个正式 `method x seed` 使用同一入口；只允许四种方法和
2026/2027/2028 三个 seeds：

```bash
METHOD=bp RUN_SEED=2026 PHYSICAL_GPU_INDEX=<空闲卡> \
  bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/29_run_a13_formal_method.sh
```

12 个结果全部存在后在 CPU 汇总：

```bash
bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/30_compare_a13_formal.sh
```

To queue multiple seeds on the same GPU after an already running launcher exits, use the
checked queue entry point. `WAIT_PID` is optional; every queued item still goes through command 29:

```bash
nohup env PHYSICAL_GPU_INDEX=4 METHOD=bp RUN_SEEDS=2027,2028 WAIT_PID=<launcher-pid> \
  bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/31_queue_a13_formal_method.sh \
  > experiments/d2nn_cifar10_high_performance_optical_backbone/runs/formal_a13_high_performance/launch_logs/bp_seed_2027_2028_queue.log 2>&1 &
```

Start the CPU-side completion watcher once. It waits for the locked 4 methods x 3 seeds and then
runs command 30 automatically:

```bash
nohup bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/32_wait_and_compare_a13_formal.sh \
  > experiments/d2nn_cifar10_high_performance_optical_backbone/runs/formal_a13_high_performance/launch_logs/wait_and_compare.log 2>&1 &
```

P03 deployment-robustness screening is validation-only and locked to training seed 2026. Split the
four formal methods over two idle GPUs:

```bash
nohup env PHYSICAL_GPU_INDEX=4 METHODS_CSV=noft,bp \
  bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/33_run_p03_deployment_screen.sh \
  > experiments/d2nn_cifar10_high_performance_optical_backbone/runs/p03_deployment_robustness_screen/gpu4_noft_bp.log 2>&1 &
nohup env PHYSICAL_GPU_INDEX=5 METHODS_CSV=fa_pretrained,fa_random \
  bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/33_run_p03_deployment_screen.sh \
  > experiments/d2nn_cifar10_high_performance_optical_backbone/runs/p03_deployment_robustness_screen/gpu5_fa.log 2>&1 &
```

After all four screen results exist, aggregate them on CPU:

```bash
bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/34_compare_p03_deployment_screen.sh
```

P03-S2 reuses commands 33/34 with the locked subpixel validation config:

```bash
ROBUSTNESS_CONFIG=experiments/d2nn_cifar10_high_performance_optical_backbone/configs/p03b_deployment_subpixel_screen.yaml \
  METHODS_CSV=bp PHYSICAL_GPU_INDEX=4 \
  bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/33_run_p03_deployment_screen.sh
ROBUSTNESS_CONFIG=experiments/d2nn_cifar10_high_performance_optical_backbone/configs/p03b_deployment_subpixel_screen.yaml \
  bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/34_compare_p03_deployment_screen.sh
```

After P03-S1/S2 lock the severity grid, run the three-training-seed by three-deployment-seed P03-F
test confirmation on two GPUs:

```bash
nohup env PHYSICAL_GPU_INDEX=4 METHODS_CSV=noft,bp \
  bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/35_run_p03_deployment_formal.sh \
  > experiments/d2nn_cifar10_high_performance_optical_backbone/runs/p03_deployment_robustness_formal/gpu4_noft_bp.log 2>&1 &
nohup env PHYSICAL_GPU_INDEX=5 METHODS_CSV=fa_pretrained,fa_random \
  bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/35_run_p03_deployment_formal.sh \
  > experiments/d2nn_cifar10_high_performance_optical_backbone/runs/p03_deployment_robustness_formal/gpu5_fa.log 2>&1 &
```

When all 36 checkpoint/deployment-seed results exist, aggregate on CPU:

```bash
bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/36_compare_p03_deployment_formal.sh
```

P04 studies post-deployment adaptation from one shared ideal BP endpoint. The dual-GPU launcher
puts global rigid shifts on physical GPU 4 and layerwise-independent shifts on physical GPU 5:

```bash
bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/39_launch_p04_adaptation_screen_dual_gpu.sh
```

The underlying single-GPU entry point is command 37. After all four methods and four conditions
finish, aggregate the validation-only screen on CPU:

```bash
bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/38_compare_p04_adaptation_screen.sh
```

After P04 establishes recoverability at 0.125/0.25 pixel, P04-S2 maps the larger 0.5/1/2-pixel
adaptation boundary without changing the four methods or optimizer. Launch global/layerwise
geometry on GPUs 4/5 and aggregate after all 24 results finish:

```bash
bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/42_launch_p04b_large_shift_screen_dual_gpu.sh
bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/41_compare_p04b_large_shift_screen.sh
```

After launching P04-S2, start the CPU-side completion watcher once. It validates that the training
launchers remain alive and automatically runs command 41 when all 24 results exist:

```bash
nohup bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/44_wait_and_compare_p04b_large_shift.sh \
  > experiments/d2nn_cifar10_high_performance_optical_backbone/logs/p04b_wait_and_compare.log 2>&1 &
```

Attribute a completed adaptation result to optical-phase versus electronic updates. This is an
internal mechanism diagnostic, not an additional feedback method:

```bash
PHYSICAL_GPU_INDEX=1 \
  bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/43_run_p04_update_attribution.sh
```

P05 trains one shared source against continuously resampled 0.25--2 pixel global/layerwise
misalignment. It keeps the A13 optical floor and electronic budget unchanged:

```bash
PHYSICAL_GPU_INDEX=4 \
  bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/45_run_p05_misalignment_vaccination.sh
```

Start the CPU watcher once. It waits for the vaccinated checkpoint, launches the same four feedback
groups on held-out global/layerwise deployments, and aggregates all 16 results:

```bash
nohup env GLOBAL_GPU_INDEX=3 LAYERWISE_GPU_INDEX=4 \
  bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/49_wait_p05_and_run_adaptation.sh \
  > experiments/d2nn_cifar10_high_performance_optical_backbone/logs/p05_pipeline.log 2>&1 &
```

Before P06 ImageNet pretraining, audit the existing full ImageNet/CLIP cache and all three proposed
transfer datasets. This is CPU-only and writes one reproducible JSON manifest:

```bash
bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/50_audit_p06_general_backbone_assets.sh
```

P06-E0 uses the real ImageNet Arrow split, matching CLIP memmap entries and the frozen P05
epoch-18 source. It performs two head-only and two exact-BP mini-batches, validates RGB
de-normalisation and checks all eight phase gradients on one physical GPU:

```bash
PHYSICAL_GPU_INDEX=3 \
  bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/51_run_p06_imagenet_smoke.sh
```

After the smoke passes, launch the class-balanced 100k ImageNet screen on physical GPUs 3 and 5.
The launcher is duplicate-safe, records its PID, logs to the run directory, and command 52 resumes
from `last.pt` after an interruption:

```bash
PHYSICAL_GPU_INDICES=3,5 \
  bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/53_launch_p06_imagenet_100k_screen.sh

tail -f experiments/d2nn_cifar10_high_performance_optical_backbone/runs/p06_imagenet_100k_screen/train.log
```
