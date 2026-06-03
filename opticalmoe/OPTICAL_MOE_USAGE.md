# OpticalMoE 大画布使用说明

## 先回答当前结果为什么很低

之前的 `runs/optical_moe_eval_smoke`、`runs/optical_moe_trainscratch_smoke2` 这类结果主要是在检查工程链路：

- 大画布输入是否能放到中心。
- fixed grating 是否能把光送到 left/right expert。
- entrance de-tilt 是否存在。
- translated detector 是否能输出 10 类 logits。
- metrics、summary、checkpoint、light field 图是否能保存。

这些 smoke run 没有加载 `runs/mnist_002/best.pt`。如果没有显式写 checkpoint 路径，`OpticalMoEClassifier` 的 expert phase 就是随机初始化，所以准确率接近 10 类随机水平是正常的。

真正有分类能力的旧 single-expert checkpoint 是：

```text
runs/mnist_002/best.pt
runs/fashionmnist_001/best.pt
```

其中 `runs/mnist_002/summary.json` 记录 MNIST test accuracy 约为 0.8199。

## 推荐启动方式

现在推荐把参数写到 YAML 里，命令只保留一行：

```powershell
python scripts/run_optical_moe.py --config configs/optical_moe_eval_mnist_migrated.yaml
```

命令行里的 `--mode`、`--run_name`、`--left_ckpt`、`--batch_size` 等都只是临时覆盖。平时不需要记这些参数。

## 最重要的几类 checkpoint

旧 single-expert checkpoint：

```text
runs/mnist_002/best.pt
```

它来自 `scripts/train.py`，参数 key 通常长这样：

```text
phase_layers.0.raw_phase
phase_layers.1.raw_phase
...
```

这类 checkpoint 需要用：

```yaml
checkpoints:
  left_ckpt: runs/mnist_002/best.pt
  left_config: runs/mnist_002/config.yaml
```

大画布 OpticalMoE checkpoint：

```text
runs/mnist_left_bank_train/best.pt
```

它来自 `scripts/run_optical_moe.py`，已经是完整大画布 MoE 模型 state_dict。再次评估同一个模型时用：

```yaml
checkpoints:
  moe_ckpt: runs/mnist_left_bank_train/best.pt
```

如果你分别训练了 left 和 right bank expert，再拼成双 expert，用：

```yaml
checkpoints:
  left_moe_ckpt: runs/mnist_left_bank_train/best.pt
  right_moe_ckpt: runs/fashion_right_bank_train/best.pt
```

## 推荐实验顺序

### 1. 迁移旧 MNIST expert 到大画布 left expert

用途：判断旧的 600x600 单 expert 放进 800x1600 bank geometry 后掉多少精度。

```powershell
python scripts/run_optical_moe.py --config configs/optical_moe_eval_mnist_migrated.yaml
```

看：

```text
runs/mnist_left_migrated_eval/summary.md
runs/mnist_left_migrated_eval/metrics.csv
```

如果 accuracy 明显低于 `0.8199`，说明大画布长距离传播、grating、detector 平移或 aperture mask 带来了迁移损失。

### 2. 小规模 debug 训练

用途：确认大画布训练能正常反向传播。这个不是最终训练。

```powershell
python scripts/run_optical_moe.py --config configs/optical_moe_train_mnist_left_debug.yaml
```

它默认：

```text
smoke_test: true
batch_size: 2
epochs: 3
```

所以会比完整 MNIST 快很多。

### 3. compensation-only 微调

用途：冻结旧 expert，只训练 grating 上的 residual prompt，用来补偿入口波前误差。

```powershell
python scripts/run_optical_moe.py --config configs/optical_moe_finetune_mnist_left_comp.yaml
```

这个策略对应：

```yaml
training:
  freeze_policy: compensation_only
```

脚本会自动把 `prompt_mode` 变成 `trainable_residual_on_grating`。

### 4. 组装 left/right 双 expert

用途：把分别训练好的 bank expert 组合成 10 类 paired detector sum。

```powershell
python scripts/run_optical_moe.py --config configs/optical_moe_eval_mixed_assembled.yaml
```

注意：当前第一版还没有 input-dependent router，所以 `mixed_mnist_fashion` 主要用于 readout/debug，不代表最终 OpticalMoE 路由能力。

### 5. 手动任务切换

如果当前任务已知，可以直接通过 YAML 指定任务，让所有输入都走指定 expert。

MNIST 任务走 left expert：

```powershell
python scripts/run_optical_moe.py --config configs/optical_moe_eval_task_mnist_left.yaml
```

FashionMNIST 任务走 right expert：

```powershell
python scripts/run_optical_moe.py --config configs/optical_moe_eval_task_fashion_right.yaml
```

对应的关键配置是：

```yaml
evaluation:
  routing_mode: fixed_task
  current_task: mnist
```

或者：

```yaml
evaluation:
  routing_mode: fixed_task
  current_task: fashionmnist
```

`routing_mode` 可以是：

```text
fixed_task      从配置文件指定当前任务，实现手动任务切换
task_aware      mixed 数据集诊断模式，使用 task_id 做 oracle routing
model_default   不额外切换，使用 model.target_side
```

现在的 `fixed_task` 仍然是固定光栅 prompt，不是训练出来的 router。它相当于“低速任务控制信号已知”的版本。

## 多类别可视化

评估脚本现在会按类别保存多个样本，而不是只保存第一个样本。

配置项：

```yaml
visualization:
  num_samples_per_class: 1
  max_classes: 10
  max_light_field_samples: 4
```

输出位置：

```text
runs/<run_name>/sample_outputs/class_samples_overview.png
runs/<run_name>/detector_energies/detector_energy_left_right_bars_samples.png
runs/<run_name>/light_fields/sample_000_label_.../
runs/<run_name>/light_fields/sample_001_label_.../
```

`class_samples_overview.png` 用来看不同类别的输入和预测。

`detector_energy_left_right_bars_samples.png` 用来看每个样本左右 detector 的 10 类响应。

`light_fields/sample_xxx/` 保存少量样本的完整传播过程，文件会比较多，所以默认只保存前 4 个选中样本。

## 为什么完整 train_scratch 很慢

大画布场大小是：

```text
800 x 1600 = 1,280,000 complex pixels
```

每个 batch 要做多段 `torch.fft` 传播。完整 MNIST 训练集约 54,000 张训练图，如果 `batch_size: 32`，一个 epoch 仍然有约 1,688 个 step。每个 step 又包含多次 complex FFT，所以几分钟没有一个 epoch 输出是正常的。

脚本现在支持 batch 级进度打印：

```yaml
experiment:
  print_freq: 10
```

## 权重保存在哪里

每个 run 都保存到：

```text
runs/<run_name>/best.pt
runs/<run_name>/last.pt
```

`best.pt` 是验证集最好的权重。

`last.pt` 是最后一个 epoch 的权重。

注意：第一个 epoch 没跑完之前，不会生成 `best.pt` 和 `last.pt`。

## YAML 字段速查

`experiment.mode`：

```text
eval            只评估
train_scratch   从随机 phase 开始训练
finetune        加载 checkpoint 后微调
prompt_train    后续 prompt/router 训练接口
```

`checkpoints.left_ckpt/right_ckpt`：

```text
加载旧 scripts/train.py 训练出的 single-expert checkpoint
```

`checkpoints.moe_ckpt`：

```text
加载 scripts/run_optical_moe.py 训练出的完整 MoE checkpoint
```

`checkpoints.left_moe_ckpt/right_moe_ckpt`：

```text
从两个已训练 MoE checkpoint 中分别取 left/right expert 并组装
```

`training.freeze_policy`：

```text
frozen              全冻结，只评估
compensation_only   冻结 expert，只训练 residual prompt
first_layer_only    只训练目标 side 的第 1 层
last_layer_only     只训练目标 side 的第 5 层
all_side            训练目标 side 的 5 层 expert phase
all                 训练全部参数
auto                根据 mode 自动选择
```
