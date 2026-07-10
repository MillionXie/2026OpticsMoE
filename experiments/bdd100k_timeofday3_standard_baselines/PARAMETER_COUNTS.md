# BDD100K TimeOfDay-3 Parameter Counts

统计时间：2026-07-10

本文统计 `bdd100k_timeofday3_standard_baselines` 中各 baseline 的参数量，并列出实验组
`qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4_token64_residual` 的可训练参数量。

## 统计口径

- Baseline 参数量由当前代码按对应 config 实例化模型后统计，均为精确值。
- Baseline 当前均未冻结参数，因此 `total parameters = trainable parameters`。
- 标准 D2NN 只统计 phase mask 参数；固定 detector region mask 是 buffer，不作为可训练参数。
- 实验组中 Qwen3-VL-2B backbone 在 student 训练中冻结；本地没有服务器 run 的 `model.json`，因此 backbone 总量采用既有实验总结中的约 `2.13B` 口径。
- 实验组 optical surrogate 和分类 head 参数量由当前代码按配置实例化统计，均为精确值。

## Baseline 参数量

| 模型 | 配置文件 | 输入 | Total params | Trainable params | 备注 |
|---|---|---:|---:|---:|---|
| Standard D2NN-64 | `bdd100k_timeofday3_standard_d2nn64.json` | 64x64 gray | 20,480 | 20,480 | 5 层 phase-only D2NN；无振幅 mask；无卷积/MLP readout |
| LeNet-5 | `bdd100k_timeofday3_lenet5.json` | 32x32 gray | 61,111 | 61,111 | 标准 Conv5-AvgPool-Conv5-AvgPool-FC120-FC84 |
| MobileNetV2 | `bdd100k_timeofday3_mobilenet_v2.json` | 224x224 RGB | 2,227,715 | 2,227,715 | 轻量 CNN baseline |
| ResNet-18 | `bdd100k_timeofday3_resnet18.json` | 224x224 RGB | 11,178,051 | 11,178,051 | 常用 residual CNN baseline |
| VGG11-BN | `bdd100k_timeofday3_vgg11_bn.json` | 224x224 RGB | 128,784,131 | 128,784,131 | 参数量较大的 plain CNN baseline |

## 实验组参数量

实验组：`qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4_token64_residual`

| 模块 | Total params | Trainable params | 备注 |
|---|---:|---:|---|
| Frozen Qwen3-VL-2B backbone | 约 2.13B | 0 | student 训练中冻结；精确值以服务器 `model.json` 的 `backbone.parameters` 为准 |
| Vision optical surrogate | 148,677 | 148,677 | 替换完整 vision transformer stack |
| Language optical surrogate | 280,773 | 280,773 | 替换完整 language transformer stack |
| Classification head | 135,427 | 135,427 | LayerNorm(2048) + bottleneck-64 head |
| Student trainable total | 564,877 | 564,877 | optical surrogate + classification head |

实验组可训练参数拆分：

| 子项 | Trainable params | 占 student trainable |
|---|---:|---:|
| Adapter total | 396,672 | 70.22% |
| Optical phase masks | 32,768 | 5.80% |
| Optical amplitude masks | 0 | 0.00% |
| Detector bias | 8 | 0.00% |
| Residual scale | 2 | 0.00% |
| Classification head | 135,427 | 23.97% |
| 合计 | 564,877 | 100.00% |

## 对比摘要

| 模型 | Trainable params | 相对实验组可训练参数 |
|---|---:|---:|
| Standard D2NN-64 | 20,480 | 0.036x |
| LeNet-5 | 61,111 | 0.108x |
| 实验组 student trainable | 564,877 | 1.000x |
| MobileNetV2 | 2,227,715 | 3.944x |
| ResNet-18 | 11,178,051 | 19.788x |
| VGG11-BN | 128,784,131 | 227.982x |

## 复现统计命令

从仓库根目录运行：

```bash
python - <<'PY'
from pathlib import Path
from experiments.bdd100k_timeofday3_standard_baselines.settings import load_settings
from experiments.bdd100k_timeofday3_standard_baselines.models import build_model, parameter_report

for path in sorted(Path("experiments/bdd100k_timeofday3_standard_baselines/configs").glob("bdd100k_timeofday3_*.json")):
    if "smoke" in path.name:
        continue
    settings = load_settings(path)
    model = build_model(settings)
    report = parameter_report(model)
    print(path.name, settings.model_type, report["parameters"], report["trainable_parameters"])
PY
```

服务器上如果已经跑过实验组，可用下面方式读取精确 frozen backbone 参数量：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("experiments/qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4_token64_residual/runs/qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4_token64_residual/model.json")
report = json.loads(path.read_text())
print("backbone.parameters =", report["backbone"]["parameters"])
print("total_trainable_parameters =", report["total_trainable_parameters"])
PY
```

