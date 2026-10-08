# 从解压到复现：按顺序执行

## 1. 解压并进入目录

建议解压到短路径，例如 `E:\OpenMoji`。进入能看见 `reproduce.py` 的那一层目录；不要在压缩包内运行。下列命令 Windows PowerShell / Linux 均可执行，无需原工程路径。

使用 Python 3.11。可以使用已有、能正常 import torch 的环境，或新建环境：

```powershell
conda create -n openmoji_repro python=3.11 -y
conda activate openmoji_repro
```

先安装与显卡和驱动匹配的 PyTorch；已有正常 CUDA PyTorch 的环境不要被 CPU 版本覆盖。原参考环境为 Python 3.11、torch 2.6.0+cu124。没有 GPU 也能运行：

```powershell
# 仅在新环境、决定使用 CPU 时执行这一行
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
```

然后安装剩余依赖：

```powershell
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

若你的环境没有 torchvision，请安装与已有 torch 匹配的版本。本包依赖文件不主动替换 torch / torchvision。

## 2. 校验完整性（无需 torch）

```powershell
python reproduce.py verify
```

修改交付文件后校验会失败；有意开发时请另存开发副本，并重新制作清单，不要删除校验来掩盖缺失文件。

## 3. 先跑一个可看的例子

```powershell
python reproduce.py demo --device cpu --sample-id test_000008
```

终端最后给出输出目录，里面有 `source.png`、`target.png`、`prediction.png` 和 `demo.json`。例子指令为 “Place a flower below the bicycle.”，应在自行车下方放一朵花；`scene_exact` 应为 true。CUDA 已配置可将 cpu 改为 cuda。

## 4. 复现全部 1000 张测试结果

```powershell
python reproduce.py evaluate --device auto --batch-size 32
```

不需要训练即可运行。`auto` 有 CUDA 就用 CUDA，否则 CPU；显存不足可改 `--batch-size 8`。结果保存到终端所示 `outputs/...`：

- `selected_checkpoint_test_evaluation.json`：整体与分操作指标。
- `same_checkpoint_remove_optical.json`：同一权重去光结果，不重新训练无光模型。
- `test_predictions.jsonl`：逐样本预测。
- `reference_comparison.json`：与原 best 四项核心指标差值，跨设备允许绝对误差 0.005。
- 相位与预测可视化由评估器写入相应子目录。

参考修改格准确率为 0.8715。超出误差容限会报错并保留实际结果；不要修改 reference 让检查通过。

## 5. 可选：重新训练

```powershell
python reproduce.py train --device cuda --epochs 100 --batch-size 32
```

使用随包 train/test 与原协议，重新初始化可训练网络，冻结视觉输入层从随包前端载入；不是从完整 Qwen 下载，也不是默认从 best 接续优化。训练输出单独保存 best / last，然后评估 best。不保存每五 epoch 的历史 PT。若只检查训练接口可使用 `--epochs 1`，但该结果不能当作正式性能。

如要基于 best 二次微调，请接手 AI 在开发副本中以严格 state_dict 加载 best、显式配置新训练方案并记录更改，不要混称为原始 100 epoch 复现。

每次运行自动创建新输出目录，也可加 `--output outputs/my_test`（该目录必须为空）。不必改任何绝对路径、SSH 设置或硬件配置。
