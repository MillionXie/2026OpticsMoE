# 环境与运行

本目录是时间一致性任务的独立仿真代码，不包含数据、特征缓存或模型权重。建议使用 Python 3.10 及以上版本；`pyproject.toml` 固定了 NumPy 2.1.3、PyYAML 6.0.2 和 PyTorch 2.6.0。准备原始视频的 Qwen 前端缓存时，另需安装 `preprocessing` 可选依赖。

在本目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python run.py smoke --output runs/smoke/check_01
python -m pytest -q tests
```

Linux 激活命令为 `source .venv/bin/activate`。GPU 训练须安装与机器驱动匹配的 CUDA 版 PyTorch，并先检查 `python -c "import torch; print(torch.cuda.is_available())"`；CPU 冒烟不需要 CUDA。

正式训练还需原始的 2250 train / 558 test manifest、对应的四帧 Vision 缓存、Language 缓存，以及训练 soft-target 文件（默认损失权重为 3）。复制 `src/lgvq_temporal/configs/paths.example.json` 为本地 `paths.local.json`，把五个路径改为本机绝对路径；不要将资产或本地路径文件放进交付包。

```powershell
python run.py preflight --paths paths.local.json --output runs/simulation/preflight_01
python run.py train --paths paths.local.json --device cuda --allow-uncalibrated-noise --output runs/simulation/train_01
python run.py evaluate --paths paths.local.json --device cuda --checkpoint runs/simulation/train_01/best_checkpoint.pt --output runs/simulation/eval_01
```

每次使用新的 `--output` 目录，程序不会覆盖已有结果。默认配置位于 `src/lgvq_temporal/configs/simulation.yaml`：8 μm 训练中插值、30% 相干直流、CCD 扰动训练，三类像素平移关闭。`camera.json` 的噪声参数是未标定的 pilot；`--allow-uncalibrated-noise` 表示明确接受这一点。默认 `evaluate` 不加随机 CCD 噪声；需要带噪仿真时显式指定 `--noise-scale`。

训练沿用原划分并以 test SRCC 选取 checkpoint，因此这里的 test 指标不是独立泛化估计。本机仅以 Python 3.11.9、PyTorch 2.10.0+cpu、NumPy 2.4.4、PyYAML 6.0.3 完成了单元测试和 CPU 冒烟；尚未在这份精简工程中运行完整训练或 GPU 评估。
