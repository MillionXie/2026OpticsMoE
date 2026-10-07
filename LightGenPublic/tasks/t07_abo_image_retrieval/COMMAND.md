# 从解压到复评/续训

以下命令在**本任务目录**运行，即看得见 `pyproject.toml`、`lightgen_abo/` 的目录。不要切到旧实验包。

## 1. 环境

先激活已有可正常使用 CUDA 的 Python 3.11+ 环境，再安装本包。没有全 Qwen 的权重下载步骤。

```powershell
conda activate xml
python -m pip install -e .
python -m lightgen_abo verify --assets assets
python -m unittest discover -s tests -v
```

如果已具备 `pyproject.toml` 对应依赖且不想变更现有环境，使用 `python -m pip install --no-deps -e .`。不要让 pip/conda 覆盖其他实验环境。首次装环境需要联网下载软件依赖，但推理与续训使用本地 processor/权重，**不访问 Hugging Face 下载模型**。

## 2. 数据只指定根目录，不搬动

数据根下面应包含 `data/abo_similarity10_manifest.csv`，图像相对路径以 manifest 为准。服务器现有位置：

```powershell
$data = '/DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data'
```

上面路径只适用于服务器；Windows 上把 `$data` 改成你已解压数据集的真实路径。如在 Linux bash，写为 `data=/DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data`，后续用 `"$data"`。

## 3. 先复评，不能先改模型

```powershell
python -m lightgen_abo evaluate --assets assets --data "$data" --device cuda --output runs/evaluate_01 --inspect-ccd
```

结果：`final_report.json` 是正常+同权重去光指标；`retrieval_predictions.csv` 是每张图；`retrieval_features.pt` 是可复查64维向量；`phase_masks.png` 可看相位；`ccd_readout_audit.csv` 记录每个测试商品首视图的读出截取比例。图与 CSV 只用于分析，没有反过来修改输入或训练。

## 4. 必要时继续训练

```powershell
python -m lightgen_abo train --assets assets --data "$data" --device cuda --config configs/train.json --output runs/train_01
```

`configs/train.json` 是唯一训练超参入口；本流程要求导入权重的 alpha 下限已经>0.4。训练batch由 `classes_per_batch × products_per_class` 决定，默认40；`--batch-size 4` 是评估显存参数。默认最多30epoch，最后5epoch读出微调。只保留best/last；训练结束读取best重新报告正常和去光结果。

先做一epoch一step的冒烟测试：

```powershell
python -m lightgen_abo train --assets assets --data "$data" --device cuda --epochs 1 --steps 1 --output runs/smoke_01
```

这是接口测试，不是性能复现；总epoch不超过读出收尾轮数时，代码保持联合训练以检查相位梯度。输出目录已存在会拒绝执行，换一个有意义的 run ID，不能覆盖旧结果。

## 5. GPU、路径与交付

Linux 可在命令前加 `CUDA_VISIBLE_DEVICES=1`；Windows PowerShell 用 `$env:CUDA_VISIBLE_DEVICES='0'`。默认只用一张；结束/中断后核对自己的PID从 `nvidia-smi` 消失，不终止其他人的进程。

源码来自 Git 指定 commit。`MANIFEST.json` 与 `assets/manifest.json` 记录源版本和权重SHA；复评报告记录数据SHA。这里不包含硬件播放指令，不要把此仿真审阅包当作已有实验室采集包覆盖过去。
