# ABO 图搜图：83.125% 仿真与实验适配交付

先看本文件；接手 AI 再读 `AI_HANDOFF.md`。这是固定权重可独立复评的工程，
不是已完成现场标定的硬件控制包。无需原 2026OpticsMoE 仓库，也无需下载完整 Qwen。

## 交付内容与数值口径

- `assets/best.pt`：唯一正式权重，包括冻结 Qwen 前端和全部光电参数。
- `assets/processor/`：离线 tokenizer / 图像 processor。
- `data/`、`protocol.json`：复现所需的 2400 张原图及每张的 SHA256、划分。
- `standalone/`：独立推理、训练、消融代码。含历史训练辅助模块，但本版只用下面的入口；不要默认运行旧 `standalone.cli` 的类别检索协议。
- `reference/`：真实原图复评报告、逐 query 预测、权重/训练审计；不是伪造或从缓存代替重新推理的结果。
- `MANIFEST.json`：源码 commit 与逐文件 SHA256。仅私下科研协作传递；ABO/Qwen 的许可证和图片版权并未因打包而消失，公开再分发须另行核实。

| 同权重测试 | Hit@1 |
|---|---:|
| 正常光电 | 83.125%（665/800） |
| 推理时去掉全部光路、保留电子部分 | 76.375%（611/800） |
| 下降 | 6.75 个百分点 |

4 个融合系数依次为 V expert/global、L expert/global：
`0.447703 / 0.447131 / 0.444921 / 0.445111`。
这些是同尺度融合系数，不是整个模型的光能/性能贡献百分比。

任务是 **200 个已见商品的不同照片检索同一 SKU**，不是判断同一类别：
10 类 × 20 商品，每商品 8 张 TRAIN（同时作为 gallery）、4 张 QUERY，合计 1600/800。
图片来自同一转台序列，不宣称新商品泛化或独立拍摄场景泛化。
无独立 validation；历史 checkpoint/超参数使用 test 选择，因此结果属于 test-selected，不能写成从未参与选模的盲测。
冻结 Qwen 64D 对照为 85.125%，本包不含完整 Qwen baseline 权重。

## 按顺序运行

以下均在解压目录执行。Linux/Windows 的 Python 命令相同。用独立环境，先安装与显卡匹配的 CUDA PyTorch，再安装依赖；不要覆盖已有实验环境。
参考环境：Python 3.11.15、torch 2.6.0+cu124、transformers 4.57.3、numpy 1.26.4、Pillow 12.2.0、RTX4090。
5090 需要支持该卡的更新 CUDA/PyTorch 构建，不要强装上述 4090 构建。

```bash
python -m pip install -r requirements.txt
python delivery.py verify
python delivery.py evaluate --device cuda --output runs/reproduce_83125
python delivery.py export-phase --output runs/export_best_phase
```

复评会分别跑正常/去光推理，最终查看：
`runs/reproduce_83125/final_report.json` 中 `metrics.normal`、`metrics.remove_optical`、`routing`、`model_audit.alpha`。
输出目录必须不存在；不要删除原结果来重跑，换一个明确的目录名。
无 GPU 时可 `--device cpu`，但很慢且浮点精度路径不同。跨 GPU/库版本可能有近似分数排序差异；请报告真实值、设备和差异样本，不要改标签凑数。

相位导出包含 `phases_radians.pt` 和真实权重绘制的 `phase_overview.png`。
PT 有 12 个逻辑相位（每侧 router、4 experts、global）与 6 张完整有效面相位。
它们是 17 μm 逻辑采样，不是可以直接塞到 8 μm 面板的硬件 BMP。

## 继续仿真微调（不是硬件微调）

先复评通过，再选择新输出目录：

```bash
python delivery.py finetune --device cuda --epochs 20 --steps 100 --output runs/finetune_phase_head
```

默认从本版 best 继续，更新相位与原 64D 投影，其他电子参数冻结；采用现有 `sku_phase_head_top1`，
双视图 TRAIN 检索目标、轻增强、EMA、路由均衡，最多每 5 epoch **评估**，不是每 5 epoch 保存一份权重。
只保存 best/last。此命令是继续优化入口，不保证重训必然恢复/超过 83.125%，也不是逐步重演历史搜索。
想训练全部光电参数，可读 `python -m standalone.retrieval_adapt --help`，使用 `sku_capacity_control`，不要擅自更换数据协议。
默认只用一张 GPU（Linux 可先 `export CUDA_VISIBLE_DEVICES=0`）；结束自动退出释放显存，勿终止其他人的进程。

硬件微调需先按下一文档接入实测 CCD，保持数据身份、捕获顺序与强度合同。
本包无 Meadowlark/TUCam 厂商 SDK、设备 LUT 或现场 ROI，不能仅把 `--device` 改成相机就运行。
