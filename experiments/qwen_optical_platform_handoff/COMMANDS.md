# 交接工程命令顺序

以下命令均从解压后的工程根目录执行。先阅读对应的 simulation/hardware 文档，不能
只复制某一条正式训练命令。

当前 Caltech101 实验仍按原命令继续：

```powershell
Set-Location E:\code\guest\qwen_mnist4_early_robust_full_data_lab
conda activate xml
python -m experiments.lab_qwen.local_four_stage `
  --profile accuracy_first_full `
  --stage vision_expert `
  --epochs 100
```

这里的 `accuracy_first_full` 是当前 Caltech101 实例的 profile，新任务不得直接沿用这个名字。

## 1. 两个工程都先做

安装与 GPU 匹配的 PyTorch 后：

```powershell
pip install -r requirements.txt
python -m experiments.qwen_optical_platform_handoff `
  experiments\qwen_optical_platform_handoff\templates\simulation_retrieval.contract.json
```

```powershell
python -m pytest experiments\qwen_optical_platform_handoff\tests -q
```

## 2. 仿真工程参考 smoke

商品检索：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16 --config experiments/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16/configs/abo_smoke.yaml --phase all
```

质量评价：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224 --config experiments/qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224/configs/kadid10k_mos_smoke.json --phase all
```

正式运行前必须新建实验目录、冻结 split、完成 task contract 和 smoke；不要直接把
上述参考配置改成正式结果目录。

## 3. 新实验电脑的硬件准备

```powershell
Copy-Item experiments\lab_qwen\LAB_CONFIG.template.yaml experiments\lab_qwen\LAB_CONFIG.yaml
notepad experiments\lab_qwen\LAB_CONFIG.yaml
```

必须填写 LUT、曝光、相位中心和四个逻辑角点后：

```powershell
python -m experiments.lab_qwen.prepare_lab
```

设备 smoke、LUT、ROI/曝光和形状一致性的详细命令在：

```text
experiments\lab_qwen\COMMANDS.md
experiments\hardware_sdk\LAB_WINDOWS_QUICKSTART.md
experiments\qwen_optical_platform_handoff\HARDWARE_PROJECT.md
```

先用 MNIST4 检查方向和四个探测区；再采 Qwen 任务。MNIST 与 Qwen 使用同一套
homography、曝光和 LUT，但任务特定 phase mask 与 ROI readout 不能混用。

## 4. 当前四阶段参考闭环

对一个已经适配了 hardware bridge 的新任务，每层执行：

```text
仿真工程导出本层 payload
→ 手动确认 phase SHA
→ acquire_folder 采本层全部 manifest 样本
→ 校验 CCD 数量/SHA/方向
→ development 选模的本地微调
→ 从最佳 checkpoint 导出下一层
```

Caltech101 参考命令为：

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\generated\formal_hardware.yaml `
  --stage-dir experiments\lab_qwen\four_accuracy_first_full\01_vision_expert `
  --clear-output

python -m experiments.lab_qwen.local_four_stage `
  --profile accuracy_first_full `
  --stage vision_expert `
  --epochs 100
```

新任务必须新增自己的 profile、dataset keys、head/loss/metric 和 session 目录；不能
沿用 `accuracy_first_full` 名称假装已经适配。
