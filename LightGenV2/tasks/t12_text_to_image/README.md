# T12 单物体文生图

T12 是 LightGenV2 的单次前向、文本条件图像生成任务。任务只生成少数类别的单个居中
物体，不做开放域、多物体、计数、空间关系、扩散去噪或自回归生成。

当前状态：模型、数据合同、Qwen/VAE 缓存、训练入口、配对 baseline、可视化比较与 CPU
结构测试已经建立；uCO3D 正式子集、正式训练结果和硬件结果尚未产生。

## 冻结协议

- 数据：uCO3D 的六个单物体类别小子集，全部记录必须明确为 CC BY 4.0。
- 候选类别：杯子、瓶子、鞋、背包、水果、玩具车；先按正式类别名和可用实例数审计后冻结。
- 每类 200 个不同物体实例，每个实例抽 10 帧；train/val/test 按实例切分为
  160/20/20，不允许同一物体的不同视角跨 split。
- 规模：训练 9,600，验证 1,200，测试 1,200，共 12,000 张。
- 图像：单个 mask 前景，12% 裁剪边距，物体不超过画布 74%，确定性中性背景，224×224。
- 文本：保留 uCO3D 的短描述，内容限制为类别、颜色、材质和外观；不要求数量及空间推理。
- 文本前端：完整冻结 `Qwen/Qwen3-VL-2B-Instruct`，缓存最终 hidden state 的 masked mean。
- 图像 codec：完整冻结 `stabilityai/sd-vae-ft-mse`，目标和输出均为 `4×28×28` latent。
- 推理：一次生成主干前向和一次 VAE decode；没有循环。

## LightGen 结构

输入为一个冻结 Qwen 文本向量和一个 256 维 style code。二者投影后注入 14×14、192 维
空间 token。正式 profile 直接复用 T01 已审计的 DC20 `BalancedVisionCore`：

```text
stage-1 input ─┬─ electronic residual-1 ─┐
               └─ optical Router+Top2 experts ─┤ detached-RMS fusion
                                                  ↓
stage-2 input ─┬─ electronic residual-2 ─┐
               └─ optical global block ─────────┤ detached-RMS fusion
                                                  ↓
                          residual latent head: 14×14 → 4×28×28
                                                  ↓
                                  frozen VAE decoder → 224×224 RGB
```

每个 stage 的电子和光学分支严格读取同一个输入；不存在 `electronic → optical` 串行关系。
第一阶段 Router、Top-2 experts 和第二阶段 global block 均沿用现有 224 SLM、478 active CCD、
518 numerical canvas、17 μm、10 cm 和尺度匹配融合合同。

训练时额外使用一个轻量 posterior encoder，从真实 VAE latent 得到 `mean/logvar`；推理时
删除该 encoder，直接采样 `N(0,I)`。第一版损失为 latent L1、latent MSE 和 warm-up KL，
不使用 GAN。若正式结果明显模糊，再新增独立 profile 引入 PatchGAN，不能覆盖本 baseline。

`compact_fft` 仅供 CPU smoke 和结构调试，不能作为论文性能或硬件结果。

## Qwen + VAE baseline

Baseline 与 LightGen 共用：

- 完全相同的 train/val/test；
- 同一份冻结 Qwen text cache；
- 同一份冻结 VAE latent cache 和 VAE decoder；
- 相同 posterior、style 维度、latent head、损失、训练轮数和采样 seed。

唯一的主干差异是 baseline 用两层条件电子残差替代整个光学支路。正式参数量当前约为：
LightGen 3.65M、baseline 3.09M，不把冻结 Qwen/VAE 参数计为可训练参数。

训练结束后使用 `compare.py` 在相同六条 prompt、相同 seed 下生成配对网格。正式质量报告
还必须在固定 test 上给出 FID/KID、CLIPScore、类别正确率和 LPIPS diversity；当前代码生成
网格不等同于已有质量结果。

## 数据准备

先使用 uCO3D 官方 downloader 只下载选定类别的 RGB、mask 和 metadata，再导出 CSV：

```text
sequence_id,category,caption,frame_path,mask_path,source_url,license
```

随后运行：

```powershell
python -m LightGenV2.tasks.t12_text_to_image.prepare_uco3d `
  --index-csv D:\uco3d\selected_frames.csv `
  --output-dir LightGenV2\tasks\t12_text_to_image\dataset\uco3d_single_object_v1 `
  --categories "mug,bottle,shoe,backpack,banana,toy_car"
```

类别名只是候选，必须以实际 uCO3D taxonomy 和样图审计结果为准。准备脚本会拒绝非
CC BY 4.0 行、空 mask、类别实例不足及跨 split 身份泄漏。

## 运行

安装生成任务的额外依赖：

```powershell
pip install -r LightGenV2/requirements/generation.txt
```

先跑结构测试和 smoke：

```powershell
python -m pytest LightGenV2/tasks/t12_text_to_image/tests -q
python -m LightGenV2.tasks.t12_text_to_image --profile smoke --phase smoke --device cpu
```

数据就绪后只缓存一次冻结特征，两条路线读取相同缓存：

```powershell
python -m LightGenV2.tasks.t12_text_to_image --profile lightgen --phase cache --device cuda
python -m LightGenV2.tasks.t12_text_to_image --profile lightgen --phase train --device cuda
python -m LightGenV2.tasks.t12_text_to_image --profile baseline --phase train --device cuda
python -m LightGenV2.tasks.t12_text_to_image --profile lightgen --phase evaluate --device cuda
python -m LightGenV2.tasks.t12_text_to_image --profile baseline --phase evaluate --device cuda
python -m LightGenV2.tasks.t12_text_to_image.compare `
  --lightgen-checkpoint LightGenV2\tasks\t12_text_to_image\runs\simulation\lightgen_parallel_seed42\best_checkpoint.pt `
  --baseline-checkpoint LightGenV2\tasks\t12_text_to_image\runs\simulation\qwen_vae_baseline_seed42\best_checkpoint.pt `
  --output-dir LightGenV2\tasks\t12_text_to_image\runs\simulation\matched_comparison
```

正式 run 只保留 `best_checkpoint.pt` 和 `last_checkpoint.pt`。验证集只负责选择 checkpoint；
测试集只能在协议和超参数冻结后评估。

## 未完成项

1. 下载候选 uCO3D 类别并生成 contact sheet，冻结最终六类。
2. 安装/缓存冻结 VAE；本机已有 Qwen3-VL-2B-Instruct 缓存，但未发现该 VAE 缓存。
3. 生成三份共享 feature cache，分别训练 LightGen 和 baseline。
4. 补固定测试集的 FID/KID、CLIPScore、类别准确率和多样性评估。
5. 仿真候选稳定后再建立 DC20 硬件 profile；不得把 `compact_fft` 数值写成硬件结果。
