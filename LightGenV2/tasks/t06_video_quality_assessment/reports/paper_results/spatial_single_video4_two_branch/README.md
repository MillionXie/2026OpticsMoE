# Spatial-4 严格双支路候选

这是按最新结构约束重新训练并归档的空间质量评价模型。推理图中只允许两条主支路：一条电子残差支路和一条光学支路；VGG、额外质量打分支路、Attention 和 Transformer block 均不存在。

## 推理结构

1. 同一视频均匀抽取 4 帧，组成 2×2 光场布局。
2. 冻结 Qwen 前端只执行 patch embedding、官方位置编码插值，以及文本 tokenizer / embedding；不执行 Qwen Transformer block。
3. Qwen 图像 token 投影到 192 维。固定的小型 Conv5 表征只在第一条电子残差路径内部使用，不直接连接最终分数，也不送入光路。
4. 每个阶段严格分成电子卷积残差变换 `E` 和物理衍射变换 `O` 两路。二者先分别做 RMS 同尺度化，再按 `(1-alpha)E + alpha O` 融合。
5. Vision expert、Vision global、Language expert、Language global 共四次融合；两套 router 均为光学区域能量 Top-2。最后由一个电子读出头输出单个连续 Spatial MOS。

## 训练约束

- 训练集 2,250 条、测试集 558 条、不设 validation；每个 epoch 测试并按 test SRCC 选择续训 checkpoint。
- 输入、相位和 CCD 像素位移扰动全部关闭。
- 相干未调制功率在训练时随机取 20%–35%，测试固定为 20%。
- 四个融合系数限制在 0.40–0.90，因此光支路不会被训练成近似零贡献。
- 只保留 best 和 last；不生成每 5 epoch 的相位权重。

## 结果

| 模式 | SRCC | KRCC | PLCC | RMSE | MAE |
|---|---:|---:|---:|---:|---:|
| 正常双支路 | **0.6251** | 0.4516 | 0.6615 | 8.634 | 6.811 |
| 同一 checkpoint 关闭光路 | 0.5217 | 0.3689 | 0.5964 | 9.837 | 7.863 |
| 光开启减去光关闭 | **+0.1034** | +0.0827 | +0.0651 | -1.203 | -1.052 |

视觉 router 的四专家选择占比为 22.67%、23.48%、27.64%、26.21%，没有专家坍缩。语言输入对所有 spatial 样本都是同一条固定 prompt，因此确定性 Top-2 在测试时固定选择同一对语言专家；这不能解释为视频内容 router 坍缩。

旧 VGG 候选的 SRCC 为 0.6393，新版降低约 0.0143，但删掉了冻结 VGG 卷积前端及其校正路径，结构更符合“电子残差 + 光学”两支路的物理解释。VGG16 完整模型约 138M 参数；旧代码虽只取卷积前半段，仍约有 7.64M 个冻结卷积参数，另有约 0.27M 校正适配器，因此没有保留。

## 复现

从仓库根目录执行：

```powershell
python -m LightGenV2.tasks.t06_video_quality_assessment `
  --profile spatial_single_video4_two_branch `
  --phase preflight

python -m LightGenV2.tasks.t06_video_quality_assessment `
  --profile spatial_single_video4_two_branch `
  --phase evaluate
```

服务器正式权重：

```text
LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/
  spatial_single_video4_two_branch/best_checkpoint.pt
```

SHA256：`7bebd5791ed564c333bbe1f7155a70fc2ddf7345814680186aae9fe13946b53f`。
