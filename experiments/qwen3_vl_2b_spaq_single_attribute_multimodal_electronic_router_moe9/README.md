# SPAQ 单属性 Qwen3-VL + electronic-router optical MoE9

这是从 `qwen3_vl_2b_spaq_single_attribute_multimodal_homogeneous_moe9` 独立复制并改造的实验。原工程没有被修改。本工程分别训练四个单任务回归模型：

- MOS：总体主观质量；
- Brightness：亮度/曝光质量；
- Colorfulness：色彩丰富度；
- Contrast：对比度。

每个任务都保留完整的 RGB Qwen3-VL 图文输入、chat template、processor、视觉 patch embedding、原生三处 DeepStack 注入、最终 RMSNorm 和 `[2048]` answer hidden。Teacher 是冻结的完整电子 Qwen3-VL-2B 加相同回归头；Student 用 optical MoE 替换视觉 stack，并可配置为冻结电子 language stack 或 optical language MoE。

## 新的真实光路映射

本工程不再使用相位 prompt、透镜相位或光栅进行 fan-out。数据流为：

```text
[T,H] hidden
-> Linear(H,120) + LayerNorm(120) + Softplus
-> [120,120] nonnegative feature
-> electronic router predicts top-k=3 and sparse weights
-> amplitude SLM directly places weighted copies at the selected 3 expert apertures
-> ideal 4f relay
-> co-planar phase SLM (expert phase only)
-> 10 cm propagation
-> square-law CCD + per-expert LayerNorm + nonlinearity
-> multiply the routing weight again and hard-zero unselected experts
-> zero-phase amplitude reload co-planar with the next expert phase SLM
-> repeat for five expert stages
-> stage-5 CCD/reload plane co-planar with global phase SLM
-> global phase
-> 10 cm propagation
-> final CCD
-> 4x4 average pooling to [120,120]
-> per-token-row LayerNorm(120, affine=False) + ReLU
-> valid token rows + Linear(120,H)
```

`amplitude_slm.weight_domain="amplitude"` 表示 SLM 加载振幅为 `w_i × A`。可选 `"power"` 时加载振幅为 `sqrt(w_i) × A`，使分支初始光功率比例为 `w_i`。默认不做输入场逐样本最大值归一化，以免悄悄改变旧实验的数值尺度；部署导出时可以单独做 SLM 灰度标定。

振幅 SLM 与专家/global phase SLM 通过理想 4f relay 共面，所以两者之间没有自由空间传播。相位面到下一 CCD/OEO 面、最后专家到 global 平面、global 到最终 CCD 均为 0.10 m。

## Attention 与 Transformer 残差

默认启用一个 native-shaped Qwen attention prelude，并训练其参数：

```text
A = X + Attention(Norm1(X))
Y = A + OpticalMoE(Norm2(A))
```

残差恒等支路系数固定为 1，与标准 Transformer residual 一致；残差相加后不额外接激活。默认
`initialize_attention_from_teacher=false`，因此只复制 Qwen attention 的结构和必要 norm，attention 投影权重独立随机初始化。设为 `true` 才复制指定 teacher block 的 attention 权重。`native_pre_attention_trainable=true` 控制 attention 是否训练。

## 防止 detector 梯度归零

旧 SPAQ 版本把最终 `[120,120]` readout 当作一整张平面做 LayerNorm，某些 token 行可能全部落到负侧，并被 ReLU 整行清零。本工程默认：

```text
layernorm_scope = "per_token"
normalized_shape = 120
elementwise_affine = false
```

即每个 token 行沿 120 个 detector channel 独立归一化。中间 OEO 仍是每个被选专家对其 `120×120` 探测区域独立归一化，两种归一化的语义不同。

## 数据与缓存

SPAQ 图像保持 RGB。固定 seed=42 按图像做 90/10 train/test split，不设 student validation；每个 epoch 在 test 上评估并按最高 SRCC 保存 best（延续本 SPAQ 系列现有协议）。四个任务使用独立 output directory、teacher cache、processor cache 和 checkpoint，不能跨任务复用。

输出包括 `dataset.json`、`config_resolved.json`、`model.json`、teacher/processor cache、teacher/student checkpoints、逐 epoch metrics、预测 CSV、训练曲线、散点图和相位图。

## 快速检查

```text
python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe9 --help
python -m compileall experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe9
pytest experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe9/tests -q
```
