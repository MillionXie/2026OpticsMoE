# Architecture

## Frozen Qwen Vision teacher

Qwen3-VL-Embedding-2B 的全部参数冻结并保持 eval。实验仅调用 `visual`，不调用 token
embedding、language decoder 或生成接口。forward hook 捕获最后一个原生 Vision block
的输出（vision merger 之前），因此二维 token 顺序仍完整保留。

轻量 head 结构为：

```text
token LayerNorm(D)
→ token Linear(D,128)
→ reshape to [B,128,Ht,Wt]
→ Conv/GN/GELU 128→64→32→16
→ bilinear feature upsample to 224×224
→ 1×1 Conv to one raw mask logit
```

mask 自身从未使用 bilinear 插值；只有连续电子 feature 使用 bilinear decoder。

## Vision Optical MoE16 student

物理参数与商品检索基线一致：

| Item | Value |
|---|---:|
| Optical token/channel width | 224 |
| Experts | 16 (4×4) |
| Top-k | 4 |
| Expert phase layers | 1 |
| Expert size | 224×224 |
| Expert pitch | 254 px |
| Active footprint | 986×986 |
| FFT canvas | 1026×1026 |
| Global phase | 986×986 |
| CCD ROI | 986×986 |
| CCD electronic readout | 224×224 |
| Wavelength | 532 nm |
| Pixel pitch | 8 μm |
| Propagation distances | 10 cm |

Student 仍使用现有的 direct amplitude loading、电子 top-4 router、phase-only experts、
OEO、global phase、角谱传播和 CCD readout。第一版关闭 phase dropout 和 k-space
constraint。

最终 CCD 张量的两个轴语义是 `[token_row, optical_feature]`。每张图只读取前
`prod(image_grid_thw)` 行，再按 `image_grid_thw` 恢复二维 feature map。CCD 后 head 的
输入维度为 224。

## Trainability

训练：

- Vision `input_adapter` 和 input LayerNorm；
- electronic router；
- 16 个 expert phase masks；
- global phase mask；
- 配置中启用的 OEO/CCD normalization affine（默认均无 affine）；
- segmentation head。

冻结：

- 所有原生 Qwen 参数；
- patch/position embedding；
- native Vision blocks 与 merger；
-完整 language model；
-未使用的 optical `output_adapter`。

`model_teacher.json` 和 `model_student.json` 保存逐张量可训练参数列表及总量。

## Mask KD

可选 KD 使用：

```text
BCE(student_logits / T, sigmoid(teacher_logits / T)) × T²
```

它只使用最终 teacher mask。缓存为 FP16 memory-mapped `.npy`，identity 包含 teacher
checkpoint SHA256、model id、pixel budget 和 image size；不匹配会拒绝复用。

