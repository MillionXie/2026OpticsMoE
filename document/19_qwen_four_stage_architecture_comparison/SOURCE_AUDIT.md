# 代码与结论对应表

本文没有仅根据工程名猜测结构。下表给出每项结论的实际代码来源，便于后续同学复核。

| 结论 | 主要来源 |
|---|---|
| 学生模式替换全部 Vision/Language block、DeepStack 数量与放置 | `experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/optics/replacement.py`，`DeepStackMultimodalReplacement` |
| DeepStack 关闭但主 merger 保留 | 同上；`vision.tap_stages=()` 与 `provider_indexes=(final_index,)` |
| Vision 1024→192、Language 2048→192 | `.../caltech101_electronic_retrieval/electronic_blocks.py`，`ElectronicSequenceCore`；warmstart `modeling.py` |
| Vision 3×3 depthwise Conv2D、Language causal kernel-5 Conv1D | `electronic_blocks.py`，`ElectronicResidualMLPBlock` 与 release YAML |
| 两级电子/光学并行与有界 gate | `.../four_layer_optical_retrieval_10cm_robust/optical_blocks.py`，`VisionTwoBlockOpticalCore`、`LanguageTwoBlockOpticalCore` |
| 192→224→224×224 光学输入 | 同上 `MoE4LanguageTwoBlockOpticalPath._encode_input_fields`；`optics/moe.py::encode_groups` |
| MoE4 2×2/top-2/router 结构 | `optics/geometry.py`、`optics/router.py` 与 `model_moe4.yaml` |
| CCD frame-mean/clip/log1p | `optical_blocks.py::RobustCCDNormalizer` |
| CCD 478→224、LN、ReLU、224→192 | `optical_blocks.py::_readout_delta`；`optics/moe.py::FullPlaneReadout` |
| mean+max 192→384 | `electronic_blocks.py::retrieval_detector_features` |
| 384→64 与 L2 | `caltech101_electronic_retrieval/modeling.py::ElectronicRetrievalReadout` |
| gallery prototype 与 cosine 排序 | `grocery10_optical_retrieval/retrieval_metrics.py::evaluate_embeddings` |
| warmstart 双源合并与分阶段冻结 | `.../warmstart5/modeling.py` |
| 5% gate 语义 | `warmstart5/modeling.py` 与 `optical_blocks.py::_bounded_fusion` |
| 17 μm、10 cm、±16 px、phase dropout 等 | `warmstart5/configs/release/stage1_optical_calibration.yaml` 及其 robust base config |
| 正式 81% 口径 | `warmstart5/FORMAL_RESULT.md` 与封存 evaluation summary |
| 师姐 nonlinear readout/dual adapter | `teams/inference/inference/model.py` |
| 师姐 2400 image/100 title | `teams/inference/inference/data.py` |
| 图像与文本分批、统一编码 | `teams/inference/inference/runtime.py`、`preprocessing.py` |
| 师姐硬件阶段与 modality 过滤 | `teams/inference/inference/hardware.py` |
| 师姐 R@1 声明及 inference-only 边界 | `teams/inference/inference/README.md` |

## 外部基础模型配置核对

Qwen3-VL-Embedding-2B 的实际 config 给出：Vision depth 24、hidden 1024、patch 16、
spatial merge 2、DeepStack indexes 5/11/17，以及 Language depth 28、hidden 2048、
词表 151,936。代码运行时也通过 `settings.resolve_architecture(model)` 读取 hidden/depth，
不依赖本文硬编码。

外部核对来源：

- [Qwen/Qwen3-VL-Embedding-2B 官方 config](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/raw/main/config.json)
- [Hugging Face Transformers 的 Qwen3-VL 官方实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modular_qwen3_vl.py)

## 参数量计算口径

- `340,274,176`：学生路径实际调用的冻结 Qwen 模块的唯一参数量，不是 Qwen checkpoint
  的磁盘总参数量；
- `2,683,709`：warmstart Stage B 中真实 `requires_grad=true` 的 student/readout 参数；
- `3,120,062`：师姐 inference weight 中四级主体、nonlinear readout 与 dual adapter 的
  参数总数；推理包会将它们全部设为不可训练；
- 光学 phase 参数为 `2×(4×224²+478²)=858,376`。
