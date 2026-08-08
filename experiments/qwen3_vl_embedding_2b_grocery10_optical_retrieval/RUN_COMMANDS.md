# Grocery10 MoE4：训练、四层实物采集、可选逐层微调与最终推理

本文只描述当前推荐的 2×2 / 四专家版本。所有服务器命令均从仓库根目录 `2026OpticsMoE/` 执行，且均为单行命令。

## 0. 当前正式版本

- 结构：Vision expert → Vision global → Language expert → Language global。
- 每个 stack 只有一个 MoE4 expert phase plane 和一个 global phase plane。
- 当前已保存的最佳 MoE4 checkpoint：

    experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/runs/qwen3_vl_embedding_2b_grocery10_moe4_from31_epoch40_replay/ema_last_checkpoint.pt

- 该 checkpoint 的完整复评结果：Top-1 67.69%，Top-3 87.31%，MRR 79.16%。
- 历史 73.46% 是 4×4/MoE16，不属于当前四专家结构，不能交叉加载。

下面用 `CUDA_VISIBLE_DEVICES=3` 举例；按服务器空闲情况修改 GPU 编号。

## 1. 从零复现推荐训练流程

### 1.1 Grocery31、MoE4 预训练 26 epoch

这一步自动准备数据、缓存 31-SKU Teacher embedding、训练、评测和可视化：

    CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/optimization/grocery31_moe4_pretrain.yaml --phase all

固定第 26 epoch EMA checkpoint：

    experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/runs/qwen3_vl_embedding_2b_grocery31_moe4_pretrain/ema_last_checkpoint.pt

### 1.2 缓存目标 Grocery10 的 Teacher embedding

31-SKU 和目标 10-SKU 的 manifest 不同，因此必须单独构建目标缓存：

    CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/optimization/grocery10_moe4_from31_strong_ema.yaml --phase cache_teacher_embeddings

### 1.3 从第 26 epoch 继续目标 10-SKU 微调 14 epoch

最终绝对 epoch 为 40，优化器按目标配置重新初始化：

    CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/optimization/grocery10_moe4_from31_strong_ema.yaml --phase train --resume-checkpoint experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/runs/qwen3_vl_embedding_2b_grocery31_moe4_pretrain/ema_last_checkpoint.pt

新复现实验的固定 epoch-40 EMA：

    experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/runs/qwen3_vl_embedding_2b_grocery10_moe4_from31_epoch40_reproduction/ema_last_checkpoint.pt

### 1.4 对固定 epoch-40 EMA 做完整评测与可视化

    CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/optimization/grocery10_moe4_from31_strong_ema.yaml --phase evaluate --checkpoint experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/runs/qwen3_vl_embedding_2b_grocery10_moe4_from31_epoch40_reproduction/ema_last_checkpoint.pt

    CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/optimization/grocery10_moe4_from31_strong_ema.yaml --phase visualize

## 2. 创建一次完整硬件 session

即使直接使用服务器已有的 epoch-40 checkpoint、不重新训练，也要先执行第 1.2 节一次，确保逐层硬件微调所需的目标 Grocery10 Teacher cache 存在。纯不微调推理不读取该 cache。

硬件配置：

    experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_from31_hardware.yaml

配置中的 `selection.mode: full_dataset` 会固定导出：

- 全部 gallery；
- 全部训练图像；
- 全部测试 query；
- 四个共享 phase BMP；
- 每个样本的第一层振幅 BMP；
- 每一层的理论 CCD 参考。

使用现有已验证最佳 checkpoint：

    CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_from31_hardware.yaml --phase prepare --artifact-profile full

如果使用第 1 节新复现的 checkpoint：

    CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_from31_hardware.yaml --phase prepare --artifact-profile full --checkpoint experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/runs/qwen3_vl_embedding_2b_grocery10_moe4_from31_epoch40_reproduction/ema_last_checkpoint.pt

默认 session：

    experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/hardware_sessions/moe4_from31_epoch40_full_001

只运行一次 `prepare`。当前正式配置禁止覆盖已有 session；若目录已有文件，程序会要求换一个新的 `--output-dir`，不会删除已经上传的 CCD。

固定播放顺序：

    hardware_sessions/moe4_from31_epoch40_full_001/00_manifest/play_order.csv

其中 `role=gallery/train/query` 分别对应登记图库、训练集、测试集。每一层必须严格保持相同文件名和播放顺序。

### 2.1 可选：创建只含 gallery + 全 test 的纯推理 session

如果本次明确不做任何硬件微调，可以不采训练集，使用同一个配置但覆盖 manifest 类型和输出目录：

    CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_from31_hardware.yaml --phase prepare --artifact-profile full --selection-mode test_only --output-dir experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/hardware_sessions/moe4_from31_epoch40_test_only_001

该 session 只能走第 4 节不微调推理。后续四条处理命令均增加：

    --output-dir experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/hardware_sessions/moe4_from31_epoch40_test_only_001

`hardware_finetune` 会拒绝没有 train 样本的 test-only manifest，避免误把测试集自动当训练集。

## 3. 实验室电脑每一层的统一采集方式

对当前层执行以下步骤：

1. 将该层 `amplitude_to_play/*.bmp` 复制到 `experiments/hardware_sdk/data/amplitude_to_play/`。
2. 手动加载 session 中该层 `00_masks/<stage>/*.bmp` 相位图。
3. 清空上一次 `hardware_sdk/data/ccd_captured/`。
4. 运行下面命令。

    python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments\hardware_sdk\configs\tucam_windows.yaml --clear-output

5. 将同名 CCD NPY 上传到 session 对应层的 `ccd_captured/`。

四层目录：

    01_vision_expert
    02_vision_global
    03_language_expert
    04_language_global

不要改 BMP/NPY basename。服务器会逐项核对 manifest；缺一张、多一种同名扩展或尺寸不符都会报错。

## 4. 路线 A：完全不微调，只做四层真实光路推理

以下四条服务器命令彼此不训练参数。每完成一层采集后运行对应命令，它只读取 CCD、执行该层后的电子处理，并生成下一层振幅。

### 4.1 Vision expert CCD → Vision global 振幅

上传到：

    hardware_sessions/moe4_from31_epoch40_full_001/01_vision_expert/ccd_captured

处理：

    CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_from31_hardware.yaml --phase process_vision_expert

下一层振幅：

    hardware_sessions/moe4_from31_epoch40_full_001/02_vision_global/amplitude_to_play

### 4.2 Vision global CCD → Language expert 振幅

    CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_from31_hardware.yaml --phase process_vision_global

下一层振幅：

    hardware_sessions/moe4_from31_epoch40_full_001/03_language_expert/amplitude_to_play

### 4.3 Language expert CCD → Language global 振幅

    CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_from31_hardware.yaml --phase process_language_expert

下一层振幅：

    hardware_sessions/moe4_from31_epoch40_full_001/04_language_global/amplitude_to_play

### 4.4 Language global CCD → 最终检索结果

    CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_from31_hardware.yaml --phase process_language_global

最终结果只使用 manifest 中 `role=query` 的测试图像与 `role=gallery` 的登记图：

    hardware_sessions/moe4_from31_epoch40_full_001/05_retrieval/metrics.json
    hardware_sessions/moe4_from31_epoch40_full_001/05_retrieval/retrieval_results.csv
    hardware_sessions/moe4_from31_epoch40_full_001/05_retrieval/confusion_matrix.csv

## 5. 路线 B：每层实测后微调其下游 100 epoch

每个实测平面及其上游永久冻结，只微调物理上位于该 CCD 后面的光学与电子参数。默认：

- gallery 作为登记图库；
- train 实测图参与适配；
- test 实测图不参与反向传播，只在最后评测；
- `adaptation.include_test_split: false`。

若主动把它改成 `true`，test 也会参与适配，最终结果会明确标记为 transductive/selection-biased，不能再称为独立测试精度。

### 5.1 Vision expert 后适配

采集并上传第一层全量 CCD 后运行：

    CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_finetune --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_from31_hardware.yaml --capture-stage vision_expert

第一阶段最佳 checkpoint：

    hardware_sessions/moe4_from31_epoch40_full_001/06_hardware_finetune/after_01_vision_expert__ccd_vhflip__full_train_v1/checkpoints/best_train_loss.pt

程序会同步更新 `00_masks/02_vision_global` 并重新生成全样本 `02_vision_global/amplitude_to_play`。加载新 mask 后再采第二层。

### 5.2 Vision global 后适配

    CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_finetune --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_from31_hardware.yaml --checkpoint experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/hardware_sessions/moe4_from31_epoch40_full_001/06_hardware_finetune/after_01_vision_expert__ccd_vhflip__full_train_v1/checkpoints/best_train_loss.pt --capture-stage vision_global

第二阶段最佳 checkpoint：

    hardware_sessions/moe4_from31_epoch40_full_001/06_hardware_finetune/after_02_vision_global__ccd_vhflip__full_train_v1/checkpoints/best_train_loss.pt

程序更新 Language expert/global mask，并生成全样本第三层振幅。

### 5.3 Language expert 后适配

    CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_finetune --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_from31_hardware.yaml --checkpoint experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/hardware_sessions/moe4_from31_epoch40_full_001/06_hardware_finetune/after_02_vision_global__ccd_vhflip__full_train_v1/checkpoints/best_train_loss.pt --capture-stage language_expert

第三阶段最佳 checkpoint：

    hardware_sessions/moe4_from31_epoch40_full_001/06_hardware_finetune/after_03_language_expert__ccd_vhflip__full_train_v1/checkpoints/best_train_loss.pt

程序更新 Language global mask，并生成全样本第四层振幅。

### 5.4 Language global 后只适配最终电子读出

    CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_finetune --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_from31_hardware.yaml --checkpoint experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/hardware_sessions/moe4_from31_epoch40_full_001/06_hardware_finetune/after_03_language_expert__ccd_vhflip__full_train_v1/checkpoints/best_train_loss.pt --capture-stage language_global

第四阶段只训练 final detector normalization 和 64D retrieval readout，不再生成光学振幅。最终独立 test 结果仍写入 `05_retrieval/`。

每阶段额外保存：

    06_hardware_finetune/after_*/metrics/adaptation_split.json
    06_hardware_finetune/after_*/metrics/history.csv
    06_hardware_finetune/after_*/metrics/summary.json
    06_hardware_finetune/after_*/trainable_parameters.json
    06_hardware_finetune/after_*/exported_downstream_masks
    06_hardware_finetune/after_*/next_amplitude_bmp

## 6. 训练已完成但导出阶段中断

不要重新训练 100 epoch。对相同 stage、相同输入 checkpoint 加 `--finalize-only`：

    CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_finetune --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_from31_hardware.yaml --capture-stage language_global --finalize-only

若该 stage 的训练是从上一阶段 checkpoint 开始，仍需同时传入与原训练相同的 `--checkpoint`。

## 7. 可选背景扣除

正式采集原图不自动扣背景。若要试验背景扣除，使用共享硬件工程的独立流程；扣除后的 NPY 可以作为该层 `ccd_captured` 输入，但同一目录不要同时保留原始和扣除后的同名不同扩展。

命令见：

    experiments/hardware_sdk/RUN_COMMANDS.md

## 8. 验证

    python -m pytest experiments/hardware_sdk/tests experiments/hardware_sdk/generators/slm_patterns/tests experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/tests -q

## 9. Session 目录约定

```text
moe4_from31_epoch40_full_001/
├── 00_manifest/
│   ├── play_order.csv
│   ├── deployment.json
│   └── sample_metadata/
├── 00_masks/
│   ├── 01_vision_expert/
│   ├── 02_vision_global/
│   ├── 03_language_expert/
│   └── 04_language_global/
├── 01_vision_expert/
├── 02_vision_global/
├── 03_language_expert/
├── 04_language_global/
├── 05_retrieval/
└── 06_hardware_finetune/
```

每个物理层目录统一包含：

```text
amplitude_to_play/      # 下载到实验室振幅 SLM 的 BMP
ccd_captured/           # 上传回服务器的原始或可选扣背景 CCD
registered_ccd/         # 尺寸、翻转和 2×2 binning 的审计记录
simulation_reference/   # 对应样本的理论 CCD/光场
electronic_output/      # 本层 CCD 后电子处理说明与误差统计
```
