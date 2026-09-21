# T12：四任务跨模态光学终身学习

本任务检验固定几何 Optical MoE 是否比普通 D2NN 更适合依次学习不同数据类型。固定顺序为：

1. Kather2016 RGB 组织图像分类；
2. CLEVR RGB 图像与文本属性查询；
3. SONYC-UST 城市音频与事件文本查询；
4. Physical Concepts 视频时序合理性判断。

四套原始数据均要求 CC BY 4.0。SEN12MS 已退出本实验：服务器中随数据附带的许可只允许评估、教育和研究用途，并限制商业使用，不能登记成 CC BY 4.0。历史 registry 中的相反记录属于错误许可判断。

## 模型

统一输入是 224×224 实振幅光场，总输入功率归一化为 1。

```text
D2NN: 224×224 输入 → 上采样覆盖 986×986 first phase
      → 1026×1026 角谱传播 → OEO → 986×986 global phase
      → 角谱传播 → OEO → 全 CCD 强度 → 16×16 pooling → 任务 MLP

MoE : 224×224 router phase → 路由 CCD → soft power routing
      → 固定 4×4 网格中的 16 个 224×224 expert slots
      → 1026×1026 角谱传播 → OEO → 986×986 global phase
      → 角谱传播 → OEO → 全 CCD 强度 → 16×16 pooling → 任务 MLP
```

电子读出头对两个架构完全相同：`LayerNorm(256) → Linear(256,64) → GELU → Linear(64,C)`。这是光电混合分类系统，不称为纯光学分类。

MoE 从第一项任务起就预分配 16 个物理槽位，传播画布、global phase 和 CCD 大小全程不变。每学一项任务新增四个可训练专家，旧专家和旧任务 MLP 冻结；router 和 global phase 通过当前任务及 replay 继续更新。路由窗口光强归一化得到功率权重，进入专家的振幅乘以权重平方根。

## 正式数据合同

正式训练不抽取固定数量样本。数据准备脚本必须把 `all_original_samples: true` 和下列原始身份计数写入 `protocol.json`；`--phase train` 会逐项验证，不满足即拒绝启动。

| 任务 | 原始全集 | 正式划分与样本定义 | 主指标 |
|---|---:|---|---|
| Kather2016 | 5,000 张、8 类，每类 625 张 | 使用全部图像；固定分层 3,496/752/752。公开包无可核查患者/切片 ID，因此只声明图像级划分 | macro-F1、accuracy |
| CLEVR v1.0 | train 70,000 图，val 15,000 图 | 使用全部带 scene graph 的 85,000 图；train 全部训练，官方 val 按 image ID 固定无交叉分成 7,500 validation / 7,500 test；每图构造 3 个正、3 个负 color-shape 查询，即 420,000/45,000/45,000 条。官方 test 没有公开答案/scene graph，不用于派生标签 | balanced accuracy |
| SONYC-UST v2.3 | CSV 中 18,510 个唯一录音：13,538/4,308/664 | 保留官方 train/validate/test；每段录音对 8 个粗事件逐项查询，`-1` 未知标签剔除，不当负例。页面文字写 669 个 test，但实际 CSV 唯一文件为 664，以逐文件清单为准 | macro-AP，逐事件 AP/AUROC |
| Physical Concepts continuity | 5,000 个 quadruplet，每个含 4 段视频，共 20,000 视频 | 使用全部 5,000 组；按 quadruplet 身份固定 hash 70/15/15，四段视频不能跨集合 | balanced accuracy |

Kather 每张图编码为 `[R,G;B,RGB均值]` 四块。CLEVR 把 RGB 编码置于左半场、固定文本编码置于右半场。SONYC 把音频时频图和事件查询编码组合成同一光场。视频固定取 8 个有序帧，拼成 2×4 光场。所有编码在 D2NN 和 MoE 间共享。
Kather 训练时对四个颜色块同步执行随机水平/垂直翻转和 90° 旋转，并使用 0.02 label smoothing；不能分别变换颜色块。相同增强和损失配置用于 D2NN 与 MoE。

64 样本 overfit 只回答“实现能否记住一个极小训练集合”，不估计泛化性能，也不进入论文表格。之前使用 2,048 条 SEN、6,000 条 CLEVR、audio-0 和 1,024 个视频 quadruplet 的运行均标为 subset diagnostic；其中断的 `single_task_d2nn_s17_v1` 不作为结果。

## 对比协议

- **Single-task D2NN**：每项任务独立从头训练，确认单任务在同一光学和电子读出预算下可学。
- **Frozen-optics MLP probe**：冻结每个单任务 D2NN 的光学相位，只用目标任务全部训练数据拟合新 MLP，得到 4×4 跨任务迁移矩阵；它不是终身学习成绩。
- **Sequential D2NN**：按固定顺序训练，无 replay；每阶段评估所有已学任务，形成下三角矩阵和遗忘曲线。
- **Sequential D2NN + replay**：与 ours 使用相同顺序、MLP、replay 容量及选模规则，用于分离 replay 本身的作用。
- **Ours**：固定 16 槽 Optical MoE，按 4→8→12→16 激活专家，旧专家冻结并使用小 replay。

每一阶段只按 validation 主指标选 checkpoint，选完后才评估 test。`continual_matrix.json` 第 i 行表示学完第 i 项任务后的状态，第 j 列表示第 j 项任务，并报告 backward transfer 与 forgetting。

## 当前数据状态（2026-09-21 实机审计）

- Kather2016：服务器已有完整 5,000 张缓存，可立即重新编码。
- CLEVR：当前缓存只有 1,000 train 图和 250 val 图，需要补齐官方 18 GB 包。
- SONYC：标注 CSV 完整，但只有 audio-0 的 1,000 个 wav，需要补齐 audio-1 至 audio-18。
- Physical Concepts：20 个 shard 完整，实测正好 5,000 个 quadruplet。

因此当前没有可接受的四任务正式成绩。数据补齐、manifest 校验和四个单任务准入完成后，才能启动三种顺序模型。

## 命令

准备四套完整数据；CLEVR 的 `--source` 指解压后的 `CLEVR_v1.0`，SONYC 的
`--audio` 指 audio-0 至 audio-18 解压后的共同根目录：

```bash
python -m LightGenV2.tasks.t12_cross_modal_lifelong.prepare_kather2016 \
  --source /path/kather2016 --out /path/t12_kather2016_full
python -m LightGenV2.tasks.t12_cross_modal_lifelong.prepare_clevr_full \
  --source /path/CLEVR_v1.0 --out /path/t12_clevr_full
python -m LightGenV2.tasks.t12_cross_modal_lifelong.prepare_sonyc_full \
  --annotations /path/annotations.csv --audio /path/sonyc_audio \
  --out /path/t12_sonyc_full
python -m LightGenV2.tasks.t12_cross_modal_lifelong.prepare_physical_concepts \
  --cache /path/physical_concepts_raw --out /path/t12_continuity_full
```

正式入口会拒绝子集数据：

```bash
python -m LightGenV2.tasks.t12_cross_modal_lifelong \
  --phase train --only single_task \
  --config LightGenV2/tasks/t12_cross_modal_lifelong/configs/initial_s17.json \
  --kather2016 /path/t12_kather2016_full --clevr /path/t12_clevr_full \
  --sonyc /path/t12_sonyc_full --video /path/t12_continuity_full \
  --out LightGenV2/tasks/t12_cross_modal_lifelong/runs/simulation/<run_id>
```

四套数据尚未同时准备完成时，可以先对已完成的数据执行单任务准入；例如：

```bash
python -m LightGenV2.tasks.t12_cross_modal_lifelong \
  --phase train --only single_task --single-task-name kather2016 \
  --config LightGenV2/tasks/t12_cross_modal_lifelong/configs/initial_s17.json \
  --kather2016 /path/t12_kather2016_full \
  --out LightGenV2/tasks/t12_cross_modal_lifelong/runs/simulation/<run_id>
```

`--phase smoke` 和 `--phase overfit` 可用于小数据工程诊断。正式 run 保存实际配置、命令、Git commit、环境、四份数据 manifest、best/last checkpoint、阶段矩阵、逐样本预测和路由统计，且不覆盖旧 run。
