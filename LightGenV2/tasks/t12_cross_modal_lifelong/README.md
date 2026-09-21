# T12：四任务跨模态光学终身学习

本任务使用四套 CC BY 4.0 数据，依次学习 SEN12MS 图图、CLEVR 图文、SONYC-UST
音文和 Physical Concepts 视频时序任务。底层信号包括 SAR、光学图像、文本、音频和
视频序列；“四模态”在本任务中专指四种任务类型。数据集不混成一个标签空间，四项任务
分别保留 10、2、2、2 类输出。

## 对比协议

- **Single-task D2NN**：四项任务分别从头训练独立的光学 D2NN，给出每项任务在当前输入
  编码和训练预算下的可学性上限。单任务未达到预设门槛时，不启动正式终身学习比较。
- **Frozen-optics MLP probe**：依次冻结四个单任务 D2NN 的光学相位，只用目标任务全部训练
  数据拟合新的电子 MLP，形成 4×4 的“源光学骨干 × 目标任务”迁移矩阵。它测量光学表征
  的跨任务复用能力，不属于终身学习结果。
- **Sequential D2NN**：按固定顺序学习四项任务，不使用 replay；每学完一个任务，就评估
  所有已学任务，形成标准下三角矩阵，用于显示共享相位的自然遗忘。
- **Sequential D2NN + replay**：与 ours 使用相同任务顺序、电子 MLP、replay 容量、
  当前/旧任务损失权重和验证选模。它用于区分 MoE 专家结构与 replay 本身的贡献。
- **Ours**：固定 16 槽光学 MoE，按 SEN12MS → CLEVR → SONYC → Physical Concepts 顺序学习。每个新任务启用
  四个预分配专家；旧专家和旧任务读出头冻结，router/global phase 通过当前数据和每个
  旧任务最多 512 条 replay 继续更新。当前任务占主阶段损失的 50%，其余 50% 在旧任务
  replay 间平均，避免后期新模态权重随旧任务数量降到 1/3 或 1/4。
- 三种顺序模型共享 1026×1026 分类传播画布、两次角谱传播、两层分类相位、每层 centered-LeakyReLU +
  Softsign OEO、输入功率和训练数据；MoE 另有共享 router phase 与一次路由探测传播。

离线联合训练 D2NN 不再属于活动实验协议。已经完成的旧 run 只作为历史审计证据保留，
不进入新的主表、均值或架构结论。

## 当前模型数据流

```text
224×224 统一振幅场
  ├─ D2NN: 上采样到 986×986 → 全孔径 first phase
  └─ MoE : 224×224 router phase → 路由 CCD → 4/8/12/16 个 224×224 experts
                    ↓
             1026×1026 角谱传播
                    ↓
       centered-LeakyReLU + Softsign OEO
                    ↓
             986×986 global phase
                    ↓
             1026×1026 角谱传播
                    ↓
       centered-LeakyReLU + Softsign OEO
                    ↓
       全 CCD 强度 → 16×16 pooling → 电子 MLP
```

D2NN 有一块 986×986 first phase 和一块 986×986 global phase。MoE 有 16 块
224×224 expert phase、一块 224×224 router phase 和同尺寸 global phase。MoE 路由权重
由各路由窗口光强归一化得到，输入振幅乘 `sqrt(weight)`，所以权重表示分配到专家的功率。

## 电子读出

第二次 OEO 后对完整 CCD 强度图做固定 16×16 自适应平均池化，再用每任务相同结构的
`LayerNorm(256) → Linear(256,64) → GELU → Linear(64,C)` 读出。MoE 与 D2NN 的头完全
相同。该实验属于光电混合系统；报告必须分别列出光学相位参数与电子读出参数，不能称
为纯光学分类。任务专用头解决不同标签空间，任务身份在评估时已知。

## 固定几何 MoE

16 个 224×224 专家从第一项任务开始全部预分配在 4×4 网格上；画布、传播核、global
phase 和 OEO 尺寸在 4→8→12→16 扩展中不变。router CCD 对全部 16 个物理槽位测光，只在
当前任务学习时已经启用的槽位中归一化。新任务可以复用所有旧专家；旧任务 replay 和
评估保持其学习时的容量（分别为前 4/8/12/16 槽），防止后来专家替换冻结的旧光学记忆。
新专家 warmup 时使用新四槽均匀功率，只训练其相位；主训练更新新专家、router、global
phase 和当前任务 MLP。

## 数据和指标

- SEN12MS：2,048 / 512 / 768 个 train/validation/test patch，来自 16 / 4 / 6 个互斥
  场景。输入为 Sentinel-1 VV/VH 与 Sentinel-2 B4/B3/B2/B8。10 类为简化 IGBP 土地覆盖；
  类别语义为 forest、shrubland、savanna、grassland、wetlands、croplands、urban/built-up、
  snow/ice、barren 和 water；实际数值编号以数据包内官方 `single_label_IGBPsimple` 映射为准。
  主指标 macro-F1。当前 validation 缺 3 类、test 缺 4 类，必须扩大或重建场景划分后再设置
  70% 以上目标。
- CLEVR：RGB 三通道按 `[R,G;B,空白]` 保真打包到左半场，问题文本位于右半场，二分类。
  任务是“图中是否存在指定颜色和形状的物体”，标签为否/是；当前为
  6,000 / 750 / 750 个问题，三个划分均正负平衡。
  现有包没有公开 test 文件，本任务按 image id 和 seed 17
  将原 validation 图像无交叠地固定分成新 validation/test；主指标 balanced accuracy。
- SONYC-UST：714 / 259 / 27 段互斥录音，每段构造 8 个事件查询，因此是
  5,712 / 2,072 / 216 个样本。八个事件为 engine、machinery impact、non-machinery
  impact、powered saw、alert signal、music、human voice、dog；每个查询做不存在/存在
  二分类。未知标签不作负例；主指标逐事件 macro-AP。当前测试只有 27 段且两个事件无正例，
  不能以普通 accuracy 作为 70% 门槛。
- Physical Concepts：continuity probe 的 1,024 个固定 quadruplet 子集；每个样本固定抽取
  8 帧灰度图并按时间顺序铺成 2×4 光场，判断 possible/impossible；quadruplet 级互斥
  划分。723 / 132 / 169 个 quadruplet 对应 2,892 / 528 / 676 个平衡样本；主指标
  balanced accuracy。官方数据和材料为 CC BY 4.0。

所有 checkpoint 只按 validation 主指标选择。每个顺序阶段选模完成后才评估 test，生成
`continual_matrix.json`；矩阵第 i 行表示学完第 i 个任务后的状态，第 j 列表示第 j 个任务，
只填写 `j <= i`，不使用 test 选择 checkpoint。当前 SEN12MS 和 SONYC 是固定子集，不能称
全量数据集基准。

## 单任务准入顺序

1. 每个任务先在 64 个分层样本上做 overfit diagnostic。二分类 balanced accuracy 应接近
   1.0，SEN12MS macro-F1 应显著高于随机水平；未通过时先修实现或优化器。
2. 分别训练 Single-task D2NN，保存 train/validation/test。train 高而 validation 低表示
   数据划分或过拟合问题；train 也低表示输入编码或计算图表达不足。
3. CLEVR 优先恢复已经验证过的冻结视觉特征适配器；视频优先加入无监督帧差/运动编码；
   两种适配器在 D2NN 与 MoE 间完全共享并冻结，避免把电子前端差异算成光学优势。
4. SEN12MS 扩大 scene-disjoint 子集并保证各划分有类别覆盖；SONYC 扩大录音测试集合，
   保证各事件存在足够正例。数据合同改变后使用新 manifest 和新 run ID。
5. 只有四个单任务的验证指标均达到预先写入配置的门槛，才运行三个顺序模型。

## 命令

先把 CLEVR 转成统一的 224×224 双模态光场：

```bash
python -m LightGenV2.tasks.t12_cross_modal_lifelong.data \
  --source /path/clevr_attribute_s17_v1 --out /path/t12_clevr_s17
```

下载并准备视频子集：

```bash
python -m LightGenV2.tasks.t12_cross_modal_lifelong.prepare_physical_concepts \
  --cache /path/physical_concepts_raw --out /path/physical_concepts_continuity_s17
```

随后运行结构/梯度验证或训练：

```bash
python -m unittest discover -s LightGenV2/tasks/t12_cross_modal_lifelong/tests -v
python -m LightGenV2.tasks.t12_cross_modal_lifelong \
  --phase smoke --config LightGenV2/tasks/t12_cross_modal_lifelong/configs/initial_s17.json \
  --sen12ms /path/sen12ms_data --clevr /path/t12_clevr_s17 --sonyc /path/sonyc_data \
  --video /path/physical_concepts_continuity_s17 \
  --out LightGenV2/tasks/t12_cross_modal_lifelong/runs/smoke/<run_id>
```

将 `--phase smoke` 改为 `--phase overfit` 执行四任务小样本记忆测试。正式训练时，
`--only single_task` 分别训练四个 D2NN；`--only sequential_d2nn` 运行无 replay 下三角矩阵；
`--only sequential_d2nn_replay` 运行同量 replay 对照；`--only moe` 运行 ours。
每个顺序任务训练 30 epoch，MoE 每个新任务另有 3 epoch 新专家 warmup。
run 保存配置、命令、Git commit、数据 manifest、
逐轮验证、best/last checkpoint、逐样本概率与路由、`continual_matrix.json` 和最终
`comparison.json`。运行目录不覆盖。`--only all` 依次运行单任务 D2NN、两种顺序 D2NN
和 Optical MoE；在单任务门槛未通过前不要使用该选项启动全量实验。

视频预处理额外需要 `protobuf`；完整 Python 依赖见同目录的 `requirements.txt`。
