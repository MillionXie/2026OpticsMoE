# T12：四模态跨任务光学终身学习

本任务使用三套 CC BY 4.0 数据，覆盖四类输入模态：SEN12MS 的 SAR/多光谱，CLEVR 的
RGB/文本，以及 SONYC-UST 的音频/文本。共享模态按物理含义合并后为 SAR、光学图像、
文本、音频四类。数据集不是混成一个标签空间；三项任务分别保留 10、2、2 类输出。

## 对比协议

- **D2NN baseline**：单个全孔径首层相位和共享 global phase，离线交错训练三项任务，
  因而在训练开始时能访问全部四种模态。
- **Ours**：固定 12 槽光学 MoE，按 SEN12MS → CLEVR → SONYC 顺序学习。每个新任务启用
  四个预分配专家；旧专家和旧任务读出头冻结，router/global phase 通过当前数据和每个
  旧任务最多 256 条 replay 继续更新。
- 两者共享 772×1026 传播画布、两次角谱传播、两层相位、每层 centered-LeakyReLU +
  Softsign OEO、输入功率和训练数据。

## 电子读出

第二次 OEO 后对完整 CCD 强度图做固定 16×16 自适应平均池化，再用每任务相同结构的
`LayerNorm(256) → Linear(256,64) → GELU → Linear(64,C)` 读出。MoE 与 D2NN 的头完全
相同。该实验属于光电混合系统；报告必须分别列出光学相位参数与电子读出参数，不能称
为纯光学分类。任务专用头解决不同标签空间，任务身份在评估时已知。

## 固定几何 MoE

12 个 224×224 专家从第一项任务开始全部预分配在 3×4 网格上；画布、传播核、global
phase 和 OEO 尺寸在 4→8→12 扩展中不变。router CCD 对全部 12 个物理槽位测光，只在
当前已启用槽位中归一化。新专家 warmup 时使用新四槽均匀功率，只训练其相位；主训练
更新新专家、router、global phase 和当前任务 MLP。

## 数据和指标

- SEN12MS：春季 scene-disjoint 子集，SAR+S2，10 类；主指标 macro-F1。
- CLEVR：RGB 三通道按 `[R,G;B,空白]` 保真打包到左半场，问题文本位于右半场，二分类。
  现有包没有公开 test 文件，本任务按 image id 和 seed 17
  将原 validation 图像无交叠地固定分成新 validation/test；主指标 balanced accuracy。
- SONYC-UST：audio-0 子集，音频+事件查询，未知标签不作负例；主指标逐事件 macro-AP。

联合 D2NN 和最终 MoE 都只按三项 validation 主指标的平均值选 checkpoint。test 仅在
选模完成后评估。当前 SEN12MS 和 SONYC 是固定子集，不能称全量数据集基准。

## 命令

先把 CLEVR 转成统一的 224×224 双模态光场：

```bash
python -m LightGenV2.tasks.t12_cross_modal_lifelong.data \
  --source /path/clevr_attribute_s17_v1 --out /path/t12_clevr_s17
```

随后运行结构/梯度验证或训练：

```bash
python -m unittest discover -s LightGenV2/tasks/t12_cross_modal_lifelong/tests -v
python -m LightGenV2.tasks.t12_cross_modal_lifelong \
  --phase smoke --config LightGenV2/tasks/t12_cross_modal_lifelong/configs/initial_s17.json \
  --sen12ms /path/sen12ms_data --clevr /path/t12_clevr_s17 --sonyc /path/sonyc_data \
  --out LightGenV2/tasks/t12_cross_modal_lifelong/runs/smoke/<run_id>
```

去掉 `--phase smoke` 运行初始训练。run 保存配置、命令、Git commit、数据 manifest、
逐轮验证、best/last checkpoint、逐样本概率与路由、最终 comparison.json。运行目录不覆盖。
