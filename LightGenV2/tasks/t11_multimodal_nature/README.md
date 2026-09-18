# CC-BY 多模态主实验（SEN12MS / SONYC-UST）

本目录是新的主实验入口，旧的 `t09_multimodal_matching` 结果只作为 pilot，不混入主表。

## 固定物理与公平性契约

- 仿真采样间隔 17 μm；相位器件部署映射为 8 μm。
- 专家输入大小 224×224，专家间隔 30 像素；有效相位面 478×478。
- MoE 与 D2NN 使用相同的双模态输入、输入功率（每模态 0.5，总功率 1）、衍射层数、传播距离、OEO 和读出窗口。
- MoE 使用四个完整 224×224 专家，路由功率为 `q_i`；D2NN 将相同输入按四个区域上采样覆盖整个孔径。D2NN 不接收更小的原始图像，也不使用额外电子分类头。
- 每一层衍射传播后使用同一 OEO；主配置先固定 centered-LeakyReLU + Softsign，其他 OEO 只能作为预注册消融。

## 主任务

1. **SEN12MS 图图**：Sentinel-1 SAR 与 Sentinel-2 多光谱配对的 10 类土地覆盖单标签分类。主指标为 accuracy、balanced accuracy、macro-F1；另保留配对/错配二分类 sanity check。
2. **SONYC-UST 音文**：城市声音与事件文本查询的多标签匹配。缺失标注 `-1` 不当作负例；主指标为 macro-AUPRC，辅以每类 AUROC、固定验证阈值下 macro-F1 和 balanced accuracy。
3. **CLEVR 图文**：保留作为第二个图文任务，采用固定可复现的属性、计数和比较问题集合；同一冻结前端同时服务 MoE 和 D2NN。

## 数据审计

先运行：

```bash
python -m LightGenV2.tasks.t11_multimodal_nature.prepare_manifest \
  --dataset sen12ms --root /path/to/sen12ms --out manifest_sen12ms.json
python -m LightGenV2.tasks.t11_multimodal_nature.prepare_manifest \
  --dataset sonyc_ust --root /path/to/sonyc_ust --out manifest_sonyc_ust.json
```

脚本只记录路径、大小、SHA256、样本计数、划分和许可元数据，不把原始数据或权重提交到 Git。正式训练必须把 manifest SHA256、原始下载 URL、版本/DOI、许可证文本和处理脚本一起写入 run evidence。

## 训练顺序

1. 生成并审计 train/validation/test manifest；SEN12MS 按官方 hold-out scene，训练集内部再按 scene 划 validation，禁止 patch 跨集合。
2. 对两种光学模型运行同一 smoke，检查张量尺寸、总功率、OEO 梯度、读出窗口非重叠和路由统计。
3. 先固定输入前端，再从同一初始化和同一预算分别训练 MoE、D2NN；按 validation 主指标选 checkpoint，test 只评估一次。
4. 保存 best/last checkpoint、完整 history、test prediction、逐样本 route power、读出能量、零读出比例、参数量、命令、GPU UUID、软件版本和 manifest SHA256。

当前状态：任务目录和审计工具已建立；等待在服务器取得官方数据元数据后再启动正式训练。任何未满足许可证、划分或公平性审计的数字只能标记为 smoke/pilot。

## 数据来源

- SEN12MS toolbox：<https://github.com/schmitt-muc/SEN12MS>；数据下载入口见其 README 的 TUM Mediatum 页面，论文和标签/划分文件按原始仓库记录。
- SONYC-UST v2.3：<https://zenodo.org/records/3966543>；使用官方 CSV 标注和 taxonomy，按发布页记录的 CC BY 4.0 条款保存逐文件归属。

## 首轮 smoke 记录（2026-09-18）

在服务器用 GPU3/GPU4、seed=17、12 epochs、phase dropout=0.05 和同一 centered-LeakyReLU OEO 分别运行现有 CLEVR pilot 的 MoE 与 D2NN。MoE 最佳 validation accuracy 49.33%（epoch 1），D2NN 51.07%（epoch 7）；MoE route mean 约为 `[0.514, 0.172, 0.289, 0.025]`，显示第四专家几乎未使用。两者都接近随机水平，因此该结果只用于检查训练器、OEO 梯度和路由统计，不能作为主表或 MoE 优势证据；本次 t09 trainer 没有访问保留 test split，故没有 test accuracy。完整路径、哈希和解释见 [`reports/clevr_leaky_smoke_20260918.json`](reports/clevr_leaky_smoke_20260918.json)。

数据集的任务含义、输入输出和训练安排见 [`DATASET_AND_RUN_PLAN.md`](DATASET_AND_RUN_PLAN.md)。当前不是把三个数据集混成一个任务：SEN12MS 是跨传感器图图土地覆盖分类，CLEVR 是图像与问题匹配，SONYC-UST 是音频与事件文本查询匹配；三者只共享物理光学和公平对照契约。
