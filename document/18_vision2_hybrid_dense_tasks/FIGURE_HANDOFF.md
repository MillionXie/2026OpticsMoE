# 初步图片交接说明

## 出图合同

- 画布：183 mm × 54 mm，双栏宽度、总高度在 4–6 cm 内；
- 字体：Arial，基础字号 7 pt，面板号 8 pt；
- 输出：SVG、PDF、600 dpi PNG、600 dpi LZW TIFF；
- 配色：低饱和蓝/青为主结果，红色只用于 error 或风险标记，灰色用于选模线；
- 统计：目前只有单次运行，不画误差条；
- SVG 保持文字节点，不转路径，方便后续修改。

## 数据字段映射

### `salicon_vision2_hybrid`

- panel a：`student_history.csv` 的 `validation_cc`、`validation_sim`、`validation_auc_judd`；
- panel b：`validation_kld`、`validation_mae`；
- 灰色虚线和蓝点：按 `validation_cc` 选中的 epoch 59；
- 图注必须写明 official public validation `n=5,000`，不能写 SALICON test。

### `isic2016_vision2_hybrid`

- panel a：`training_history.csv` 的 `train_mean_iou`、`train_mean_dice`；
- panel b：`test_metrics.json` 的正式 test 指标；
- 灰色虚线：按 training loss 选中的 epoch 96；
- 不得绘制 history 中的 `test_*` 零值，它们表示没有执行逐轮测试；
- 若画箱线图/violin，读取 `test_predictions.csv`，并明确它是 379 个逐样本结果，而 sensitivity/specificity 的整体值采用全像素汇总。

### `lsp_vision2_hybrid`

- panel a：`student_training_history.csv` 的 test PCK/PCKh；
- panel b：test NME 与 normalized router entropy；
- 灰色虚线：training loss 选中的 epoch 148；
- 红色 ×：epoch 149 的 monitored-test 峰值，只是诊断标记；
- 图注必须披露 test 每 epoch 被监控，不能把峰值描述为 validation-selected checkpoint。

## 推荐后续补图

优先级从高到低：

1. 三个任务各 3–5 seeds 后，将主指标改成 mean ± s.d.，同时保留代表性训练轨迹；
2. 与参数量匹配的纯电子 Vision2 baseline；
3. 仿真与硬件逐层替换后的性能下降曲线；
4. 每级光学融合 gate、CCD 强度分布和 phase mask 的联合图；
5. LSP expert selection frequency 与每个 expert 的任务性能，区分“低熵尖锐路由”和“全局专家塌缩”；
6. ISIC 逐样本 IoU 分布与最好/中位/最差样例；
7. SALICON 多目标、中心偏置和边缘目标的失败案例。

## 不要做的事情

- 不要把三种任务的主指标归一化后画成一个“综合性能分数”；
- 不要给单次运行添加虚构标准差；
- 不要把 ISIC 训练 IoU 当成 test IoU；
- 不要把 LSP epoch 149 观察峰值写成独立验证集选模结果；
- 不要读取 live `runs/` 后直接覆盖 evidence；新实验应新建带日期/variant 的快照目录并更新 manifest；
- 不要使用 JPEG 保存线图或 mask，避免压缩伪影。

## 复现和编辑流程

1. 在仓库根目录运行 `python document/18_vision2_hybrid_dense_tasks/scripts/build_report.py`；
2. 检查脚本无完整性报错；
3. 优先打开 SVG 编辑排版；
4. 修改图注时同步检查 `METRIC_DEFINITIONS.md` 和 `evidence/summary.json`；
5. 若替换证据文件，先更新 `SOURCE_MANIFEST.csv` 并保留旧快照，不直接覆盖本次结果。
