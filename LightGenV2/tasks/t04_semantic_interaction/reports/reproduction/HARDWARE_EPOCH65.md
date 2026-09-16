# OpenMoji 实测适配：用户指定第65轮

2026-09-17：用户指定采用第65轮记录，不再以自动选出的epoch50作为该展示结果。原始历史及自动best保持不变。

## 结果与口径

六层真实CCD会话为`test1000_02`。固定原模型后，只微调343316个末端电子读出参数；光学相位、router、alpha和前端冻结。四操作各200条用于适配、50条留出，总计800/200，seed20260916。
下面是**200条选模留出集**的历史实测适配指标，不是未微调1000条结果，也不是独立最终测试。第65轮为事后人工指定，不是原scene-exact优先规则的自动best。

```json
{
  "epoch": 65,
  "split": "holdout200",
  "samples": 200,
  "metrics": {
    "changed_cell_accuracy": 0.8575,
    "scene_exact_match": 0.725,
    "edit_grid_iou": 0.8308333333333333,
    "object_f1": 0.944735209235209,
    "cell_accuracy": 0.9894444453716278,
    "foreground_category_accuracy": 0.9506666681170464,
    "preserved_cell_accuracy": 0.9940714311599731,
    "task_accuracy": 1.0
  },
  "checkpoint_status": "epoch65_not_saved_recovery_pending"
}
```

表格建议标注：**实测＋末端适配，修改格准确率85.75%（200条选模留出集，epoch65）**。不能简写为全1000条实测准确率85.75%。

## 权重限制

原训练只保留best/last：原100轮run为epoch50/100，续训run为epoch50/200。第65轮未单独保存PT，目前不能提供与该记录匹配的epoch65权重，也未切换任何部署模型。
如需实际部署该轮，必须按原100轮学习率日程、初始权重、缓存、种子和顺序重放至65轮，保存并复评；不能直接把总epochs改成65（这会改变cosine日程）。重放结果只有通过核验才能称为已恢复，不能预先保证逐位一致。

## 可追溯来源

- 原run：`runs/hardware/head_adaptation_800_200_20260916/epochs.jsonl`，`epoch == 65`。
- 续训run复制保留同一历史：`runs/hardware/head_adaptation_800_200_e200_20260916/epochs.jsonl`。
- [全部逐轮指标](../../runs/hardware/head_adaptation_800_200_e200_20260916/00_逐轮结果.md)，含800条适配、200条留出和全1000条三种口径。
- 原100轮源码`de8084b1`；续训源码`440b1a8f`；实际运行环境师弟RTX4060、PyTorch2.8 CUDA12.6。
- 原模型SHA256：`a69ddcee827749fb9202f9aef11ea45011e433d8b2f0151be2eec3db7dbff9eb`。
- 数据/配置/缓存身份和文件校验记录见上述run的`config.json`、`split.json`、`resume_provenance.json`及`download_manifest.json`。本整理不改写原指标、样本或权重。
