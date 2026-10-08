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
  "checkpoint_status": "replayed_epoch65_reloaded_metrics_verified"
}
```

表格建议标注：**实测＋末端适配，修改格准确率85.75%（200条选模留出集，epoch65）**。不能简写为全1000条实测准确率85.75%。

## 已恢复的权重与加载方法

已按原100轮cosine日程从原始模型重放到65轮（不是从best接着训练），恢复成功。原800/200划分、缓存、种子、顺序和训练范围均校验不变；6项GPU环境测试通过。重载PT后全部1000条、三个分区及四操作的全部汇总指标与历史第65轮误差为0；另起进程复评也通过，训练退出码0。

- 本地选定PT：[last_checkpoint.pt](../../runs/hardware/head_replay_epoch65_20260917/last_checkpoint.pt)，内部`epoch == 65`。**不是同目录best_checkpoint.pt（epoch50）**。
- 师弟电脑：`E:\code\guest\2026OpticsMoE\OpenMoji_Lab_SHS_8um\runs\head_replay_epoch65_20260917\last_checkpoint.pt`。
- SHA256：`8d7c2f6788a9ac67f79f28a3b9065ca633a7d31f7beddde313fdeb87e5c96557`。
- [核验报告](../../runs/hardware/head_replay_epoch65_20260917/replay_verification.json)与[selected_predictions.json](../../runs/hardware/head_replay_epoch65_20260917/selected_predictions.json)包含真实重新评估结果。下载文件均经SHA256核验。
- 重放源码`7fdd931e`；逐轮loss最大差异`2.672672271705756e-06`，中间轮次分组指标最大差异0.005，但最终第65轮指标全部一致。原第65轮PT未保存，不能证明权重逐位相同；应称“原日程重放、指标复现通过的第65轮权重”。

PT只包含末端shared_readout及优化器等元数据，不是独立完整网络。先用原工程加载固定原模型，再加载此PT：

```python
payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
assert payload["epoch"] == 65
assert payload["base_checkpoint_sha256"] == "a69ddcee827749fb9202f9aef11ea45011e433d8b2f0151be2eec3db7dbff9eb"
model.shared_readout.load_state_dict(payload["shared_readout"], strict=True)
model.eval()
```

只能用于与本次实测前端/光学合同一致的工程。没有覆盖原模型、CCD或历史best，也未自动切换硬件程序的部署配置。

## 可追溯来源

- 原run：`runs/hardware/head_adaptation_800_200_20260916/epochs.jsonl`，`epoch == 65`。
- 续训run复制保留同一历史：`runs/hardware/head_adaptation_800_200_e200_20260916/epochs.jsonl`。
- [全部逐轮指标](../../runs/hardware/head_adaptation_800_200_e200_20260916/00_逐轮结果.md)，含800条适配、200条留出和全1000条三种口径。
- 原100轮源码`de8084b1`；续训源码`440b1a8f`；实际运行环境师弟RTX4060、PyTorch2.8 CUDA12.6。
- 原模型SHA256：`a69ddcee827749fb9202f9aef11ea45011e433d8b2f0151be2eec3db7dbff9eb`。
- 数据/配置/缓存身份和文件校验记录见上述run的`config.json`、`split.json`、`resume_provenance.json`及`download_manifest.json`。本整理不改写原指标、样本或权重。
