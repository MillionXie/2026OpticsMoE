# LSP：两级光学 alpha 不低于 0.5（2026-09-10）

本轮从已交付的 `refinement_20260909/staged_heatmap/best_checkpoint.pt`（epoch50，
PCK0.7347857143，SHA256 `495b9c2c4e3df15d3715f1ce8f2faea7cb9156275b31103ec684f4e96a328518`）
续训，不覆盖旧模型。原模型alpha为0.133557/0.072869，去光PCK0.7330。

## 硬约束与初始化

两个融合点独立采用 `alpha = 0.50 + 0.45*sigmoid(logit)`，训练和推理同样生效。
E/O先按有效位置RMS归一化，再 `(1-alpha)E + alpha O`，共同缩放保持输出尺度。
这不是罚项；再大的负logit也不能让alpha低于0.5。初始化均设0.55。

加载旧最佳权重时先检查源SHA、旧架构和router合同，完整加载相位、router、电子支路、姿态头，
然后明确重置两项fusion logit，最后建立EMA。不继承旧EMA平滑器或优化器状态。
新checkpoint架构标识带 `_alpha0.500_0.950`，旧profile加载时会报错，防止悄悄按旧区间解码logit。
每轮和保存前检查alpha有限且在指定区间内。训练前评估是**已将alpha改为0.55的模型**，不是旧73.48%的未修改模型。

不改光路、478场、224专家、10cm、光router Top2、ROI、噪声/no-shift、电子层数或姿态头。
保留最佳方案的heatmap MSE（坐标辅助项为0）及原有正则，不引入新loss。
alpha是融合系数，不是准确率贡献；训练结束仍必须报告去光消融。

## 60轮计划

1. epoch1–10：冻结电子主干/适配/融合logit，alpha固定0.55；训练光学、router、CCD读出和头。
   feature phase LR0.006，router0.001，CCD readout0.0001，pose head0.0002。
2. epoch11–50：联合微调，电子LR0.00001、phase0.003、router0.0003、读出/头0.0001，余弦降至20%。
   alpha在[0.5,0.95]内学习。
3. epoch51–60：固定光学和电子主干，只以0.00002微调CCD读出及头。

完整10428 train / 1000 test，batch24，seed42。epoch1、每5轮、最后一轮测试，最高PCK选EMA；
test明确用于选模。保留满足新alpha限制的epoch0为候选，旧低alpha结果不得参与本轮best竞争。
记录每轮live/EMA alpha、相位变化、分组LR、PCK/PCKh/NME；仅存best/last。

## 实验室命令

从对应Git commit的源码根目录执行，选择空闲GPU；正式输出目录必须未存在。

```bash
ROOT=/DATA/DATA1/guest3/2026OpticsMoE
TASK=$ROOT/LightGenV2/tasks/t02_keypoint_detection
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=3 /home/guest3/miniconda3/envs/xml/bin/python -u \
  -m LightGenV2.tasks.t02_keypoint_detection.refine --profile alpha50 \
  --source "$TASK/runs/simulation/refinement_20260909/staged_heatmap/best_checkpoint.pt" \
  --data-root "$ROOT/data/lsp_pose" --cache-dir /DATA/DATA1/guest3/.cache/huggingface/hub \
  --run-dir "$TASK/runs/simulation/alpha50_staged_seed42_20260910" --batch-size 24
```

验证用 `--smoke --batch-size 4 --workers 0`，输出另选 `runs/smoke/alpha50_20260910`。
不能把smoke当正式成绩。迁移后评估新权重必须使用 `run --profile alpha50 --phase evaluate --checkpoint ...`，
不能使用旧 `main_dc20_no_shift_warmstart`。原始低alpha包保持不变，新结果完成后再决定交付。
最终以新run的 `final_report.json` 为准；目前不预设PCK能保持73.48%，也不宣称光贡献已提升。
