# LSP 光学最佳版本：73.4786%（2026-09-09）

本次只交付最佳光学模型：`staged_heatmap` 的 epoch50 EMA，非最后一轮权重。
完整兼容源码是为避免 import 缺失；**仅包含一份模型 PT，不含千问baseline或其他候选权重**。
原始训练记录保留在同一run目录。

## 已在师姐服务器准备的目录

包根：`/root/autodl-tmp/lsp_optical_best_7348_20260909/source_tree`

PT：`LightGenV2/tasks/t02_keypoint_detection/runs/simulation/refinement_20260909/staged_heatmap/best_checkpoint.pt`

PT SHA256：`495b9c2c4e3df15d3715f1ce8f2faea7cb9156275b31103ec684f4e96a328518`。
包根的 `data`、`cache` 软链接复用此前 `lsp_lightgen_20260909/source_tree` 的完整资产，
不覆盖旧工程；因此不能单独删除原包的数据/cache。

## 直接评估（不训练）

```bash
cd /root/autodl-tmp/lsp_optical_best_7348_20260909/source_tree
source /root/autodl-tmp/lsp_lightgen_20260909/env/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m LightGenV2.tasks.t02_keypoint_detection.run \
  --profile main_dc20_no_shift_warmstart --phase evaluate \
  --checkpoint LightGenV2/tasks/t02_keypoint_detection/runs/simulation/refinement_20260909/staged_heatmap/best_checkpoint.pt \
  --run-dir LightGenV2/tasks/t02_keypoint_detection/runs/simulation/sister_best7348_eval01
```

结果在新run下 `selected_checkpoint_test_evaluation.json` 和 `metrics/`，含关节预测。
profile 用于建立匹配结构；evaluate 会 strict-load 上述 PT，不会重新 warmstart、重置相位或alpha。
换新实验时换run名，不覆盖原始训练证据。

## 方法及证据

输入224×224，冻结 Qwen 图像嵌入 → 两级192维光电融合 → 14×56×56姿态热图；
原生Vision Transformer及language不执行。光router Top-2/4、224×224专家、478×478场，
532nm/17μm采样/10cm，保持同尺度 `(1-alpha)E+alpha O`。
电子层没有新增；本次训练移除坐标辅助loss，保留原光学扰动（no-shift、DC20等）。

原A100完整1000张test：PCK@0.2=0.7347857143，PCKh@0.5=0.8524285714，NME=0.2297784154。
test参与选模；这是仿真，不是实测CCD结果。换GPU的BF16结果可能有轻微差异。
关闭光支路后PCK=0.7330，说明本候选光贡献有限，不宜将全部性能归于光学。

训练起点及60轮计划见 `reports/reproduction/STAGED_REFINEMENT.md`（任务目录内）。
新包只带best，不带旧初始化PT；如需复跑完整历史训练链，需另取对应源权重，不能把当前best冒充随机初始化。
`PACKAGE_MANIFEST.json` 给出训练commit、交付commit及逐文件SHA256。
