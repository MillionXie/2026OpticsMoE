# A100 formal 原始数据归档

本目录按红框表格的任务顺序整理了 A100 测量证据。每个任务目录都包含 `Ours/` 与 `Baseline/`；文件只来自本地审计包中已下载的服务器原始记录、历史 Git 证据或当前 A100 首层到读出头测量，不生成新的数值。

## 目录顺序

1. `01_LGVQ_temporal`
2. `02_LGVQ_spatial`
3. `03_ABO_image_to_text`
4. `04_ABO_image_to_image`
5. `05_LSP`
6. `06_SALICON`
7. `07_OpenMoji`

## 文件口径

- `Ours/` 中的 `report.json`、`timing_*.csv`、`component_*.csv` 和 `power_samples.csv` 是从 `ours_narrow_clean_final` 的完整原始记录按任务筛选出的派生副本；没有改变数值。每个目录的 `source_manifest.json` 记录了完整源文件和 SHA-256。
- `Baseline/` 中的 `redbox_source_report.json` 是与表格红框数值对应的历史 A100 证据；`current_firstblock_formal/`（若存在）是后来统一“首个原生 Vision Transformer block 到最终读出”的同步墙时间复测，二者不能混为同一时间口径。
- LGVQ 的 `Baseline/server_evaluation/` 还保留了服务器正式目录中的逐视频预测/计时、功率采样和报告原文件。时间质量的 1200.053 ms 是 batch-1 单视频时间乘以 16 的表格口径，不是原生 batch-16 的一次测量。
- Ours 的物理光路固定项为每次物理 pass `0.714 + 0.300 + 0.0307 = 1.0447 ms`；能源报告采用光学平台功率与 A100 组件采样的混合代理，详见各 `report.json` 的 `measurement_policy`。

## 审查结论

根目录的 `redbox_values.csv` 是表格数值的逐行审计；`verification` 列为 `PASS` 或 `PASS_WITH_SCOPE_WARNING`。如果某任务只有历史汇总而没有逐样本原始文件，目录中的 `MISSING_RAW_DATA.md` 会明确说明，不能将汇总值当成逐样本原始记录。

完整源文件仍保留在上一级 `evidence/`，本目录只是有来源清单的组织副本。整理完成后请以 `SHA256SUMS.txt` 校验文件完整性。

说明：上一级审计包的 `evidence/server_2026OpticsMoE_a100_formal/temporal/per_video_predictions_and_timing.csv` 当时被 OneDrive 占用，导致上一级总清单将它列在 `SHA256SUMS_SKIPPED.txt`；本目录 `01_LGVQ_temporal/Baseline/server_evaluation/per_video_predictions_and_timing.csv` 是同一服务器文件的可校验副本，已包含在本目录的完整 `SHA256SUMS.txt` 中。
