# 六任务 baseline 代码交接

按论文表格顺序导出对应历史版本，每个任务为独立源码目录及 ZIP：

| 顺序 | 目录 | 指标 | 源码版本 |
|---|---|---|---|
| 01 | `01_lgvq` | Temporal/Spatial SRCC 0.7663/0.6908 | `4b24e9ad` |
| 02 | `02_abo_image_text` | R@1 0.7358 | `b9a4be68` |
| 03 | `03_abo_image_image` | R@1 0.8513 | `5f731fae` |
| 04 | `04_lsp` | PCK@0.2 0.7226 | `bcec3a59`（权重复评版本） |
| 05 | `05_salicon` | CC 0.8748 | `d8653fa6` |
| 06 | `06_openmoji` | 修改格准确率 0.8420 | `966f4070` |

## 生成代码包

在包含上述历史提交的仓库中运行：

```bash
python LightGenV2/scripts/build_baseline_handoff.py --output LightGenV2/reports/20260915_baseline_methods/code_packages
```

输出目录必须尚不存在。已有导出保留；再次导出使用另一个明确命名的路径。构建器从 Git 对象读取源码，不依赖工作区当前版本。普通交接只需发送生成的 ZIP，接收者无需原仓库和 Git 历史。

每个包包含 `source/` 原始代码及依赖配置、`run_baseline.py` 入口、`README.md` 运行说明、`METHODS.md` 技术说明、`requirements.txt`、`REFERENCE.json` 指标与权重身份、`SOURCE_MANIFEST.json` 源码哈希。总目录附六个 ZIP 的 `SHA256SUMS.txt`。

数据、预训练 Qwen、任务头权重和特征缓存单独提供；具体文件要求见每个包的 README。需要任务头的 LGVQ、LSP、SALICON 和 OpenMoji 均区分重新训练与固定权重复评。保留必要的历史公共模块，以维持原模型计算和配置继承；入口默认只运行对应 Qwen baseline。

本次验证涵盖源码逐字节一致性、Python 语法、静态依赖收集、打包完整性和校验器行为；未重新训练，也未重新计算完整测试集指标。
