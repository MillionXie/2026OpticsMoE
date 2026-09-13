# ABO 高速相机工程：从这里开始

1. **现在采到哪里**：打开 [实时进度](reports/00_current/03_live.html)。当前使用逐层保持，不自动切相位。
2. **当前日志与操作**：[当前说明](reports/00_current/00_READ_ME.md)。下一层必须等用户指令。
3. **找旧实验/图片**：[分类目录](reports/00_current/04_storage.html)，不用逐个翻 results。
4. **设备和实验命令**：[COMMAND.md](COMMAND.md)、[README.md](README.md)。

## 文件只分这几类

| 目录 | 内容 | 清理约定 |
|---|---|---|
| `sessions/` | 真实CCD、逐样本记录、各层输入；主要在师弟电脑 | 正在跑的绝不移动/删除 |
| `results/` | 实测与诊断数据 | 当前批次原位；未被引用的旧诊断集中到 `90_history/` |
| `reports/00_current/` | 进度、当前结论、文件分类导航 | 日常只看这里 |
| `generated/` | 标定和相位/振幅BMP | 受配置引用保护 |
| `assets/`、`models/`、`vendor/`、`runtime/` | 数据、权重、SDK和依赖 | 不纳入普通清理 |
| `releases/` | 给其他电脑的部署包 | 不是训练结果，不当成最新模型 |

根目录现有Python文件暂不搬动：入口和相互导入依赖这些位置，采集期间不重构。
以后同一天、同一目的只建一个结果批次，其下分参数组；不再散建十几个平级试错目录。
临时预览表覆盖固定文件，只有实际归档操作留下带日期的恢复清单。

整理工具默认只预览：`powershell -File organize_results.ps1`。
指定 `-Apply` 才移动未发现外部引用、超过两小时未写入的旧诊断；不删除数据。
恢复位置记录在 `reports/00_current/04_storage_applied_*.json`，恢复前检查原路径没有同名新实验。
