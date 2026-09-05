# LightGenV2

`LightGenV2` 是 `2026OpticsMoE` 内的新一代任务目录。旧 `experiments/` 暂时作为
已经验证过的实现来源，不再向其中新增同类试验；新的训练入口、run、报告和交付包
统一从本目录进入。

## 项目地图

| 编号 | 任务 | 当前数据集 | 状态 | 入口 |
|---|---|---|---|---|
| T01 | 物品检索 | Caltech101（可替换） | 待迁移 | `tasks/t01_object_retrieval` |
| T02 | 关键点检测 | LSP（可替换） | 待迁移 | `tasks/t02_keypoint_detection` |
| T03 | 显著性分析 | SALICON（可替换） | 待迁移 | `tasks/t03_saliency` |
| T04 | 语义交互 | OpenMoji（可替换） | 待迁移 | `tasks/t04_semantic_interaction` |
| T05 | 视频分类 | 未确定 | 规划中 | `tasks/t05_video_classification` |
| T06 | 视频质量评价 | LGVQ（可替换） | **当前主任务** | `tasks/t06_video_quality_assessment` |
| T07 | 商品图搜图 | ABO（可替换） | 待迁移 | `tasks/t07_abo_image_retrieval` |
| T08 | 商品图搜文 | ABO（可替换） | 待迁移 | `tasks/t08_abo_image_text_retrieval` |

目录名按任务而非数据集命名，因此以后更换可公开发表的数据集时，不需要重命名工程。

## 现在从哪里开始

```powershell
Set-Location C:\path\to\2026OpticsMoE
conda activate xml

# 检查新工程和 T06 后端
python -m LightGenV2.scripts.check_environment --task t06

# 不读取数据的 CPU 结构冒烟测试
python -m LightGenV2.tasks.t06_video_quality_assessment --phase smoke
```

T06 的训练、评估、硬件六阶段和打包命令全部集中在
[`tasks/t06_video_quality_assessment/README.md`](tasks/t06_video_quality_assessment/README.md)。

## 数据和运行产物放在哪里

- 原始数据仍位于仓库根目录 `data/` 或 `paths.local.yaml` 指定的位置。
- 每个任务自己的划分、manifest 和准备脚本放在该任务 `dataset/`。
- 仿真产物放在该任务 `runs/simulation/`。
- 硬件采集和本地微调放在该任务 `runs/hardware/`。
- 快速测试放在该任务 `runs/smoke/`。
- 论文级表格、图和结论放在该任务 `reports/`。
- 对外交付的实验室 ZIP 放在该任务 `releases/`。

`runs`、checkpoint、CCD 原图和 ZIP 默认不进入 Git。每个 run 内必须保留配置、
命令、环境和结果摘要，以便之后判断是否可以清理。

如果某台机器的数据/缓存路径不同，只复制一次配置：

```powershell
Copy-Item LightGenV2\paths.example.yaml LightGenV2\paths.local.yaml
notepad LightGenV2\paths.local.yaml
```

保留 `null` 的项目继续采用正式 profile 路径；只填写该机器确实不同的路径。

## 共享代码边界

- `common/`：至少被两个任务复用且接口已经稳定的网络/指标代码。
- `hardware_common/`：与具体任务无关的 SLM、CCD、标定和采集能力。
- 任务专属网络始终留在 `tasks/tXX_*/models/`，避免为了优化一个任务而影响全部任务。

详细约束见 [`AI_RULES.md`](AI_RULES.md)。

## Git 同步规则

代码修改完成后必须测试、commit 并 push GitHub；服务器和实验室电脑只通过 Git 拉取
源码。大权重、缓存、CCD 和 ZIP 不进入 Git，继续通过 SHA256 清单传输。禁止以 SCP
直接覆盖源码，也禁止强推 main。完整要求见 `AI_RULES.md` 第 15–20 条。
