# LightGenV2

**不知道该打开哪个工程时：**回到仓库根目录的
[唯一工程入口](../START_HERE.md)。外部旧副本、实际运行目录和数据资产的区别在那里列明。

`LightGenV2` 是 `2026OpticsMoE` 内的新一代任务目录。旧 `experiments/` 暂时作为
已经验证过的实现来源，不再向其中新增同类试验；新的训练入口、run、报告和交付包
统一从本目录进入。

## 项目地图

| 编号 | 任务 | 当前数据集 | 状态 | 入口 |
|---|---|---|---|---|
| T01 | 物品检索 | Caltech101（可替换） | DC20 正式复跑完成 | `tasks/t01_object_retrieval` |
| T02 | 关键点检测 | LSP（可替换） | DC20 正式复跑完成 | `tasks/t02_keypoint_detection` |
| T03 | 显著性分析 | SALICON（可替换） | 核心源码已归主线；历史资产及部署边界见任务说明 | `tasks/t03_saliency` |
| T04 | OpenMoji robust 消融 | OpenMoji（可替换） | 仿真及硬件源码分版本登记；现用实验目录保护 | `tasks/t04_openmoji_robust_ablation` |
| T05 | 视频分类 | 未确定 | 规划中 | `tasks/t05_video_classification` |
| T06 | 视频质量评价 | LGVQ（可替换） | Spatial／Temporal 核心及适配入口已归主线 | `tasks/t06_video_quality_assessment` |
| T07 | 商品图搜图 | ABO（可替换） | rank72 最终模型封存；训练及回放源码已归主线 | `tasks/t07_abo_image_retrieval` |
| T08 | 商品图搜文、文搜图 | ABO easy100 | 双向模型与 baseline 分别保留 | `tasks/t08_abo_image_text_retrieval` |
| T09 | 图文／音文匹配 | CLEVR／SpeechCommands | 核心运行源码已归主线；资产闭包仍有待办 | `tasks/t09_multimodal_matching` |
| T10 | 专家数扩展 | 各任务固定预算实验 | 修正后源码及结果复用身份已登记 | `tasks/t10_expert_scaling` |
| T11 | 病理终身学习 | 原病理四任务 | 最终源码及必要对照已归主线 | `tasks/t11_lifelong_optics` |
| T12 | 图文编辑／文生图 | 任务正式划分 | 最终、历史版本及 baseline 分别保留 | `tasks/t12_text_to_image` |
| T13 | 时序鲁棒训练 | 任务正式划分 | 核心源码及教师依赖已归主线 | `tasks/t13_temporal_robust_training` |
| T16 | 多模态终身学习 | 遥感／图文／语音／物理 | 最终版及必要对照已归主线 | `tasks/t16_zero_phase_ccd_lifelong` |

目录名按任务而非数据集命名，因此以后更换可公开发表的数据集时，不需要重命名工程。

本表是工程导航，不是完整迁移验收表。旧 `t04_semantic_interaction` 保留为兼容后端及
应用开发线（布局/物体大小/电子容量及对外复现）的入口，不能只当作robust旧试错清理；
其历代版本和最终身份边界见该任务README。新的 robust 消融从 T04 robust 入口进入，
两线不能互套指标。精确源码、PT、数据身份及剩余例外
见 [`TASK_REGISTRY.json`](TASK_REGISTRY.json)；各版本指标以任务 README／原报告为准。
本机、训练服务器和实验室主目录均已实际建立 `main` 源码入口；实验室主目录的
落地收据见[入口核验](../maintenance/storage/LAB_ROOT_MAIN_ADOPTION_20261007.json)。
受保护的ABO/OpenMoji历史运行目录尚未全部改用主线配置，不能覆盖现用实验目录；
主目录源码到位也不等于SDK、数据、PT及现场回归已全部通过。

## 现在从哪里开始

```powershell
Set-Location C:\path\to\2026OpticsMoE
conda activate xml

# 检查基础Python/Torch环境；不代表任务数据或硬件已就绪
python -m LightGenV2.scripts.check_environment

# 不读取正式数据/PT的T06 CPU模型合同检查
python -m pytest experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/tests/test_model_and_training.py -q
```

T06 的训练、评估、硬件六阶段和打包命令全部集中在
[`tasks/t06_video_quality_assessment/README.md`](tasks/t06_video_quality_assessment/README.md)。
T06旧兼容profile可用 `check_environment --task t06 --profile <名称>` 检查。
缺失PT/输入或源码SHA不匹配会返回失败，不能把基础环境检查通过当作当前实拍版就绪。

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

## 八任务进展总表

老师查看整体进展、性能、速度、功耗和待办时，统一使用
[`PROJECT_SCORECARD.md`](PROJECT_SCORECARD.md)。该表固定每个任务一行；没有可追溯证据的
字段保持 `—`，不得凭印象补数值。

所有任务的 train/test、无 validation 选模方式和 Router 口径见
[`DATA_SPLIT_AND_ROUTER_PROTOCOL.md`](DATA_SPLIT_AND_ROUTER_PROTOCOL.md)；冻结 Qwen 在 RTX
5090 D 上的统一性能、速度和功耗测量边界见
[`RTX5090D_QWEN_BASELINE_PROTOCOL.md`](RTX5090D_QWEN_BASELINE_PROTOCOL.md)。
论文总表如何同时呈现 Ours、D2NN 和 Frozen Qwen 三组，见
[`PAPER_TABLE_GUIDE.md`](PAPER_TABLE_GUIDE.md)。

## Git 同步规则

代码修改完成后必须测试、commit 并 push GitHub；服务器和实验室电脑只通过 Git 拉取
源码。大权重、缓存、CCD 和 ZIP 不进入 Git，继续通过 SHA256 清单传输。禁止以 SCP
直接覆盖源码，也禁止强推 main。完整要求见 `AI_RULES.md` 第 15–20 条。
