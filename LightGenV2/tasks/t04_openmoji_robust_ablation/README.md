# T04 OpenMoji 鲁棒性消融：当前统一入口

核验日期：2026-10-02。模型源码整合仍在进行；本目录先建立当前入口和精确身份，
不是已经通过独立环境运行的完整部署包。旧 `t04_semantic_interaction` 是共享模型/数据后端，
不是可随意删除的试错副本。

## 当前 rank64（不要混用 rank16/rank48/完整头）

| 版本 | 原正常仿真（实验台 CPU） | 同权重直接实拍 | 原 decoder 微调 | 已有 bias 校准后 |
| --- | ---: | ---: | ---: | ---: |
| G2 无 robust trick | .9390 | .6185 | .9015 | .9045 |
| G5 CCD噪声+DC30%+训练内网格代理 | .9270 | .6910 | .9180 | .9305 |

服务器正常仿真 G2/G5 为 .9385/.9290；和实验台 CPU 计算路径分别列出，不能混成同一测量。
G1 理想与 G2 直接部署共用原 PT。G5 校准后 PT 自身理想仿真 .9295，不用它降低原 G1 .9390 比较基准。
共享头271,384参数；实拍仅训练原 decoder 的30,162参数，不增加层、不改光学上游/alpha。
微调用独立实拍 TRAIN2000 梯度，原 TEST1000 每5epoch选PT及 bias 校准，用户明确授权；
TEST 不进梯度，但这些是开发指标，不是独立测试。保留未校准最佳、last和校准最佳。

| 最终候选 | SHA256 | 原证据入口 |
| --- | --- | --- |
| G5校准最佳 | `1fa31ec7b30a554280d9115b54f580d40b9c754f805db4ec9714ec28d22af41f` | [G5最终报告](../../../handoffs/openmoji_robust_ablation_20260928/midrank48_candidate/FINAL_RANK64_VERSION_20261002.md) |
| G2校准最佳 | `1cbc3d2574827272dafee7102a4a402ace7eab3f373ea177cbc02e66d90fe6db` | [G2 TRAIN2000报告](../../../handoffs/openmoji_robust_ablation_20260928/midrank48_candidate/G2_RANK64_TRAIN2000_RESULT_20261002.md) |

上述目录包含 PT、report、history、逐样本和严格默认推理重载记录；不复制大权重进 Git。
每组 TEST1000 与独立 TRAIN2000 各六层，分别6000与12000真实CCD，保留源身份及收据，不跨组复用。
2000us/GainX4/wait240、既定ROI和方向、保零 bounded BMP；原捕获合同不变。
当前 rank64 精确版本的单样本延迟尚未测量，不能借用旧完整头或rank48时间。
正式报告的逐文件SHA与四项候选身份已登记在本任务的
[封存证据清单](reports/reproduction/FINAL_IDENTITY_20261002.json)；原完整报告/逐样本数据没有搬走。

## 实际代码与迁移边界

- 仿真实际源码：训练服务器 `.worktrees/t04_openmoji_robust_20260928`。
  已保护 HEAD `3b956503a4ecae9b6c20c789ead28621e42e6eeb`，另有未提交 profiles/README 和 rank64 runner，
  以 SHA overlay 保留；单拿 HEAD 并不代表最新版。
- 实验台：`E:/code/guest/2026OpticsMoE/OpenMoji_Robust_Rank64_SHS_20261002`，
  使用 `source/LightGenV2/tasks/t04_openmoji_robust_ablation`、共享模型及原设备模块。
- [源码来源清单](../../TASK_REGISTRY.json)；[复现入口](reports/reproduction/README.md)。
- 迁移必须连同 lowrank64 共享头支持、bounded profiles、CPU CCD 边界、TRAIN2000 微调与已有 bias 校准核对；
  不能把旧默认 rank16 捕获脚本直接标为 rank64 最新版。

当前不启动新训练/采集/评估、不恢复外部上传。迁移通过严格 PT 重载与依赖测试后才发布到 main。

### 2026-10-05 源码治理补充（不覆盖上面的实验结果）

已发布共享 `lab_runtime` 和历史 `standalone` 依赖，实际服务器导入及六项合成CPU边界
测试通过；后者仍是历史标准头入口，不是rank64部署默认。实验台36份源码已用Git原字节
封存，其原目录和有效实验数据未动。

rank64原G2/G5权重的精确身份现集中在
[`configs/lab/rank64_20261002.json`](configs/lab/rank64_20261002.json)，两份原PT在实验台
重新计算SHA与该配置一致。`lab_checkpoint_identity.py`提供不加载模型／设备的文件SHA和
架构身份检查，九项测试通过；测试无需私有归档Git引用。
统一 `lab_shs_capture` 已接入显式 `--lab-identity`，不传仍是历史rank16，不能只换PT来
复现rank64。rank64要求CPU／2000us，并采用实验台原有保存前暗帧守卫；22项无设备单元
检查和实际服务器依赖导入／六项合成CCD边界检查通过。尚未做现场采集回归，
TRAIN2000微调及bias校准入口已按实际代码归入主线。现用rank64原工程继续保护，
完整迁移、设备依赖和现场回归尚未完成；这些治理检查不代表重新测量准确率。

统一入口的rank64选择参数为：
`--lab-identity LightGenV2/tasks/t04_openmoji_robust_ablation/configs/lab/rank64_20261002.json`。
项目／数据／设备依赖仍需提供原审计配置；本轮不自动执行该入口或替换现用脚本。

TRAIN2000入口为 `lab_tune2000.py`，已有bias校准入口为
`lab_calibrate_rank64.py`，均显式使用 `--group g2` 或 `--group g5`；不再依赖
运行过程中偷偷替换组别映射。保留原TRAIN2000梯度、每五轮TEST开发选模、仅原decoder
微调及现有edit_head.bias校准，不增加网络模块。TEST选模结果不是独立泛化指标。
五项入口合同测试及实际服务器九项CPU依赖／合成边界检查通过，包括原rank64 decoder
30162参数、bias保存后严格重载和默认门限等价、上游修改识别。这些测试未加载正式PT，
未重评数据集或打开设备，不能替代完整资产配置与现场回归。
源码发布 `20b8e46951a334a6414aca23578cef26356111b4` 已同步本机、GitHub及服务器main引用；
实际运行checkout、原数据与权重没有切换或修改。
