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
