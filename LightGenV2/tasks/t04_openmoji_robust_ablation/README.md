# T04 OpenMoji 鲁棒性消融：当前统一入口

## 2026-10-08 授权：累计组电子适配，目标89.0%–89.5%（进行中）

暂不推进单措施消融。历史87.05%是早期editor16累计G5的真实CCD适配：
原clean CPU .9165、直接 .5730，独立TRAIN1000梯度，原decoder30162参数、
160epoch、TEST1000每5epoch开发选模，第105轮最佳，strict CPU .8705。
原PT SHA094d01157e63d56d195f154bc0671e9e2ce43ac32df34f47d893ea233c0489a4；
适配best SHA7f3268c8d94dd2ec1389a9ebc247b0a1a5301a9ae9e43315177364803cd5e40a。
这不是后来直接.7315的PT，不能将两个版本拼成同一恢复链。
历史report/strict_reload/history在私有`editor16_robust_chain_20261003/g5_adapter`保留。

后来的累计对齐PT d4481ef804dbd8f27f2f24f98c08dc7bf161b3137cefcc2613310291fe56f653
自身CPU仿真.9120、直接.7315；已有末端适配.8670，低学习率160轮及另一320轮配置
最高strict CPU均.8685。320轮结果best SHA
cb1864fe7217bb4d0891d3712c37173877c84746baa4ed756fbdc989e7a77140。
均按无trick仿真.8950作为用户目标参照；89.0%即相差0.5个百分点，不换成较低参照。

新有限run `editor16_align_decoder_lr1e5_wd005_160_20261008`已派发既有实验台任务
`OpenMoji_AlignDecoder_LR1e5_WD005_1008`：从精确d4481ef8原部署PT重新训练，
同独立TRAIN1000/固定TEST1000，各原6000CCD只读复用；AdamW lr1e-5/wd.05，
160epoch/seed927，TEST每5轮最高选PT开发口径，无TEST梯度/无VAL选模。
仍只训练原decoder，不改光学相位、alpha或架构。理由是历史TRAIN接近饱和、
延长320轮未提高TEST，有限检验低学习率与更强正则化，不保证目标或宣称单因素因果。
派发成功不等于开始梯度；须核CPU缓存基线.7315、实际epoch/loss及终了strict CPU。
原上游保护SHA f385956d653201b10c1bcc1bb3e8bd1f6a25a987fc328f660e6666d26e3bf66f
必须不变。最终保存best/last/逐样本/报告和SHA，确认进程及GPU释放。

沿用Git封存源码`bc1951cfbd0bcbd60a6111884286b506790e20fb`，未修改实验台源码；
入口`lab_editor16_robust_chain.py` SHA0adc3fc6...bccb62、原实际适配backend
`lab_tune_g2_test.py` SHAd4c0c4fd...2c9a0f；来源逐项见
`maintenance/storage/T04_WINDOWS_SOURCE_PRESERVATION_20261005.json`。
参数仅在任务命令中配置，不新增工程/分支/worktree，不打开SDK、不重采、不覆盖旧run。
原始CCD/PT和ABO等封存保持，外部上传暂停。当前没有本run的新精度。

## 2026-10-07 新授权：单对齐空间扰动训练（结果待实拍）

不改变下方封存rank64结果。editor16的新对照仍使用同一初始权重、完整TRAIN5000、
seed73和120epoch；插值概率1、单前向、paired-clean=0、consistency=0，不含CCD/DC，
不做实拍后电子微调。原光电参数共同训练，alpha/frontend冻结。
`train_editor16_robust_chain.py`新增TRAIN-only `--spatial-shift 1`
`--spatial-rotation .25 --spatial-dropout .02`：复光场实虚部共享仿射、零填充，
16×16粗空间遮挡，无inverted-dropout放大；每次光学调用/批共享几何和遮挡。
平移单位是仿真像素，旋转单位度，范围均为实验假设而非设备标定。
eval关闭全部新扰动，默认参数全零保持旧路径；不增加推理层。
四项合成测试在服务器Torch通过（恒等、零场/梯度、实虚一致、遮挡不放大），
不能代替新权重的严格CPU审计或实际光测。新PT必须独立新CCD，不复用旧上游采集。
仅在既有运行工作树按Git精确任务文件同步，保护原overlay，不新建工程副本。

封存指标日期：2026-10-02；源码治理状态更新至2026-10-06。主要仿真、捕获和适配源码
已归主线，但不是已经通过独立环境/设备回归的完整部署包，也不覆盖其他窗口的新实验。
旧 `t04_semantic_interaction` 是共享模型/数据后端，
不是可随意删除的试错副本。

## 当前 rank64（不要混用 rank16/rank48/完整头）

仅取回主线时可先看[rank64封存版本说明](reports/reproduction/RANK64_SEALED_20261002.md)，
其中有原仿真／直接实拍／微调的完整对应及PT路径；原私有完整报告和逐样本未搬走。
该说明绑定2026-10-02版本，不覆盖其他窗口的新实验。

主线入口已解除旧ABO控制脚本及资产工程source配置的依赖：采集显式提供原机器
配置、相位SDK/LUT，由已审计的主线SHS模块控制。末端微调也从主线读取模型配置。
见[参数与剩余资产要求](reports/reproduction/README.md)。本轮未切换实验台现用目录，
未打开设备；现场回归仍待排期，不能据此声称新测精度或全部迁移完成。

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
| G5校准最佳 | `1fa31ec7b30a554280d9115b54f580d40b9c754f805db4ec9714ec28d22af41f` | [主线G5版本及原报告身份](reports/reproduction/RANK64_SEALED_20261002.md) |
| G2校准最佳 | `1cbc3d2574827272dafee7102a4a402ace7eab3f373ea177cbc02e66d90fe6db` | [主线G2版本及原报告身份](reports/reproduction/RANK64_SEALED_20261002.md) |

原私有资产目录`handoffs/openmoji_robust_ablation_20260928/midrank48_candidate/`
包含 PT、report、history、逐样本和严格默认推理重载记录；不复制大权重进 Git。
该目录16份此前无恢复身份的历史辅助程序已原字节三端保全，见
[用途与恢复清单](../../../maintenance/storage/OPENMOJI_HISTORICAL_TOOLS_20261006.json)。
它们含rank48诊断、早期rank64准备、已过期队列及被拒绝的额外residual试验，
不是16个当前入口，也不能替代本页rank64最终协议；原PT/CCD/结果与测速均保留。
主线入口不要求新机器先拥有该私有目录才能阅读版本说明；原完整报告按封存页SHA定位。
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

本轮不启动新训练/采集/评估、不恢复外部上传。现有源码的发布/测试不等于
完整机器资产闭包、现场回归或实际运行目录切换；现用工程继续保护。

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

补充TRAIN准备入口为 `lab_prepare_extra_train1000.py` 和
`lab_prepare_extra2_train1000.py`，按实验台原脚本原逻辑保留（仅文件名归入任务目录）。
依次使用seed1002／1003、每种操作250条，排除TEST和此前TRAIN的ID及原图SHA重复；
输出已存在时拒绝覆盖。这两个入口不会调用设备或选择模型。仍需提供原project及
相邻 `OpenMoji_Lab_SHS_8um/data` 数据位置；本轮仅用临时合成文件验证，未生成正式新划分。
