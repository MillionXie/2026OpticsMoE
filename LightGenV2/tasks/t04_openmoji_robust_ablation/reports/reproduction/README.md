# OpenMoji 当前复现入口

当前版本身份、指标和物理合同只维护在[任务 README](../../README.md)。
操作源码尚未完整合入本目录；不把“已有入口”说成“已完成源码统一”。

原已验证复现证据：

- [rank64 G5 最终权重、strict reload、冻结参数与原数据位置](../../../../../handoffs/openmoji_robust_ablation_20260928/midrank48_candidate/FINAL_RANK64_VERSION_20261002.md)。
- [rank64 G2 TRAIN2000、最佳/last及校准权重SHA](../../../../../handoffs/openmoji_robust_ablation_20260928/midrank48_candidate/G2_RANK64_TRAIN2000_RESULT_20261002.md)。
- [rank48 历史两路线记录](../../../../../handoffs/openmoji_robust_ablation_20260928/midrank48_candidate/TWO_ROUTES_20261002.md)，仅作重要容量对照，不覆盖 rank64 结果。

复现所需：精确代码/overlay、配置、原G2/G5 PT与适配 PT、OpenMoji素材许可、
源身份互斥 TRAIN/TEST manifests、词嵌入/视觉前端、各层真实CCD与收据及设备合同。
代码通过 Git 同步，数据和 PT 通过 manifest+SHA；不上传凭据，不把暗帧或旧mask数据混入。

## 主线入口的操作顺序（2026-10-05治理核对）

以下是参数合同，不是允许立即重跑已完成实验的指令。本轮未启动训练、采集或正式评估。
在已发布源码仓库根目录使用模块入口；原实验工程作为资产位置，不把其 `source/`
悄悄覆盖为主线。先核验原PT及所有资产，再排期现场回归。

1. 采集使用 `LightGenV2.tasks.t04_openmoji_robust_ablation.lab_shs_capture`，必须显式给
   `--group g2|g5` 与 `--lab-identity` 指向本任务 `configs/lab/rank64_20261002.json`。
   缺少身份参数时仍是历史rank16，不是rank64。`--project` 为资产工程，`--scope test|train`、
   `--output`、`--limit` 必须对应本次数据身份；CPU和2000us合同不变。
2. 独立TRAIN两轮的准备入口是 `lab_prepare_extra_train1000` 和
   `lab_prepare_extra2_train1000`，不能仅看文件名判断最终TRAIN2000用哪一轮。
   当前 `lab_tune2000` 明确绑定 `data_train_adapt1000` 与 `data_train_extra1000`，
   补充manifest固定SHA为 `427e9fc203dc2c38bd3d1336ee881f9447ca07923a3a9aab05b3c67bf82db403`。
   第二补充轮不自动替代该manifest。
3. 末端微调模块 `lab_tune2000` 要求 `--project`、`--group`、`--train-run`、
   `--extra-run`、`--test-run`、`--base-cache` 和新 `--output`。
   前三套采集目录分别是首轮TRAIN1000、补充TRAIN1000、TEST1000；每套6000张CCD，
   同组原PT、相位、曝光、方向一致。`--base-cache` 是此前首轮decoder运行生成的
   特征缓存目录，不是原图、另一组缓存或任意PT目录。默认160epoch，模型回放CPU，
   仅原decoder可迁移到选定训练设备。
4. 既有bias校准模块 `lab_calibrate_rank64` 要求 `--project`、`--group`、
   `--checkpoint`（本组微调best.pt）、`--cache`（同运行train_features.pt与
   test_features.pt所在目录）和不存在的 `--output`，保留格默认下限.98。
   输出best_full.pt把校准写入原bias，并严格重载核对默认.5门限；不增加层。

上述TRAIN梯度与TEST开发选PT/bias流程仅用于已批准协议，不称独立泛化。
原正式best/last、缓存和收据保留，不复用旧队列来覆盖输出。
历史脚本用途和禁止误用边界见仓库
`maintenance/storage/T04_RUNTIME_ENTRY_BOUNDARY_20261005.md`。

### 资产与配置缺口

当前入口仍要求资产工程的 `resolved_config.json`、`weights/`，其相邻
`OpenMoji_Lab_SHS_8um` 内data、assets、frontend，以及资产工程source内原光学配置。
这不是不带资产即可运行的独立包。设备SDK授权、LUT、相机配置和现场显示状态也未由
文档或合成测试证明可用。正式运行目录尚未切换；不能把Git引用同步当作现场迁移。
