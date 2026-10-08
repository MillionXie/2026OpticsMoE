# OpenMoji 当前复现入口

当前版本身份、指标和物理合同只维护在[任务 README](../../README.md)。
仿真、显式rank64捕获、TRAIN2000末端微调及bias校准源码已归主线；
机器资产/SDK闭包、现场回归和实际运行工程切换仍未完成，不把源码发布说成设备就绪。

原已验证复现证据：

- [主线内rank64封存版本、三类指标和PT位置](RANK64_SEALED_20261002.md)：包含G2/G5、冻结边界和原完整报告SHA，不依赖私有文件才能阅读。
- [rank48历史容量对照](RANK48_HISTORY_20261002.md)，不覆盖rank64或其他窗口的新实验。

原完整报告和两路线过程日志仍原位保存在私有资产目录
`handoffs/openmoji_robust_ablation_20260928/midrank48_candidate`，通过登记SHA查找，
不将包含旧定时作业指令的过程日志当成当前执行入口。

复现所需：精确代码/overlay、配置、原G2/G5 PT与适配 PT、OpenMoji素材许可、
源身份互斥 TRAIN/TEST manifests、词嵌入/视觉前端、各层真实CCD与收据及设备合同。
代码通过 Git 同步，数据和 PT 通过 manifest+SHA；不上传凭据，不把暗帧或旧mask数据混入。

## 主线入口的操作顺序（2026-10-05治理核对）

### 原封存rank64仿真训练程序

原服务器根目录的 `train_rank64_common_20261002.py` 已纳入任务模块
`LightGenV2.tasks.t04_openmoji_robust_ablation.train_rank64_matched`。
仅移除指向旧工作树的 `sys.path` 注入，训练体与原SHA
`f8e0786a90a1b45a911020251dfddffa34436aa1890a3ba622c2a7b44d23eef1` 对应；
源码保全测试会重建原文件并验证SHA。原文件、启动脚本、权重和结果仍保留。

从已发布主仓库使用 `python -m LightGenV2.tasks.t04_openmoji_robust_ablation.train_rank64_matched`，
参数为 `--group r0_base|r3_ccd_dc30_grid --epochs 15 --steps 100`，可显式提供
`--resume`、`--output-root`。此处只是原接口记录，不授权重跑或覆盖正式run。
该原程序使用TRAIN5000梯度、原TEST1000每五轮开发选模，正常仿真评估临时关闭噪声/DC/grid；
不是 `train.py` 的4000FIT/1000VAL协议，也不是实拍后仅decoder微调。
资产仍由 `train.py` 的BASE/SOURCE及resolved_config定位；必须先取得并核验原资产，
不能把源码保全测试说成已从零训练复现或新实测。

### 已有实拍与微调入口

以下是参数合同，不是允许立即重跑已完成实验的指令。本轮未启动训练、采集或正式评估。
在已发布源码仓库根目录使用模块入口；原实验工程作为资产位置，不把其 `source/`
悄悄覆盖为主线。先核验原PT及所有资产，再排期现场回归。

1. 采集使用 `LightGenV2.tasks.t04_openmoji_robust_ablation.lab_shs_capture`，必须显式给
   `--group g2|g5` 与 `--lab-identity` 指向本任务 `configs/lab/rank64_20261002.json`。
   缺少身份参数时仍是历史rank16，不是rank64。`--project` 为资产工程，`--scope test|train`、
   `--output`、`--limit` 必须对应本次数据身份；CPU和2000us合同不变。
   主线采集还必须显式传 `--machine-config`（原机器本地控制配置）、`--phase-sdk`、
   `--phase-lut`，可传 `--amplitude-sdk`。只读取配置，不改写原文件。
   控制源码使用主线已审计的SHS bench和几何模块，不再从旧ABO工程导入控制脚本。
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
`OpenMoji_Lab_SHS_8um` 内data、assets、frontend。模型配置来自运行主线仓库，
不再依赖资产工程 `source/` 副本；两份配置与原实验台核验一致（YAML换行差异已分开记录）。
这不是不带资产即可运行的独立包。设备SDK授权、LUT、相机配置和现场显示状态也未由
文档或合成测试证明可用。正式运行目录尚未切换；不能把Git引用同步当作现场迁移。
