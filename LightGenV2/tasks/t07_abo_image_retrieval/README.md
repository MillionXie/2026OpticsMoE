# T07 ABO 图搜图：封存 rank72 最终版

本页是 main 的正式任务入口。T08 的图搜文、文搜图属于另一任务，不能用本页结果替代。
历史协议、baseline、测速仍保留，不以最新模型覆盖历史证据。
本机旧DVP/MNIST启动脚本不是rank72入口，具体用途和不可直接运行的路径边界见
[历史调试区说明](lab_dvp8um/README.md)。MNIST测速保留，不套给ABO最终权重。
历史`build_lab_package --dvp-overlay`仅导出原DVP相机适配器，现从共享目录
`LightGenV2/hardware_common/dvp_legacy.py`的已提交版本读取，避免依赖本机未跟踪副本。
原适配器SHA为`cab6e0ef3cdbf0840666e5f0d17baa26821011772d394a1c0b1a3f386028ca06`；
临时包字节及manifest校验通过，未加载SDK或连接设备，不代表rank72完整部署包。
冻结 CLIP／YOLO／DeepSeek 的历史图搜图对照见
[三方向 baseline 历史表](../t08_abo_image_text_retrieval/reports/reproduction/ABO_BACKBONE_RESULTS_20260923.md)，
该表旧 Ours 行不是下方封存 rank72 结果。

## 最终身份与结果

| 项目 | 当前封存值 |
|---|---|
| 模型 | 六层光学＋原电子残差，rank72 轻量 RGB token 旁路 |
| 旁路参数 | 92,168；不加载完整 Qwen/attention/Transformer |
| 权重 | epoch13，SHA256 `25f23260864a00a0f32b5de209c27f5876166018c8d8ab0e880de81e0d2c2c22` |
| 正常仿真 R@1 | .83375 |
| 同权重去光 R@1 | .8000 |
| 未微调实拍 R@1 / R@5 | .81125 / .94125 |
| 实拍 MRR | .8694430763 |
| 数据/采集 | 1600真实图库＋800真实查询，六层各2400，共14400有效CCD |
| 选模边界 | 原800查询在开发期使用过，不称独立泛化 |
| 专属测速 | 尚无已核验的 rank72 测速；不套用其他权重的时间 |

视觉与语言内部 alpha 约 .445–.448；视觉/语言各 router、expert、global 三次捕获，
Top2 路由。冻结前端的 RGB patch token 经 `1024→72→192` 小旁路后，仅在
最后语言 block 的49个图像 token 槽位按 .5/.5 混合，再进入原空间池化线性64维读出。
这不是两个检索模型在输出端混合；相位和光学合同未为整理而改变。

## 唯一源码入口与复现边界

模型入口：`LightGenV2.tasks.t07_abo_image_retrieval.standalone.model.OpticalRetrieval`；
CLI：`python -m LightGenV2.tasks.t07_abo_image_retrieval.standalone --help`。
main 收录实际服务器提交 `d979b53ed509907a3630bd2a36c317dfb3aac965` 的原始模型及
CLI相对依赖；没有按时间戳拼接其他权重架构。15个核心运行源码已核验SHA一致。
封存PT的严格CPU加载、结构审计及两个合成token读出测试通过；这不是重新评估800查询。
源码清单及审计见 [归并记录](../../../maintenance/storage/T07_RANK72_SOURCE_PUBLICATION_20261004.md)。

权重本地：`handoffs/abo_latestfresh35_lab_20260930/rank72_epoch13_best_snapshot.pt`。
数据及前端/processor 资产按原 manifest 保留，不进Git；CLI的 `--assets`、`--data`
必须指向原已核验资产，不能把其他ABO split或模拟图库代替本次实拍图库。
固定评估必须显式指定上述PT和完整SHA；新输出目录不得覆盖历史结果。
当前不安排重训、采集或新查询评估；用户已将该权重封存为最终版，外部上传暂停。

2026-10-07补齐既有 baseline 引用的 `standalone.enrolled_abo` 划分工具，
它与上述原提交及实际受保护服务器源码逐LF SHA一致，未改数值或划分方法。
200商品各12图按固定 `sha256(abo-enrolled42:<sample_id>)` 排序，8图库/4查询，
拒绝照片数量不足、跨图库/查询完全重复及覆盖已有输出；这不是按模型分数挑数据。
源码与依赖见[划分工具身份](enrolled_protocol_source_import_20261007.json)。
7项合成文件测试通过，没有重建原protocol或读取正式图像，原2400图身份与数据仍保留。
工具仅保留复现能力，不授权重新prepare来替换封存协议；其source_commit等provenance
会随重新生成改变，不能仅凭样本数相同认定协议SHA一致。闭合SKU图搜图不是未见商品
或独立采集会话泛化，近重复/同转台会话仍按原限制报告。

实验室原验证工程：
`E:/code/guest/2026OpticsMoE/ABO_I2I_Lab_DVP_8um/candidate_latergb_rank72_20260930`。
硬件400µs、Gain X4、wait240、已核验ROI/方向、同仿真的保零有界BMP合同；逐层全量、
单采集进程，旧暗帧隔离保留不混入有效CCD。
离线回放、SHS控制层与逐层runner源码已归主线，见[硬件入口边界](hardware/README.md)。
SDK路径前检不等于设备回归或运行工程切换，不能只换PT就宣称main可完整接管硬件。

## 报告、baseline与历史保护

`configs/optical_top2_dc20*.yaml`、`refine_*.yaml` 与 `polish_*.yaml`
是9月9日旧图搜图训练及必要对照配置，不是rank72最终配置。2026-10-08核验
这9份文件与原提交 `d979b53ed509907a3630bd2a36c317dfb3aac965` 及服务器受保护
工作目录逐LF SHA一致，原样纳入main；已有历史复现说明和baseline源码清单引用
它们，因此不作为无用试错删除。此次只解析配置和核验来源，不执行旧训练，
不改其中历史数据/PT路径，也不表示旧资产在新机器上已就绪。

旧[9月23日DVP全800查询报告](reports/20260923_dvp8um_full800/README.md)也保留为
历史对照：c9926cba主体、800真实查询对1600仿真图库，仿真R@1 .825、实拍.1975，
不是本页rank72双侧真实图库／查询合同。原说明中的故障归因是当时解释，
此次只做原产物身份保全，未重新验收设备或证明因果。800逐查询report.json保持
原位，并与既有恢复ZIP逐字节相同，SHA b8b41e15…93e8ea1b；仅该具体产物退出
Git待同步列表，原README及新报告／源码／合同仍可见。测速和原CCD不删除。

`LightGenPublic/tasks/t07_abo_image_retrieval` 是旧独立审阅包：epoch15 EMA、480条查询、
120个训练商品中心、Hit@1 .689583／去光 .645833，成功定义为同类其他商品。
它不是本页rank72、不是同一检索协议，也不是实验室硬件入口。原源码、说明和资产
继续保留，不用其指标或架构覆盖封存版本；日常修改只从本页正式入口进入。

正式报告：`handoffs/abo_latestfresh35_lab_20260930/rank72_physical_full_report_20261001.json`，
SHA256 `94947127262ead3fc440b5e67b513f0038949b25795c5be0b916305d251e6eef`，含800逐查询预测。
权重/报告身份见 [FINAL_IDENTITY](reports/reproduction/FINAL_IDENTITY_20261002.json)。
原始有效CCD、收据、原数据、best/last及baseline保持原位置，整理不复制或删除。

当前工作目录的完整历史README保持原样；Git `508a1c35447e0ef91de57251ef24750fab715a21`
保留其全部内容。旧480-query/120商品中心协议、不同数据集筛选、旧权重实拍、
完整测速等不是本页800-query/1600真实图库结果，不混表、不丢弃、不按“不是最终版”删除。
历史复现说明及baseline记录继续保留在原 `reports/reproduction/` 与对应run中。
本机和训练服务器主目录已实际使用main；实验室原工程仍保护，未整仓切换。
剩余独有代码、资产和部署例外仍须逐项核对，本页不表示整仓整理已完成。
