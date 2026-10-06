# T07 ABO 图搜图：封存 rank72 最终版

本页是 main 的正式任务入口。T08 的图搜文、文搜图属于另一任务，不能用本页结果替代。
历史协议、baseline、测速仍保留，不以最新模型覆盖历史证据。
本机旧DVP/MNIST启动脚本不是rank72入口，具体用途和不可直接运行的路径边界见
[历史调试区说明](lab_dvp8um/README.md)。MNIST测速保留，不套给ABO最终权重。
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

实验室原验证工程：
`E:/code/guest/2026OpticsMoE/ABO_I2I_Lab_DVP_8um/candidate_latergb_rank72_20260930`。
硬件400µs、Gain X4、wait240、已核验ROI/方向、同仿真的保零有界BMP合同；逐层全量、
单采集进程，旧暗帧隔离保留不混入有效CCD。
离线回放、SHS控制层与逐层runner源码已归主线，见[硬件入口边界](hardware/README.md)。
SDK路径前检不等于设备回归或运行工程切换，不能只换PT就宣称main可完整接管硬件。

## 报告、baseline与历史保护

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
