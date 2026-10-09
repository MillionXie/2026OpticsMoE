# 新数据集：MangoLeafVarietyBD v2原始图像

用户2026-10-10要求换用工程未出现、CC BY4.0的新数据集重做残差消融。
作者来源：https://data.mendeley.com/datasets/hb3kvgfcvm/2，DOI10.17632/hb3kvgfcvm.2。
作者明确CC BY4.0、2744张原始照片、8个芒果品种。并非病害MangoLeafBD
（后者CC BY NC3.0）；不混用名称或许可。Beans原作者为MIT，未选；
本次仓库源码/配置/报告全局检索未发现MangoLeafVarietyBD或其DOI使用记录。
类别以作者ZIP实际目录的8个名称排序保存，不假定与旧Kather编号语义相同。

下载固定v2作者ZIP约4.78GB，核对作者API SHA256：
8fa888ec8b2795f9f2e1785a2343201998d3eb005c22f3f04621680edaddc80b。
必须恰有2744原始图片、8类，否则暂停报错，不能擅自挑子集。
原ZIP保留；EXIF纠正后RGB双三次150用于缓存，随后原固定RGB100振幅编码。
固定seed20261010按类精确像素重复组70/15/15划分，保留全部样本，重复组不能
跨集合；保存原图ID、像素组哈希、支持数、缓存SHA。具体大小以准备manifest为准。
叶片／植株身份未提供，仍是图像级研究，不能声称植株独立泛化。

继承T18用户批准公式U'=[0.7exp(i phi)+0.3]U（未调制振幅系数而非功率），
只作用专家/global相位；router不变。无残差rho0从头配对训练。
2/4/6主干、九密集专家、逐层OEO、8CCD窗口、无CNN/Linear，30轮seed17，
优化及数据增强完全复用T18。此新数据类别空间下窗口几何不改。
按各组验证NLL选模，配对数据顺序和增强哈希锁定后各测试一次；不看测试调rho。
准备阶段CPU不占GPU；按2→4→6深度依次，每深度两组最多两GPU并行。
`mango_suite.py`自动准备、smoke、配对训练、一次评估、汇总；报错保留日志，
不覆盖旧Kather结果，不改旧图。数据不进Git，源码仅main Git发布同步。
运行数据根：`/DATA/DATA1/guest3/t18_mango_variety_v2_20261010`。
运行目录：本任务`runs/simulation/mango_variety_s17_20261010`。
当前仅协议，未宣称残差或更深网络一定更好。
