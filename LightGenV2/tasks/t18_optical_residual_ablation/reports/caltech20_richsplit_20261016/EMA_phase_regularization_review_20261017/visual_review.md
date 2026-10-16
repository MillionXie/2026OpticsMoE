# 实际图像视检

两张PNG均已用实际图像查看工具逐张核验，SVG由同一绘图调用输出；没有PDF或旧图重绘。

- `figures/accuracy_depth246.png`：2/4/6层同一图中给出TRAIN/VAL/开发TEST，全部accuracy轴60–100%，未绘macro；六层为实际120选E108的同一权重，95.57/78.85/80.29及+1.95百分点与CSV一致。六层330/411仅较四层329/411多一张，不解读为稳定深度优势。
- `figures/EMA_phase_training_review.png`：真实101–120轮两条训练、验证轨迹及raw相位粗糙度，不做平滑；训练/验证accuracy轴从60%起，粗糙度用本身的非百分比量纲。图例/标题/说明无重叠或缺字；完整100轮prefix仍在CSV。

图中两方法仅相位TV系数不同；小验证/NLL差异与多次开发TEST限制在正文说明。光学结构与电子参数未改变，冻结100轮对照未重训。所有图、绘图源与聚合CSV的SHA见 `figures/artifact_sha256.csv`；原始预测、PT、逐样本ID和完整日志不公开。
