# T07 ABO 商品图搜图：独立光电工程

当前入口为 `python run.py`，只使用本文件夹的 `standalone/`，**不导入T01或旧experiments，也不加载完整Qwen模型**。
日常步骤看 [COMMAND.md](COMMAND.md)；维护、导出与历史数值证据看 [复现入口](reports/reproduction/README.md)（源码仓库内）。

## 给老师检查的结构

- 固定图片224×224 + 固定英文检索指令（见standalone/data.py）；无类别/商品标题输入。
- 冻结Qwen patch Conv3d、视觉位置表、主merger，以及固定prompt用到的token embedding行。
  原始tokenizer和processor保留；未知prompt/token会报错，不能冒充支持任意语言输入。
- V：196×1024 → Linear/LN到192 → 两级光电融合 → 投影回1024并加输入跳连 → merger得到49×2048。
- 图像特征填入文本模板的image token位置；L：S×2048 → 192 → 两级光电融合。
- 每个模态：光Router一次CCD选Top-2；四个专家中的两个一起传播/CCD；融合后电子重编码、再加载全局相位/CCD。
  V/L共六次捕获，不是三层专家再额外加Router。Router能量标准化、softmax、Top-2仍为电子运算。
- 电子残差：V两组3×3 depthwise二维卷积，L两组核长5因果一维卷积；每组含192→384→192通道MLP。
  无attention、Transformer、VGG或额外图像分支。当前未采用失败试验中的扩核/增强读出头。
- 同尺度融合：En=E/rms(E)，On=O/rms(O)，M=(1-alpha)En+alpha On，F=rms(E)M/rms(M)。
  尺度统计停止梯度，alpha范围[0.01,0.95]，best约0.087～0.104；不是性能贡献比例。
- 最后有效token mean/max拼成384维 → LN → Linear64 → L2。头仅25,408个参数。
- CCD明确保留 `frame mean → clip12 → log1p → pool224 → row LayerNorm → ReLU → Linear192`；
  **不是纯线性CCD读出**，不得在部署时擅自去掉并继续引用当前指标。
- 专家224×224，2×2排列、间隔30，有效场478×478，FFT画布518×518；532nm、10cm、逻辑17um。
  8um相位SLM需物理映射，不可直接把224逻辑像素当硬件像素。相位为2πsigmoid(raw)，当前从best续训，不是全零初始化。
- 独立模型保留原相位、融合及电子权重。只删除无输出作用的L隐藏状态重构投影、原始TF模块、LM头、
  未用的词表行、旧优化器和历史候选。输入前端约2915万冻结参数，可训练约278万。

## 当前结果及边界

固定best源SHA256：`3674981c3499555077c7eadc3072a11e07583675000ec4c012616a96185867f5`（epoch10）。
旧版A100为70.0000%；旧版4090为70.2083%。独立版最终隔离目录4090完整复评为70.2083%、去光67.2917%，
测试embedding对旧版平均余弦0.99999756；存在微小混合精度差，不宣称逐位相同。最终实跑结果见包内reference和[验收记录](ACCEPTANCE.md)。
独立assets约85.6MB，包含processor、模型、仅训练样本的教师目标；无完整2B模型。

ABO similarity10：10类200商品、每商品12视角；train/val/test按商品120/40/40，
即1440/480/480图片，val不使用。480测试图查120训练商品中心，**同类别算相关**，不是同商品实例检索。
每个query有12个正候选，报告Hit@1（旧称R@1）、mAP等，不混用positive Recall。
先逐视角L2归一化，再按商品均值/L2；不按真实类别预筛选候选。

训练包括监督对比、训练教师向量/类别中心、Router均衡、相位DC与工作点约束；
训练中20%～30%相干未调制分量、截断偏置高斯CCD噪声、相位旁路；位移=0、k空间约束关闭。
评估是确定性理想仿真，**不是实际光路或固定20%零级光测试**。
EMA、每5epoch test选best，仅best/last；成绩为test-selected，不能声称独立无偏测试。
冻结完整Qwen baseline历史95.2083%，当前仍明显落后。完整baseline需另行加载大模型，
不混入此精简学生运行入口。

## 文件怎么找

| 文件 | 内容 |
| --- | --- |
| run.py | 独立命令入口 |
| standalone/frontend.py | 冻结Qwen必要前端，无模型/Transformer类 |
| standalone/model.py | 电子残差、融合、图文组装、检索头 |
| standalone/optics.py | Router、相位、传播、CCD及实测注入接口 |
| standalone/data.py | 固定数据协议、图库和指标 |
| standalone/cli.py | 评估、去光、独立微调、报告和相位图 |
| standalone/export.py | 一次性CPU选取原权重；交付接收人不需要执行 |
| assets/ | 运行资源（在交付包内；不进入Git） |
| reference/ | 交付版本实跑报告（在交付包内） |

源码中的 `legacy_run.py`、旧configs/refinement/retrieval_contract只保留历史审计，不进入独立ZIP。
旧后端被其他任务共用，不做全仓删除。历史证据仍在Git和runs；本次不删除数据/旧权重。
`build_lab_package.py`白名单打包当前代码、assets、ABO数据和reference；名字沿用公共规则，
**此次是仿真/微调包，不是包含SLM/CCD厂商SDK的实验室控制包**。实测注入接口不能冒充已验收的自动采集流程。

GPU默认一张，同一助手最多两张；退出后确认自己的PID已释放，不杀他人进程。
