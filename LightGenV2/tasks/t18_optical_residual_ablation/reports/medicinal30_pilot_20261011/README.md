# 完整30类叶片：两层快速可行性预检

用户要求自然图像、类别多于8、两层精度目标70%、避免长时间等待。
Butterfly长队列已停止，原历史和数据保留。新数据是作者完整Medicinal Leaf v1，
实际30类／1835图（各类34–122），作者已背景处理的自然RGB叶片，CC BY4。
[作者发布](https://data.mendeley.com/datasets/nnytj2v3n5/1)的文字数量矛盾，实际
完整API清单和逐图原生SHA／字节数全部核验，以[data_identity.json](data_identity.json)为准。
固定seed20261011精确像素连通分组，train1273／val263／test299；30类全部保留。
没有可靠叶片／植株身份，不能称生物个体独立，只是图像级探索。

两层短试验`medicinal30_L2_s17_b4_batched_pilot_20261011`于北京时间19:54:11启动，
训练源码`2cbe6ae27a0f606111e257703d731eca9f607bf2`，67个实际来源逐一匹配发布Git blob。
GPU1／2各一个rho0／0.3子进程，GPU4未用。每臂最多30轮、约30分钟上限；
选模仅每轮EMA验证balanced NLL最低首次，至少10轮及patience6，不按accuracy反换。
同EMA未增强全train每5轮及端点，梯度只train；pilot完全不评估test。
两臂可能不同停止轮，不能冒充共同预算正式残差收益。

新执行方式保留九专家独立相位和相干残差、原全场OEO／传播／RGB编码，
无CNN／Linear／电子学习head。micro4、有效batch16加权含尾及辅助项，
每macro一次clip／AdamW／EMA，显存上限3GiB另留800MiB。
30个固定32×32 CCD窗四行8／8／8／6，两层449848可训练参数／11张相位。
旧Butterfly train8图新初始化吞吐[预检](execution_probe.json)记录全部20配置：
两层两rho的micro4通过原定数值门槛，短程每样本约快4倍；16层rho.3的micro2／4
未通过，失败保留、不放宽、不能将两层门槛套到深层。
两层实际新30类数据重新smoke也通过，两rho初始化参数完全相同；该数值核验
不是任意权重逐像素等价证明。首rho.3轮18.0秒，真实完整轮时长须持续从history读取，
不能拿短程内核测试代替整轮训练时间，也不能把速度作为70%已达成的证据。

扩展门槛是无残差L2验证NLL获选EMA的accuracy和macro recall均≥70%，
实际L2≤60秒／轮、16层实际／投影≤180秒／轮且总预算可接受；未达不自动扩展长套件。
未来仍为2／4／8／16层每深度rho0／0.3，先锁共同预算和实际生产micro数值，
所有选择完成后每臂获选EMA单次test。详见[协议](../../MEDICINAL_PILOT_PROTOCOL_20261011.md)。

独立CPU审计只读已完成结果、PT、CSV、history及train／val缓存标签，不执行光模型：

```text
CUDA_VISIBLE_DEVICES='' /home/guest3/miniconda3/envs/xml/bin/python LightGenV2/tasks/t18_optical_residual_ablation/audit_medicinal_pilot.py --run LightGenV2/tasks/t18_optical_residual_ablation/runs/simulation/medicinal30_L2_s17_b4_batched_pilot_20261011 --data /DATA/DATA1/guest3/t18_medicinal_leaf_v1_20261011/medicinal_fixed_split.npz --out LightGenV2/tasks/t18_optical_residual_ablation/reports/medicinal30_pilot_20261011
```

完成后审计首次最低NLL、PT／config／67来源／data、每参数有限非零梯度、预测ID／
标签／argmax／support／CM／NLL／macro，配对核对共同轮的初始化／order／transform。
从保存history输出真实PNG／SVG／PDF和曲线CSV，只画train／val；单种子无误差条。
原始图、缓存、预测、PT和运行目录不进Git。所有负结果保留；低train不单独证明
容量不足，是否过拟合／未收敛须看同轮train／val及验证损失趋势。
