# 完整60类Butterfly-10K深度／残差实验

最新人工要求：换数据，2／4／8／16层，每深度rho0／rho0.3配对；
验证集选模，检查并有界处理过拟合／欠拟合／未收敛，最多三GPU，使用完释放。
12层已取消；旧FishNet和Mango结果／数据／PT全部保留，不重测。
完整实验合同见[协议](../../BUTTERFLY_PROTOCOL_20261011.md)。

## 当前状态

截至北京时间2026-10-11 18:06，唯一有效队列
`butterfly60_s17_L2L4L8L16_val_3gpu_ready_20261011`，CPU父PID560528，
数据准备PID560244已成功退出。全部10013张原文件身份和解码已核验，6780分组，
train6980／val1441／test1592；缓存ID／标签／60类支持数与manifest一致，
标本／原图像素／缓存像素连接组均无跨划分，未进行test模型推理。
数据[身份收据](data_identity.json)记录实际SHA和支持数；其中3类val仅1张，
不能用总体accuracy掩盖长尾，宏平均及每类支持仍须报告。
四深度GPU smoke全部通过，18:01后自动进入base100训练：
GPU1／GPU2分别16层rho0／0.3，GPU4为8层rho0，子PID569626／569692／569761。
每卡仅本项目一个GPU子进程，实际显存1236／1086／998MiB，Torch上限3GiB另留800MiB。
当前尚无完成epoch精度，所有组test评估0；不将4个smoke完成误写成4组训练完成。
实际GPU／来源／数据核验见[启动收据](gpu_smoke_start_20261011.json)，
PID与显存为该时刻快照，后续按run的current_processes.json核对。
源提交`2ddf96fb8812b4bf72c6d342a682e9ef0646cfe9`，63个依赖实际文件与发布Git blob一致。
服务器核心13项CPU检查全过5.58秒，含四种深度相位／参数／配对初始化、真实合成短程
训练的val选模／锁后单次test／尾batch，以及准备失败不占GPU。
随后新增60类预测CPU审计检查亦通过，合计14项／5.77秒。
Windows9项通过，5项原生Torch检查跳过，不改共享环境。
GPU检查确认rho0精确一致、初始化不变、router无残差、全部相位梯度有限非零。
生产micro1的联合均值损失与逐样本加权累积，最大梯度相对L2差分别为
3.36e-8／3.78e-8／4.04e-8／4.25e-8，最小余弦均大于0.9999997，重复前向误差0；
峰值分配227／294／437／728MiB（2／4／8／16层）。此为同micro1图的累积检查，
不宣称旧batch2／single形状等价或任意权重逐像素梯度精确一致。

初始`butterfly60_s17_L2L4L8L16_val_3gpu_20261011`和
`butterfly60_s17_L2L4L8L16_val_3gpu_checked_20261011`仅等待准备后因下载失败退出，
没有GPU／训练／test，失败日志和状态保留，不复用为正式训练。
直接作者站服务器DNS失败、镜像API／resolve限流、LFS raw返回指针已逐项定位。
现有下载清单独立与作者primary API核对，普通文件raw、438个LFS文件resolved，
均核对发布文件身份；已核验文件复用，不删数据，不放宽哈希检查。

作者完整基准固定revision `354805681278db357ffdff8749a29e2ca145d32e`，
全部60类／10,013图；原始文件清单包含Git及LFS内容身份的规范SHA256为
`814eea924ddfff7dfd71916c4c2092b9682e1a8b0b2b7147bdf4df97af922c36`。
实际6780连接分组／标本，无跨类冲突，完整实际三划分支持见数据身份收据。
CC BY4许可、349个未在逐图CSV出现的文件、长尾及家系／地点依赖限制详见协议。

## 结果产物

现在没有新训练精度，不能发布模拟的train／val／test曲线或推断深度结论。
每5轮保存同一EMA未增强train精度，每轮val NLL／精度。所有方案和四深度锁定后
才对获选EMA每臂单次test，未获选方案没有test指标。
每深度共同100轮起；val持续改善共同最多200，明显过拟合或低拟合停滞最多一次
配对正则／优化恢复100轮，按两臂平均best val NLL择整个方案，保留失败／负结果。

完成后只从保存PT／history／预测作CPU审计，不重复光模型推理：

```bash
CUDA_VISIBLE_DEVICES='' /home/guest3/miniconda3/envs/xml/bin/python \
  LightGenV2/tasks/t18_optical_residual_ablation/audit_butterfly.py \
  --suite LightGenV2/tasks/t18_optical_residual_ablation/runs/simulation/butterfly60_s17_L2L4L8L16_val_3gpu_ready_20261011 \
  --data-root /DATA/DATA1/guest3/t18_butterfly10k_20261011 \
  --reports LightGenV2/tasks/t18_optical_residual_ablation/reports/butterfly60_20261011
python LightGenV2/tasks/t18_optical_residual_ablation/plot_butterfly.py
```

阶段fit结果可分别加`--fit-only`，只审计／绘制实际已完成候选的train／val，
不造test曲线。最终PNG／SVG／PDF／CSV同时报告train／val／test、macro、NLL、
best epoch与实际预算；单seed无误差条。画图需视觉核验，正式结果不混写FishNet身份。
只精确路径提交报告，经Git发布三端核对；完成后确认GPU进程退出再暂停自动跟进。
