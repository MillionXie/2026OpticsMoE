# 原完整网络输入细节三组：已获批准并实际启动

用户在解释“输入50/100”与rho含义后明确要求“你赶紧试一试吧，我的时间很紧迫”。本次执行具体三组提案；原网络尺寸不变。聚合报告及方案发布到MillionXie/2026OpticsMoE main已有明确许可，原始检查JSON与逐样本数据继续留在本地/服务器。

## 已通过的实际检查

151文件来源`d5158a53e4fc1a5a42fff54b25b8c06353f2c1e5`，当前与原始Gitblob一致。原144/141/136/135/126/130依赖不修改。
CPU无训练数据/测试数组：原50编码在有/无仿射条件下逐元素复现一致；100编码独立配方一致。三配置各随机/暗/黑输入共9项真实CPU原生BMP磁盘回放，518/478光场逆变换与五capture顺序、CCD/routing匹配，概率最大误差0，黑输入全零；共18次模型前向。这是fresh证明，尚不是选中训练权重/现场实拍验收。

实际GPU6每配置20类及尾14、micro1对micro8概率/capture误差0；全部11组phase梯度有限且非零，最大相对L2误差`6.35e-7`，最小cosine大于`.99999999999979`。三配置同初始phase SHA`f7de57b62eb33f6531128061deaa9ec2fb1bacbeea7ba02ecc50528189e60f37`，数值检查没有optimizer更新/test。

## 真实启动与范围

run `runs/simulation/resisc20_s17_mainL4_input50input100_rho_controls120_micro8_gpu256_20261014`，北京时间2026-10-14 17:32:02 CPU父3940020实际启动。17:32:45只读核实父CUDA不可见、全部三子command/物理GPU UUID匹配，pending0/失败0/test0。各子正在生产准入与训练初始化；此快照尚无完整epoch，不宣称新输入精度改善。

17:34:39再核实三生产门槛全部passed：20类/尾14概率及capture误差0，11组phase梯度有限非零，最大相对梯度误差`6.34e-7`。三个真实子都仍在指定卡运行；当前尚无首完整epoch，不把门槛通过当作精度验收。

17:37:37已保存真实完整epoch：A/B各E3，C为E1；三父子仍健康、失败0/test0。A50rho.3 E3 val26.8763%/macro26.8592%/bNLL2.435836/61.24秒；B100rho.3 E3 val26.0186%/macro26.0031%/bNLL2.446265/60.23秒；C100rho0 E1 val12.7949%/macro12.7961%/bNLL2.862408/159.20秒（含首轮全train评估）。A/B普通轮约1分钟；C卡首评估轮更慢，不能将A/B速度当三组统一估计。早期且C不同轮，不判断最终输入或残差收益，也不承诺120轮都在2小时内。

| 配置 | 图片每通道采样 | 残差rho | 物理GPU | 启动子PID（历史） |
|---|---:|---:|---:|---:|
| 新同次控制 | 50×50 | .3 | 6 | 3940641 |
| 保留更多输入细节 | 100×100 | .3 | 2 | 3940707 |
| 新输入无残差配对 | 100×100 | 0 | 5 | 3940779 |

每组最多120完整epoch或7200拟合秒，既有early-stop先到则停止；三组总拟合上限6 GPU小时。时间在完整epoch开始前检查，已开始的一轮及必需端点评估完成后退出，不声称精确秒级硬杀。无第四候选/自动retry/追加预算/test。

网络908552phase、专家/router224/global478、主干4层+额外router、传播/孔径/routing/OEO/CCD/8bit及20类固定读出保持。只改变公共150→100/同theta仿射之后的通道采样50→100，四tile拼接后均送入同224输入；不是把网络改成50/100。新输入细节上限仍为公共100预处理，没有学习编码器/电子学习分类头。

每组完整11198train/1399val、seed17/same order/affine/MixUp.6、LS.08/TV.005/capture.05/EMA.95/LR规则、effective batch16/micro8/700 Adam updates每轮/尾14保持，fresh从E1开始。原120混合执行历史仅作参考。

## 完成后准入

所有三组fit真实完成且父/全部GPU子退出，才执行一次新input-aware保存科学CPU审核；exclusive claim/out/failure保护，不重复已有旧审核。按first minimum valEMA balancedNLL选EMA，不能按最高accuracy另挑。
原val65/macro60/eachrecall20/gap15/零类/路由/输入变化和实际selectedtrainedBMP门槛不放宽。无eligible则报告本次有界失败。对应真正选中权重native、独立2/3和正式六组尚未完成；最终test75目标不保证，test不选模。
