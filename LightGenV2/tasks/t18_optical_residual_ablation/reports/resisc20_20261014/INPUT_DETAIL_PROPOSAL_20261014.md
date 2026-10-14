# 已获人工批准：保留原网络的输入细节对照

2026-10-14，用户在下述普通话解释后明确要求“你赶紧试一试吧，我的时间很紧迫”。具体三组120轮/每组7200拟合秒范围已批准，实际入口与CPU/GPU检查和启动见[input_detail_start](input_detail_start/README.md)。不是无限搜索授权。

## 用普通话解释这个方案

“输入50”说的是照片细节：红、绿、蓝每个通道先缩到50x50，再拼接并放大送入光学网络；不是把专家改成50x50，也不是只有50张训练图。
“输入100”是每通道保留100x100细节，减少这一步压缩；专家始终224x224、全局层478x478，全部相位参数和层数保持。
“rho=.3”是相干残差系数：70%相位调制光场振幅加30%未调制参考光场振幅；不是30%光功率。“rho=0”是无残差对照。

三组分别为：原图片细节+残差、更清楚的图片输入+残差、更清楚的图片输入+无残差。
前两组判断提高输入细节是否有用；后两组判断同一新输入下残差是否有用。不预设更清楚就一定更准。
用户已允许聚合报告和本方案的指定GitHub发布；最初不理解训练问题，解释后另明确批准尝试。

完成120后没有eligible。新增CPU320个train缓存样本显示原150->100->50通道重采样的相邻像素差平方能量保留均值8.30%、相对去均值像素方差的重建误差24.75%；这只是描述性细节度量，不证明分类因果或准确率收益。
网络全部908552相位参数、专家/router224、global478、层数/路由/传播/CCD/OEO/8bit/BMP及20输出窗口保持，避免永久缩相位网格。

具体建议最多3个fresh L4：

| 对照 | rho | RGB每通道tile | 用途 |
|---|---:|---:|---|
| A | .3 | 50 | 全新micro8从第1轮的同次控制 |
| B | .3 | 100 | 与A只改变输入tile细节 |
| C | 0 | 100 | 与B构成同新输入的残差配对 |

保留原150->bicubicAA100/clamp与同theta affine100；只将之后的bilinearAA50改为100（100时保持100），RGB/R/G/B/mean拼200再bilinear224/RMS.25。
原A为拼100再224。物理孔径/实际SLM振幅224与相位224/478不变；没有CNN/Linear/预训练/电子学习编码或读出。
新B/C不是112个原始独立采样，信息上限仍为共同100预处理；不偷偷改变增强theta/LR/MixUp等来叠加原因。
A必须fresh，因为旧历史E1-21micro1/E22后micro8；不拿旧120当严格同次输入单因素控制。三组seed17初始phase相同，same order/affine/MixUp.6，LS.08/capture.05/TV.005/EMA.95/lr规则/macrobatch16/micro8/700updates/tail14不变。
原固定RESISC20 train11198/val1399/test1403及所有类别/cache/split不改，不更易类、不读取test或test选方案。

每组最多120轮或7200拟合秒，先完整epoch预算/既有early-stop到即停；三组共最多6拟合GPU小时（完成epoch/selected端点评估粒度披露），只物理GPU2/5/6各最多一个新子，仍global6/percard2/Torch2GiB+800MiB，CPU父CUDA不可见。
当前普通约1min一轮加周期train评估，120未必在2小时内；没有第4候选/自动retry/延长/期限承诺。原75目标不保证。
新151文件运行闭包已实现并冻结，CPU编码/native和GPU数值门槛实际通过，三组queue与训练进程已启动；最新状态见独立启动报告。

已按批准范围独立实现输入-aware profile/入口、数值门槛与CPU审查/对应出口依赖闭包；禁止改144冻结旧源或换旧权重推理来假称修复。
真实CPU编码/native磁盘/BMP合同、actualGPU20class/尾14/B1vs8/prob<=1e-6/gradrel<=.02/cos>=.999/all11有限非零、samepair init/order/aug门槛必须先通过再fit；原freshgate输入和部署图不能盲套新B/C编码。
完成全部真实exit后独占science CPU核所有actualepochs/源/data/PT/config/IDlabels/CMsupportmacroNLL，按firstminimumvalEMA balancedNLL选，raw/last不test，不按最高accuracy反选。
strictval65/macro60/everyclass20/trainvalnozero/last20zero<=half/gap15/all4/inputvariation/perclassrouting及actualselectedtrainednative不放宽；无eligible如实有界失败，无自动继续。
只有成功后对应新input合同与selectedtrainednative、独立2/3末E几何和逐类验证、正式六共同pairprofile/init/order/aug和真全6CPU科学审核/actualexit/durable锁后once-test，不能用合成flags或旧conditionalqueue替代。

本次比较完成后按实际结果判断输入方案，不因开始对照而直接采用新输入作为最终配置；不自动追加候选或预算。
