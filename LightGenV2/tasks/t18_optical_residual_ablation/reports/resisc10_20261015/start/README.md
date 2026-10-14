# 人工要求原二十类验证表现前十类 已真实启动

用户2026-10-15明确要求十分类，并务必从之前20类选性能高的前十个。采用此前原20 input100 rho.3按首次最低验证EMA balancedNLL选E111的逐类验证召回率降序，同分按类名排序，固定以下10类；没有读test或根据新成绩再换名单。与原随机15选择不同，属于验证表现引导的开发子任务，不能声称未见验证/原20目标验收或独立泛化。

| 排名 | 原类别 | 原20所选模型验证召回 |
|---:|---|---:|
| 1 | meadow | 90.00% |
| 2 | desert | 88.57% |
| 3 | island | 85.71% |
| 4 | lake | 81.43% |
| 5 | bridge | 68.57% |
| 6 | harbor | 68.57% |
| 7 | intersection | 64.29% |
| 8 | tennis_court | 52.86% |
| 9 | wetland | 48.57% |
| 10 | thermal_power_station | 45.71% |

固定输出按原类别索引[2,3,7,8,9,10,11,16,18,19]排序。保留原cache和所有split/ID顺序，各类原560train/70val/70test元数据，合计train5600/val700/test700。只读取train/val，测试数组/推理0；原20与15全部数据/PT/历史/收据保护。新manifestSHA2ee418593db66d0ba00bfbeda6602ae12631ec5ed28afe6dd16fbec995aadf5a，引用原cacheSHA652912ee341b51dfce1a5dfc9a2b7618d6d1869bb16a8a38d1f21b7e82447431。排名CSV Git原始字节SHA82207360e23518b932fef7ad88610206bbc185eebe70506b984d92c211576671，原20scienceSHAadf635d3f4c5876f43a4d2bf3aad15f1e4e953899fd5a07eeff600505ffb4b73。

## 实际启动和预算

run resisc10_s17_mainL4_input100_rho_pair120_micro8_gpu25_20261015；约北京时间01:05:37启动CPU父585499，GPU2 L4_10_rho03子586248/GPU5 L4_10_rho0子586314。PID为历史入口，当前须核实际proc/command/CUDAUUID。首完整观察父/两子healthy、active2/pending0/fail0/test0，GPU2/5各一，A100项目0。160current/original冻结e380a206b5b032d5d6bf1a855846a73de556564f，所有原159/151及144/141/136/135/126/130源保持；报告HEAD不等运行身份。

网络保持908552相位、expert/router224/global478、四主干E/G/E/G加router五captures、原光路/OEO/CCD全478/all4/8bit jointM/input100。仅十个原坐标64x64窗口及10能量/pseudocount归一化，不改物理孔径、不缩网络或加学习head。两freshseed17同phase/order/affine/MixUp.6/LS.08/TV.005/capture.05/EMA.95/lr.002/规则，micro8/effective16、350Adam完整更新每轮、尾16。
每组最多120轮或7200拟合秒，min10/early10delta.002/LR6bad*.5floor2e-5先到停止；epoch开始时间边界检查，完成已开始轮与必要selected终点评估，非硬秒杀。最多4新增fitGPU小时，无第三candidate/retry/延长/test；仅GPU2/5各一，Torch每子2GiB+800MiB，CPUCUDA空，A100继续禁止。不改DNS/共享环境/抢杀外国进程/连实验室。

## 启动前真实检查与保留

最终CPU resisc10_cpu_native_normalization_checked_20261015真实核10标签/支持/ID/固定原窗口/初始化/10路LS和黑图均匀.1与sum1；两freshstatic及6random/dim/black native磁盘BMP逆478/CCD路由/capture5一致，14次CPUforward、概率误差0、test0。Fresh离线证明不等selectedtrainednative或实拍。
最终actualGPU2 resisc10_gpu2_numeric_normalization_checked_20261015核两rho/all10/macro16/tail16微批1对8：prob/capture0，gradrelative最大5.321e-7/cos>.99999999999985/all11finite非零、sameinitSHAf7de57b62eb33f6531128061deaa9ec2fb1bacbeea7ba02ecc50528189e60f37、optimizer0/test0。两实际producer owngates也passed，最大gradrelative5.421e-7。
最初prepare因本地CRLF和Git/Linux原始字节SHA差异在创建out/claim前拒绝，失败证据保留，修正引用GitblobSHA。首CPU/numeric准入未检查10路黑图概率归一化；训练启动前审查发现分母15伪计数遗留，改为10，并加黑图sum1断言后使用独立normalization_checked准入目录。原初准入claims/out/logs全部保留，不覆写；正式训练仅使用最终e380源码及最终准入。旧20/15源没有改动。
五真实门槛/启动收据一次SHA传输本地服务器，机器JSON及逐样本身份仅私有，不Git上传。

## 首完整轮与指定时间

约01:10只读观察：rho.3 E5 val53.2857%/macro53.2857%/bNLL1.518141，48.94秒（全train评估轮）、同Egap2.5179pp，普通E4约46.25秒；rho0 E4 val46.2857%/macro46.2857%/bNLL1.735211，56.12秒普通轮。不同epoch不算配对收益，早期不宣称最终准确率或75。当前已授权试验仍在训。
按人类时间要求，今天2026-10-15北京时间03:30同聊天统一给明确成绩/完成或未完成状态，无论是否达标都汇报，不因未75延长预算。03:30为汇报承诺，训练完成时间仅估计；普通监控静默，真正新失败/需人处理提前通知一次。每日09晨报保持。

真正saved CPU科学审查仅两fitcomplete、父/两GPU子actualrc0、160current/original与独占freshclaim/out/failure/log屏障后一次audit_resisc10.py，独立核预测ID/label/10CM/support/macroNLL/firstminNLL、PT/init/11grad/所有actualorder-affineMix/350E/尾16/LR/early/time。当前未执行完整science；对应私有auditdispatcher inspect真实readyfalse/outfailureclaim0，不重复旧15/20审查。
strictval65/macro60/everyrecall20/nozero/末20zero/gap15/all4/inputvariation保持；sciencepassed不等eligible，无eligible有界失败。native/独立2或3/formal6/真实science/durableonce-test仍需十类合同对应新证明，不能套旧入口或因65直接test，当前test未授权。[独立四图工具准入](../plot_preparation/README.md)。
