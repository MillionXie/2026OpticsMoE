# 小批次专家均衡估计对照

## 动机与边界

batch8 SAM首轮独立CC=.86239605，但第5轮回落到.86172517；目标.87未达到。
其epoch1 EMA（SHA e4930c1264442d7056fd0451385cf8fb301e83dd580200e15c6b7a7fe934725c）
在随机128张**训练图**上的专家次数60/71/53/72，有效专家数3.93988。
同一组输出拆分batch8/32/128，平均硬均衡损失为.1347656/.0615234/.0152588，
importance损失为.1042316/.0353219/.0075005。这表明单batch惩罚混入抽样方差，
但不证明它就是性能下降的原因。CPU诊断及样本身份见原batch8 run的
`candidate_train128_balance_diagnostic.json`；不是完整测试集均衡结论。

设每图概率p_i、硬Top2份额h_i=selected_i/2，均有4维、和为1。
旧损失含批次均值乘积，包括i=j的自身项。
本对照定义 C(x,y)=4·Σ(i≠j)<x_i,y_j>/[B(B−1)]，分别采用：

- soft-load：C(p,h)，替换原4<mean(p),mean(h)>；保留原capture效率项。
- importance：C(p,p)−1。
- hard-load：C(h_ST,h_ST)−1；h_ST=h+p−detach(p)，沿用原直通梯度约定。

不同样本独立抽样时，它估计总体均值的乘积，而非单batch自身方差。
有限样本值可以为负，不做截零；它不是概率、不是测试指标。
同一对专家完全垄断时hard仍为1，因此不是取消均衡。
这是本项目的数学估计器试验，不宣称某篇论文的完整算法复现。
它可能增加梯度方差，是否更好必须实测；一次128图诊断不是泛化保证。

## 严格配对

配置`configs/moe_alpha40_sam_batch8_crosssample_20260913.yaml`继承原batch8。
仅`loss.router_balance_estimator`从batch改为cross_sample。
均从87ad正式权重出发：batch8、20轮、LR、EMA、SAM .05、GT+KD2、均衡系数、
Qwen冻结前端、85412参数读出头、alpha>=.4、Top2、DC20–30%及全部光路均不变。
无新增可训练参数；仅训练时替换损失估计，eval仍使用历史损失，预测和state_dict不变。
不把正常推理改成去光，不训练独立纯电模型，不更换baseline。

```bash
python -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --phase all --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_sam_batch8_crosssample_20260913.yaml
```

原10,000 train / 5,000 public-test身份不变；第1/5/末轮测试、按public-test选best，
不声称独立盲测泛化。只保留best/last。当前为待测试/启动配置，不表示已有性能。
启动前先通过CPU回归及真实输入更新审计，push GitHub后才运行正式训练。
同一助手合计不超过两张GPU，任何已结束父子进程需检查释放。

## 验证与启动

源码`2dc9fd5f3c2c502a9b1f068f0d80cd0ece9da4f1`通过252项CPU测试（45.52秒、13条既有警告），
已push至GitHub `experiment/salicon-crosssample-20260913`。
真实8图CPU单步SAM：冻结前端3933184参数逐值未变、无梯度；native Transformer调用0；
eval切换估计器标志预测逐值相同；读出头85412参数。六张相位都有有限非零梯度及更新，
router原始参数RMS更新3.1325e-6，其余约4.6474e-5至4.7835e-5；alpha .43072176/.44106695。
审计`runs/smoke/crosssample_20260913/report.json`中的8图训练CC不是测试性能。

原batch8第5/10轮下降后已在UTC2026-09-12 19:41停止PID/PGID561115及全部五个子进程，
GPU3已释放；保留best第1轮SHA e4930c12…34725c、last第11轮
SHA `224db722d59d7a4d046348701ca0195702a7513e703bc26c217e4d9c49399611`，CPU重载检查有限值。
不是完成20轮。原run的`manual_stop_report.json`及`stopped_checkpoint_integrity.json`记录具体身份。

本配对于UTC2026-09-12 19:42:21启动PID/PGID601787，固定worktree
`.worktrees/t03_baseline50_20260912`（运行中禁止checkout），GPU3 UUID
`GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd`，当前总计一张自有卡。
run为`runs/simulation/moe_alpha40_sam_batch8_crosssample_20260913_seed42`，
完整命令、配置SHA与来源在`launch_record.json`，日志`console.log`。
此处为已启动、待完整测试状态，不是目标达成或新最佳。

首轮完整5000图测试CC=.8624925288200378，live训练记录alpha=.43065083/.44102651；
这不是独立复评数值，也不把live alpha当作EMA权重审计。需继续看第5/10轮趋势，
不得因辅助loss值更低（估计方式改变）就直接与旧组的总loss比较优劣。

第5轮回落至约.8619，尚未显示持续收益，继续观察第10轮。
`midrun_state_audit.json`读取的仍为epoch1 EMA，SHA
`036bc8caedcde4d6dabce276b1a2e4af15d960d88620098b19e1c198849b8bfe`。
alpha=.43068659/.44104557，同规格Qwen decoder核对通过，85412头参数、479364光参数、
六张相位尺寸和原光路合同未变。router物理相位较87ad的RMS变化7.95e-5 rad，
其余约.001463至.001561 rad；这不是完全未训练，但状态检查不能代替完整独立测试。

## 停止与独立复评（2026-09-13）

第5/10轮CC分别.86188473/.86173971，低于首轮；UTC2026-09-12 20:17停止PID601787
及全部五个子进程，GPU3释放。实际last第10轮，不是完成20轮。
best仍为上述036bc8ca权重；last SHA
`4cfa2ac828452e7bdb3e81f8eeea0091faf8e2c4e4509684ebdd4d19aea2eaed`。
两者CPU重载core/head有限值通过，证据`stopped_checkpoint_integrity.json`。

固定best、同5000张、batch48独立float64 CC **.8624925081777596**，
legacy CC .8624925288200378；差2.06e-8。KLD .1141688072、SIM .8243188085、
NSS .9647891521、AUC .7700042558、peak-normalized map MAE .0816669730。
`candidate_recheck/reproduction.json`、逐图CSV及log保存原始证据；源码2dc9fd5f。
test IDs SHA `625dec6bc15b2d737d39bc252cfa0c354de217fec0266dcda568913f4a3496d0`。
复评PID632783与同组子进程全部退出，GPU3释放。
这是微小改进，未达到.87；无独立盲测，也尚未重做该权重的完整去光/路由审计，
不能挪用e493或87ad的去光数字。

随后同权重去光也完成：完整5000图独立CC **.8422947020969439**，相对正常.8624925081777596
绝对下降.020197806080815672、相对下降2.341794996%。不重训电子模型，旁路router和两个光分支，
融合返回原E（系数1），不伪造CCD。checkpoint与test IDs SHA均与正常复评相同。
`candidate_remove_optical/reproduction.json`和`candidate_ablation_summary.json`保存完整证据。
PID647248及同组子进程已全部退出、GPU3释放，之后才启动完整router/alpha与首16图可视化审计PID648863。
此处相对性能变化不是光分支的严格因果贡献比例，也不等于融合alpha。

完整5000图router/alpha审计及首16张固定顺序样例也已完成，PID648863退出、无残留同组进程。
alpha .4306865931/.4410455823；Top2计数2346/2629/2312/2713（共10000次选择），
使用率23.46%/26.29%/23.12%/27.13%，有效专家数3.980722/4，无未使用专家。
`candidate_selected_evaluation/selected_checkpoint_test_evaluation.json`记录相位相对87ad变化，
`best_visualization/best_phase_overview.png`及16组`saliency_examples`用于可视化，
是圆周相位残差图与归一化展示，不是SLM灰度BMP，也不从展示截图计算指标。
该报告SHA再次确认036bc8ca未变化；指标与独立重评对应同一权重。

## 本地证据镜像（2026-09-13）

服务器与本地均保留任务内同名`runs/simulation/moe_alpha40_sam_batch8_crosssample_20260913_seed42`。
本地根目录为`C:\Users\Xml12\OneDrive\2026OpticsMoE`。
已传输并逐文件核对57个原文件（57,725,262字节），另附`transfer_manifest_20260913.json`。
包括best/last、配置/训练记录、独立正常与去光复评、完整路由审计、相位图和首16张固定顺序样例。
清单SHA256：`9347dbca9f01f3de856373fff84ff09804507bd5fe1d70c9b5f83c5b46e0896a`；
传输归档33,942,097字节，SHA256：`8f284177bc5aef78b8ca9abe2f5e72f234328443c66a3fee2941c97a9f4d56a8`。
归档路径和文件类型先检查，再解压到此前不存在的同名目录；未覆盖旧run。
这是候选证据镜像，**不是硬件就绪ZIP**；本任务仍需任务专用实验室打包/设备闭环验证后才能交付实测。
