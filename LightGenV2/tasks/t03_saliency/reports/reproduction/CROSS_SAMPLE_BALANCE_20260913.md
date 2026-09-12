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
