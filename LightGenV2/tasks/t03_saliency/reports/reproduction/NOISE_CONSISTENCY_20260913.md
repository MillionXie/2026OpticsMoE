# 双噪声预测一致性（训练方法对照）

## 先诊断，再决定

固定87ad权重、seed20260913，随机128张训练图，每张3次原噪声前向与1次普通eval参考，
全部CPU无梯度、无参数更新、不涉及测试集。记录在`runs/smoke/noise_variance_20260913/report.json`。
训练子集均值：clean CC .88511204、noisy CC .88442175，平均差仅.00069029；
因此**不支持把当前精度瓶颈主要归因于巨大clean/noisy平均偏差**。
单图3次噪声CC标准差均值.004936，中位.004311、p95约.012716、最大.018952。
前两次预测之间CC均值.996388，JS均值.00053994；噪声总体不大，但有少量波动较高的图片。
三次预测平均的CC .885470只用于诊断；不做推理集成、不拿训练子集数字作为最终性能。

## 具体改变

借鉴[R-Drop，NeurIPS 2021](https://arxiv.org/abs/2106.14448)及[作者实现](https://github.com/dropreg/R-Drop)
的双随机前向分布一致性训练思路，适配为光学/CCD噪声，而非新增普通dropout或论文网络。
这不是SALICON上的原论文复现，不保证能带来提升。

每批同一组图像做两个独立的原训练噪声前向，得到空间概率密度p、q；
两次都保留原幅度/相位未调制20%–30%、相对相位随机性及其他噪声，不临时清零DC。
两次均计算原GT多指标、原教师空间CC KD2、原路由均衡/DC/CCD工作点正则；
取两个完整loss的平均，再加`2 * (KL(p||q)+KL(q||p))/2`。
两个预测都有梯度，不detach任一目标，不与GT或教师做推理混合。
每幅图在H×W位置上归一化为分布，KL按位置求和、按图像求平均，不按像素数稀释系数。

SAM仍只扰动既有电子参数。SAM的两次反向各含上述两次噪声前向：
共4次forward、2次backward、1次AdamW/EMA更新；噪声1/2在SAM第二轮精确重放，
不把权重扰动效果和另一组随机噪声混淆。原相位梯度照常更新，前端冻结。
这比原SAM增加约一倍前向训练工作与部分激活显存，**不是相同计算预算对照**；
独立样本仍10000，没有额外标签。推理仍一个正常模型、单次预测，无额外参数或传播阶段。

配置`moe_alpha40_noise_consistency_20260913.yaml`继承extra_control，固定87ad初始化，
batch32、原学习率、EMA.995、SAM.05，20轮预算，第16轮进入低LR精修。
光router Top2、alpha>=.4、同尺度融合、478ROI/224专家/17微米/10cm、85412参数头全部不变。
保持10000 train/5000 public-test、无额外val、第1/5/末轮评估选best；
仍披露反复public-test选模偏差，目标.87未达到前不宣称完成。
标准测试关闭随机光扰动；不能将标准CC当成实测硬件成绩。

```bash
python -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --phase all --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_noise_consistency_20260913.yaml
```

先CPU回归与真实输入更新检查、GitHub同步，再用一张空闲GPU启动。
训练记录`train_noise_consistency_kl`；只best/last，持续回落时保存实际轮数并停止。
既有GSAM失败组不恢复；不叠加GSAM、ASAM、其他辅助头或数据增强。
