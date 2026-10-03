# 暂定 CC 0.86205：结构、参数与性能来源核查

## 当前优先入口（2026-09-13）

下方87ad和e493为历史权重审计。当前最高已独立复评候选是
[cross-sample第1轮036bc8ca](CROSS_SAMPLE_BALANCE_20260913.md)：CC=.86249251，
同权重去光=.84229470，alpha=.43068659/.44104558，头85412、光参数479364，
完整正常/去光/专家审计与可视化已逐文件校验并镜像本地。
后续MixUp首轮周期测试.86257908仅属在训结果，未独立复评，不与该正式候选混用。
本页下文“新在跑cross-sample”等为当时记录，以本段及对应run最终记录为准。

## 补充候选：batch8，独立CC .86239605（2026-09-13）

`runs/simulation/moe_alpha40_sam_batch8_20260913_seed42`已停止并保留best第1轮和last第11轮；
不是完成20轮，也没有达到.87。本轮没有新增推理结构，改变的是训练批次、学习率及EMA时间尺度。
best SHA e4930c1264442d7056fd0451385cf8fb301e83dd580200e15c6b7a7fe934725c；
同规格头85412、光参数479364，alpha .43068656/.44104564。
完整5000图独立CC=.86239605；同权重去光=.84230251、相对下降2.32997%；
专家份额23.43/26.29/23.13/27.15%。旧87ad记录继续保留，不把它的去光数字挪用到新权重。
新在跑cross-sample组的首轮.86249253尚未独立复评，不能混入本节。

本候选64个文件（不含清单自身）、57,743,959字节已镜像到**本地同名run目录**，
逐项尺寸/SHA及文件数核验通过，清单`transfer_manifest_20260913.json`自身SHA：
`64a34e2c7f80f9628bed90fc94772ea4e58f8e80acca90bd3ed3d523121f6414`。
权重、配置、全部逐图CC、去光/均衡报告、相位图及16个连续测试样例均保留；
没有原始完整数据/Qwen资产/硬件适配打包，因此**不是实验室就绪ZIP**。

直接看`candidate_summary.json`，图片在`candidate_selected_evaluation/best_visualization/`。
`best_phase_overview.png`显示相对圆周均值的相位，不是SLM可加载BMP；router棋盘纹和
后两专家底部较强相位结构可见，但不能仅靠外观判断路由坍缩，须结合上述完整路由分布。
`saliency_examples/sample_000.png`等为固定测试顺序前16图，并非挑高CC图。
这些图的GT/prediction**各除以自己的最大值显示**；图中Absolute error也基于这两个峰值归一化图。
已核对`SaliencyAccumulator.update`：项目报告的MAE同样是逐图峰值归一化后的平均绝对误差，
不是sum=1概率图的原始MAE；CC/KLD/SIM仍分别按其原有概率图指标实现计算。
不要从带色标、缩放压缩后的截图重算正式指标；使用独立复评CSV/JSON及指标源码。
已人工查看相位总览及首个样例，不能将这两张图片检查扩大为完整硬件部署验收。

## 本地交付证据镜像（2026-09-13）

以下两套已结束的run已从服务器同步到本地同名的`LightGenV2/tasks/t03_saliency/runs/simulation/`：

|run|已核验文件数（不含传输清单自身）|数据字节数|
|---|---:|---:|
|`moe_alpha40_sam_spatialcc_kd2_seed42`|36|40097461|
|`qwen_aligned_head_50_20260912_seed42`|23|19966812|

本地仓库根为`C:/Users/Xml12/OneDrive/2026OpticsMoE`。先核对传输清单本身SHA256，
再逐一核对全部59个文件的大小/SHA及无多余文件，均通过；两份best仍是上述87ad与baseline689d权重。
每套的`transfer_manifest_20260913.json`记录源路径和逐文件哈希，其自身SHA依次为：

- 光模型：`6568dbeb7e5c77a835c7dadbb6395d1c61366fd0dd6aa0789054c747871946bd`
- baseline：`d1b72ff81e679a7d964405f2954086c591a7ad3d03b039cb75de81d2f66b23d0`

这只是约60MB的权重/配置/日志/可视化/复评证据镜像，不包含原始SALICON数据或完整Qwen前端资产，
也不是实验室硬件就绪ZIP，不说明.87已达标。数据仍由Git忽略，源码只通过Git同步。
光模型`best_visualization/best_phase_overview.png`是相对相位可视化，不能作为SLM BMP直接加载。

## 2026-09-13：可执行的权重结构检查

`audit_checkpoint.py` 只用CPU，将候选与上述正式87ad参考权重逐项核对：完整core/head张量名、
形状和dtype、非有限值、同规格85412参数头、alpha>=.4、479364光学参数与六张相位尺寸。
如提供Qwen checkpoint，还会直接比对其decoder张量规格。记录被实际读取字节的SHA256。

```bash
CUDA_VISIBLE_DEVICES='' python -m LightGenV2.tasks.t03_saliency.audit_checkpoint \
  --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_reliable_teacher50.yaml \
  --checkpoint LightGenV2/tasks/t03_saliency/runs/simulation/moe_alpha40_reliable_teacher50_20260912_seed42/best_checkpoint.pt \
  --reference LightGenV2/tasks/t03_saliency/runs/simulation/moe_alpha40_sam_spatialcc_kd2_seed42/best_checkpoint.pt \
  --qwen-checkpoint LightGenV2/tasks/t03_saliency/runs/simulation/qwen_aligned_head_50_20260912_seed42/best_checkpoint.pt \
  --output LightGenV2/tasks/t03_saliency/runs/simulation/moe_alpha40_reliable_teacher50_20260912_seed42/final_state_audit.json
```

最后打包前、训练结束后运行这条命令。已有输出时拒绝覆盖；如需中途检查，用另一个明确epoch的输出文件名。
这是**张量/配置结构检查，不是完整合规或准确率验收**。相同state形状不证明计算图相同，
仍需要下面的源码/前向hook核查、完整5000张独立复评与硬件测试；不把输出中的state_checks_passed
解释成CC达到0.87或实际光路已验证。以下原始审计历史保留。

检查器源码4f96c36e通过232项CPU测试，已推送GitHub。首次真实中途检查为可靠教师试验epoch5 EMA：
checkpoint SHA `37303905f441e907a385f0e716ad126c705db02ef04ce81f3aa6e7a4ddfd3018`，
alpha .430724/.441071，core/head规格与87ad一致，头与本轮Qwen实际权重的decoder规格一致。
该报告在候选run的 `state_audit_midrun_20260913.json`，只是当时读取的权重快照，不替代最终检查。

同日CPU只读alpha诊断：固定seed20260913随机抽取64张train图，未用test、未保存修改后的模型。
原alpha约.43072/.44107时CC=.886806，固定两级.41为.887164，仅首级.4001为.887198；
两级.5降为.879261。这里的CC是64张训练子集诊断，不是新测试性能，不据此手动修改正式alpha。
样本ID顺序SHA256 `bd82ee468ebc1c6dea6550143e3be03df3481955afbcc20b1d0ff2068700ee67`。
该小样本结果不支持“仅调alpha即可带来约.008提升”，也不证明全量最优alpha已找到。

## 状态与权重身份

2026-09-10用户要求暂停优化、先核查结构。没有达到原0.88目标；不再自动启动后续训练。
暂定候选为`runs/simulation/moe_alpha40_sam_spatialcc_kd2_seed42/best_checkpoint.pt`，
已完成50轮，选择epoch5 EMA；SHA256：
`87ad4db51e3f58f9a41d6df09092439e5a09e81e93f88a3bfe2d3d008fafb29a`。
完整5000张CC=0.8620496600，独立float64复评约0.86204960。不是最新未完成试验的成绩。

本次只读CPU核查源码`1caec282f68444499796a073f473740418b8ce77`，加载上述实际字节并核对SHA，
严格加载`core`及`saliency_head`。环境torch2.6.0+cu124，CUDA_VISIBLE_DEVICES为空，OMP/MKL各2。
模型参数用去重的`named_parameters()`统计；以一张RGB224合成图作前向hook/梯度连通性检查，
不是重新评估性能。合成图不能证明模型精度；精度依据下面的完整数据复评文件。

已按UID、命令、cwd和PGID确认并终止自己的两组：

- 深监督：PID/PGID2048201，子进程2053765/2053766/2053932/2056317/2056503。
  best为epoch0，last为epoch13；停止于下一轮中，不是完成40轮。
  last SHA=`b1ee4c77d5c095555e3f10bb54b46bbe51db00a423f1fd2c16e5f7549243fd83f`。
- 实际裁剪教师组：PID/PGID2199357，子进程2205780/2205781/2205976/2208240/2208414。
  best为epoch0，last为epoch1；停止于第2轮中，不是完成40轮。
  last SHA=`6016088becec98d5e16f9a30e81f874019a259430544fa5521d4d76cea5fec98`。
- 两组best均SHA=`bf93ba1f31b00b5dc3fe33b70f81cfd6459ddd583eedfa76f7d016c5ac1cf529`；
  它们保留初始化，封装元数据不同，不用其文件身份替代正式87ad候选。
- 停止后上述全部PID消失，nvidia-smi中GPU0/3不再有本助手进程；best/last均可读取。
  没有删除任何run、权重、缓存或其他人的进程。proxy裁剪对照没有启动。

## 预测主路径

这是SALICON **Vision-only** 单图显著性模型，没有文本/语言分支。不是运行完整Qwen后再加光学头。

```text
RGB图224×224
  → 冻结Qwen patch embedding + 位置embedding
  → 196个1024维token（14×14，原Qwen block-major排列）
  → Linear1024→192 + LayerNorm
  → 第1级：电子残差E1 ║ 光router Top2 → 四专家选二 → 光特征O1
  → E1/O1同尺度凸融合F1，仍196×192
  → 第2级：电子残差E2 ║ F1重新编码 → 全局光相位传播 → O2
  → E2/O2同尺度凸融合F2 + LayerNorm
  → 恢复二维排列[B,192,14,14]
  → 原85,412参数渐进解码头
  → [B,1,224,224] logits → 空间softmax → 显著性概率密度
```

两级电子残差均为原192宽度的局部二维混合与192→384→192通道MLP；
当前仅在每级MLP内部增加384通道3×3 depthwise卷积，不是attention或新CNN主干。
解码头192→128，随后14→28→56→112→224逐级上采样、通道128→96→64→32→16，最后1通道。
其中卷积/残差单元原本就存在；不能因为名字叫block就把它说成Transformer。

光路保持17µm、10cm、478×478总有效区；router相位224×224，四专家各224×224，
全局相位478×478。一次router测量、两次光特征测量；不是六层光或新增多次预测集成。
CCD采用`mean_only`：强度除以其空间均值，没有log/gamma。
训练特征传播保持20%–30%随机相干未调制分量；现有标准eval关闭随机光学扰动，
不能将此CC称为已经实测或在20%–30%实测漏光下验证的成绩；也不泛称router用同一DC实现。

## 电子参数不能只报读出头

以下是实际参与预测主路径的可训练电子参数，包含光学输入/输出的电子接口。

|部分|参数数|
|---|---:|
|输入Linear1024→192 + LN|197,184|
|两个电子残差，共计（含新增DW卷积）|382,084|
|光学振幅编码Linear192→224 + LN|43,680|
|两级CCD→192维读出映射|86,400|
|输出LN及两个alpha标量|386|
|显著性解码头|85,412|
|有效可训练电子合计|**795,146**|

优化器登记电子参数795,147，其中另有1个`hybrid.residual_logit`，只连接到下述被丢弃的兼容输出，
本次对最终logits求导为None；因此不能把它算作有效预测支路。
另有冻结、实际用于预测的Qwen前端3,933,184参数：patch embedding1,573,888，位置embedding2,359,296。
二者合计有效电子参数4,728,330（冻结+可训练），不包括光学相位或废弃兼容尾部。

光学可训练参数单独计：router50,176，四专家200,704，全局228,484，总479,364。
四专家每次激活两个，不代表只保存/训练两个专家的参数。
新同规格Qwen baseline只训练adapter+decoder=282,596，而光电有效可训练电子为795,146；
因此**不是总可训练电子参数匹配的baseline**。它匹配的是适配器/头规格，Qwen还使用完整冻结视觉主干。

本次直接比较最初`moe_router_scale_dc20_seed42`与正式best的core权重：
没有删除key，没有已有tensor改变shape，唯一新增是：

```text
hybrid.blocks.0.mlp.1.0.conv.weight  [384,1,3,3] = 3456
hybrid.blocks.1.mlp.1.0.conv.weight  [384,1,3,3] = 3456
```

初始及当前head均85,412参数。因此从0.82906到本候选，新增预测电子参数仅6,912，
不是引入VGG、完整Transformer、大读出头或额外教师推理分支。不能反过来把原有电子部分隐去不报。

## 本次发现的真实冗余：兼容尾部仍计算，但不影响预测

CPU前向hook实际发现：

- 原生24层Qwen Vision Transformer调用**0次**；deepstack merger没有调用。
- `core.hybrid.output_adapter`192→1024（冻结197,632参数）仍计算，用于填充旧Qwen接口返回值。
- Qwen `visual.merger`（冻结25,174,016参数）仍执行其LN/两个Linear。
- 最终显著性头读取保存的192维latent，不读取上述兼容输出或merger输出；
  因而这些模块不构成额外提高CC的电子预测支路，却消耗真实时间/显存。
- 加上无效residual_logit共25,371,649参数的兼容计算需单独披露。
  原生TF302,309,376参数及deepstack merger75,540,480参数也仍被旧包装器/快照保留，
  但未执行；“保留/加载”“实际执行”“影响预测”三个概念必须区分。

结论：**任务模型没有靠增加大电子网络取得这次提升，但现有实现不是干净的最小部署实现。**
本次按用户要求只核查、暂停训练，没有移除该尾部，没有宣称已经节省运行时间。
若后续精简，应在固定权重上做输出等价和完整测试核验，再重新测时延/功耗，不能靠参数表推算提速。

代码依据：T03 `modeling.py`、`BalancedVisionCore.forward_groups`、
`experiments/vision2_hybrid_dense/modeling.py`及继承的SALICON/LSP forward。

## Alpha与同权重去光

实际best中：alpha1=**0.4307218194**，alpha2=**0.4410670400**，均大于0.4。
当前范围是[0.4,1.0]，即`alpha=0.4+0.6*sigmoid(raw)`；不是某些历史版本的上限0.95。
融合按每图有效token/channel做RMS同尺度处理：

```text
En = E / rms(E); On = O / rms(O)
M  = (1-alpha)*En + alpha*On
F  = rms(E) * M/rms(M)
```

这些RMS在当前训练实现中detach，不增加可学习参数。最终共同缩放保护下一层数值尺度。
alpha是归一化特征的混合系数，**不是“光贡献了43%的精度/信息”**。

|完整5000张，同一87ad权重|CC|
|---|---:|
|正常光电，独立float64|0.8620495784|
|去光，无重训|0.8411720440|
|绝对下降|0.0208775344|
|相对正常CC下降|2.42185%|

去光是同时旁路光router/专家/全局传播，每级返回E（系数1），不是保留(1-alpha)衰减；
没有单独训练纯电子模型。它证明该权重依赖光计算，不是可加性的光电性能占比分解。
两份`reproduction.json`已重新读取核对相同checkpoint SHA、5000样本及相同test IDs SHA。
完整数据及命令见[去光复评](OPTICAL_ABLATION.md)。
专家使用23.49%/26.25%/23.16%/27.10%，没有明显全局坍缩；不能仅凭这一统计推断所有条件都均衡。

## 为什么从最初提升到现在

下表是已完成并读取过结果文件的里程碑，不是各项可独立相加的因果贡献。
来源均在本任务`runs/simulation/<run>/selected_checkpoint_test_evaluation.json`。

|阶段/run|CC|主要变化与解释边界|
|---|---:|---|
|moe_router_scale_dc20_seed42|0.82905983|初始光router方案|
|moe_staged_alpha_ge040_seed42|0.85134034|CCD均值归一化、alpha≥.4后的分阶段适应等历史联合改动|
|moe_alpha40_refine_weakaug_seed42|0.85468765|修正实际LR配置、光相位/电子分组调速、温和增强与分阶段精修|
|moe_alpha40_generalize_kd060_seed42|0.85812016|EMA、权重衰减和训练图教师密度蒸馏；严格输入/目标对齐|
|moe_alpha40_viewreg_cffn_kd2_seed42|0.85953132|早期弱增强、仅6912参数空间DW、较强早期KD的组合|
|moe_alpha40_sam005_seed42|0.86133209|电子子空间SAM，改变训练更新、推理无新增模块|
|moe_alpha40_sam_spatialcc_seed42|0.86172946|教师监督从逐值KL转空间相关性，减少对数值逐点复刻的要求|
|moe_alpha40_sam_spatialcc_kd2_seed42|0.86204966|空间CC蒸馏系数.6→2，同架构加强教师梯度|

总CC绝对提高约0.03298983。主要证据指向训练/监督改进，而非网络变大：

1. 更弱的图像增强和准确的教师目标配准避免图像与蒸馏目标错位；不是增强越强越好。
2. 分组学习率和分阶段训练让相位、电子读出共同适应，避免只延长epoch或重置alpha。
3. 教师只离线给训练图提供软目标，EMA只是单模型权重平均；二者不在推理中形成多模型集成。
4. SAM先在电子参数邻域寻找高损失方向，再回传更新原模型；光相位仍随第二次反向更新，
   不是“只训练电、光被冻结”。配对普通续训为0.85962192，SAM.05为0.86133209，
   这一对照比简单前后对比更支持SAM的作用，但仍仅单seed。
5. 6912参数DW的增益不宜夸大：同训练控制0.85883157、加DW0.85921630，差约0.00038473；
   强KD组合才到0.85953132。没有证据把全部提升归给这两个卷积。
6. 最近MGD/深监督/裁剪等尝试没有成为正式候选，不能把它们描述为0.86205的组成或原因。

训练集10000、public-test5000，test用于周期选模，多次试验也使用该测试结果；
这些是有选择偏差的单seed公开测试成绩，不是独立未接触测试或真实光路成绩。
同规格头Qwen现为约0.88968469，仍未追平；本次按用户决定先停在当前候选。
