# SALICON 泛化优化：同步弱增强与早期重新适应

## 轻量电子容量对照：16维全局空间混合

`moe_alpha40_viewreg_global16.yaml`继承已完成的强KD配置，不继承mix50；
同一较早来源、相同80轮/增强概率1/KD2→.6/损失/光学噪声/光路/alpha约束。
只在两层**已有电子残差**内部扩大空间交流，物理/模型主路径仍只有E和O两支。
参考[MLP-Mixer](https://arxiv.org/abs/2105.01601)跨位置共享通道的静态MLP思想，
不加载其预训练权重、完整网络或额外主干，不使用attention、Q/K/V或输入条件权重。

位置在原`token_pointwise`内：原LayerNorm、3×3深度卷积和GELU之后，
先恢复Qwen块序到真实14×14空间；每通道的196个位置经196→16→196静态空间MLP，
加回其输入，再执行原192→192 pointwise以及原有门控残差。后续原通道MLP（含已有空间FFN）、
两级同尺度光电融合和85412参数解码头不变。没有新增第三条特征输入或输出分支。

新增两矩阵/层，无新bias；每层`196*16+16*196=6272`，两层12544参数。
结合已存在的空间FFN，相对最初3×3电子残差总共新增19456，而非仅12544。
新增矩阵MAC约2408448/图（不含GELU/重排，非实测速率/功耗）。
4×4低频二维DCT基初始化下投影，上投影全零，初始函数保持；两矩阵均训练，
下投影在上投影离开零后获得梯度。基在真实14×14坐标上构造，不把块序当作连续图像。
初始化不消耗全局随机流，样本独立，固定196 token合同不接受padding/变长序列。

新增矩阵单独AdamW组`electronic_global_spatial`，初始LR=2e-4、weight decay=0，
随既有阶段/平台调度缩放；前5轮原电子主体冻结，新投影和既有空间FFN可训练。
只允许审计过的rank16，暂不与GRN/13×13核组合。严格记录`_global_r16`结构后缀，
迁移只容许对应四个新矩阵；同结构checkpoint复载不重新置零已学矩阵。

```bash
python -m pytest LightGenV2/tasks/t03_saliency/tests -q
python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_viewreg_global16.yaml --phase all
```

产物在本任务`runs/simulation/moe_alpha40_viewreg_global16_seed42`；按实时资源选GPU。
与强KD组的0.859531作完整测试对比，同时检查alpha、专家选择和实际相位变化。
这是小容量全局上下文的待验证假设，不声称论文保证SALICON改善或已经达到0.87。

## 后续单变量：原图与弱增强混合训练

`moe_alpha40_viewreg_mix50.yaml`继承强KD组全部设置，只把
`augmentation.apply_probability`从默认1改成0.5。每张训练图独立抽取，约一半走
原图/原密度/原fixation/原教师logits，其余使用既有同步弱增强。
这里的“原图”不代表关闭光学噪声：所有训练样本继续按原配置承受20%–30%随机未调制分量。
不改变推理结构、电子参数量、光路、Top2、alpha或损失；不加在线教师。
依然从同一较早0.85468765来源开始，不从0.859531 best开始，以隔离增强概率的影响。
最多80轮，61轮起全用原图精修；只保存best/last，完整测试选模偏差仍适用。

动机：强KD组在关闭图像增强后的61–65轮明显恢复，但继续精修随后趋于平台。
在源码`966fb800`上做过128张train-only诊断：从有序10000训练图用
`sorted(random.Random(17042).sample(range(10000),128))`抽样，batch16、torch seed42，
用强KD配置的`AlignedWeakLoader`（内部seed=42+1703）生成一组增强视图，
既有教师SHA/原图缓存不变，让冻结同头Qwen实际推理同一组增强图（无外围autocast）。
独立float64逐图CC：实际教师与变换教师的相似度均值0.96358252；对同一增强GT的CC
分别0.88439619与0.88765667，实际重新推理平均反而低0.00326047，128张中60张更好。
因此不直接投入全量重生成教师增强缓存；小样本诊断也不证明哪种蒸馏训练最终更优。
此次优先检验保留原图监督是否减轻训练/推理输入分布差异，而非把该假设写成已取得提升。

默认p=1不消耗新增概率RNG，保持原增强序列；概率选择用独立seed=42+4703。
混合分支不修改原始batch；教师按sample_id与该图的实际变换对齐。
历史CSV增加`augmentation_images`/`augmentation_total_images`，可核验真实增强数量；
eval不使用该loader。p=0/p=1、混合对齐、计数、默认兼容和配置不变量有单元测试。

```bash
python -m pytest LightGenV2/tasks/t03_saliency/tests -q
python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_viewreg_mix50.yaml --phase all
```

新run为本任务`runs/simulation/moe_alpha40_viewreg_mix50_seed42`，不得覆盖已有run。
以全5000张测试与原强KD组对比，目标仍为CC≥0.87。

## 2026-09-10已完成：强蒸馏组的新最佳权重

`moe_alpha40_viewreg_cffn_kd2_seed42`完成80轮，选择epoch65 EMA，
重新加载best在全部5000张public-test上得到：

|指标|结果|
|---|---:|
|CC|0.8595313201904297|
|KLD|0.11409103026390076|
|SIM|0.8229441816329957|
|NSS|0.9672923250198364|
|AUC-Judd|0.7698853058936339|
|MAE|0.07595092777013779|

比此前未加校准的0.85812014提高约0.00141118，距离0.87仍差0.01046868。
这是更新后epoch65的结果，不是epoch0保留权重；末轮CC=0.85943667。
alpha=0.43102312/0.44123352；专家选择次数2327/2639/2301/2733，
占比23.27%/26.39%/23.01%/27.33%，有效专家数3.97723/4，无未使用专家。
相对初始化，router/四专家/全局相位的圆周相位RMS变化分别约
0.00881 / (0.08998,0.09132,0.15370,0.15069) / 0.11044 rad。
相位确实训练更新，但这些数值不等于光学准确率贡献比例。

该分数遵守既定标准eval：图像不增强、随机光学扰动关闭；20%–30%未调制分量仍在训练中保留。
public-test参与选模和平台调速，结果有选择偏差，不是未接触的独立测试。
控制组、普通KD的空间FFN组及13×13组尚未全部完成，不据此宣布完整消融胜负。

- 训练源码commit：`966fb80087776a9a22982a82e684fff32d52e71f`。
- best SHA256：`c88e1a41febc878cf87d6c80d4e27d7ac0c6bddc11d73a29497490d2e841ad73`。
- resolved_config SHA256：`ec5bade2f177b230e4d3d3e6448ecf26bde884a2b8cc640cce203696a656df54`。
- selected_checkpoint_test_evaluation.json SHA256：`7aed5b0dc0b493fb6a9b2b120575ffed25a93644304ef40a95f49151b54392dd`。
- 全部产物位于本任务`runs/simulation/moe_alpha40_viewreg_cffn_kd2_seed42/`；
  `best_visualization/`包含相位与显著性样例。只保存best/last，不新增周期权重。

复现训练使用本文后面的强KD命令；固定权重额外独立float64复查（不训练）：

```bash
TASK=LightGenV2/tasks/t03_saliency
python -m LightGenV2.tasks.t03_saliency.recheck_aligned --system optical --config "$TASK/configs/moe_alpha40_viewreg_cffn_kd2.yaml" --checkpoint "$TASK/runs/simulation/moe_alpha40_viewreg_cffn_kd2_seed42/best_checkpoint.pt" --run-dir "$TASK/runs/simulation/aligned_recheck_20260910_viewreg_kd2" --batch-size 32
```

使用尚不存在的复查目录，不能覆盖既有证据。上述额外复查也已完成：
独立float64 CC=0.8595312562517528，与同次累积器CC差3.39e-10；
batch32与训练后batch48复评仅约6.4e-8差异。5000个ID清单SHA256
`625dec6bc15b2d737d39bc252cfa0c354de217fec0266dcda568913f4a3496d0`
与此前同规格头Qwen/光电复查完全一致。
复查`reproduction.json` SHA256：
`7f3f27837c96c152620f251d1b3da5409beb4cf064b06d32b6dbc2b08aa9ab89`。
按sample_id与旧光电`aligned_recheck_20260909_optical/per_image_cc.csv`一一配对，
2936/5000张改善，CC差均值0.00141112、中位数0.00135197。
这只是当前单seed、已参与选模测试集上的描述统计，不是多seed显著性结论。

## 追加的13×13轻量空间上下文对照

`moe_alpha40_viewreg_kernel13.yaml`仅相对`moe_alpha40_viewreg_control.yaml`
扩大现有两个192通道depthwise token mixer：3×3→13×13。
不是新增残差层、特征分支、attention或预训练CNN。解码头保持85412参数，
不加入两参数校准、GRN或空间FFN。新增参数严格为`2*192*(13²-3²)=61440`，
在14×14 token网格上新增约1204万MAC/图（只计扩大卷积的差额，不是整机耗时/能耗）。
光Router Top2、alpha≥0.4、20%–30%训练零级扰动、478 ROI、224专家及光学传播次数均不变。

动机参考[RepLKNet (CVPR2022)](https://arxiv.org/abs/2203.06717)的大核深度卷积设计；
仅借鉴扩大空间上下文，不引入其完整网络或声称论文证明本任务有效。
相比之前3→5的小范围扩大，本次核在14×14特征网格上覆盖更大邻域。
从控制组同一0.85468765较早来源开始；原3×3放在13×13中心，其余零初始化，
不消耗额外随机数，不重置alpha。训练之前核验原函数保持，随后外圈允许训练。

其余学习率、权重衰减、80轮预算、前5轮冻结电子主体、60轮同步弱增强、
后20轮无增强精修、KD=.6、EMA及完整测试规则与控制组完全相同。
这意味着新增核也在前5轮冻结，避免将初始化适应差异混入对比。

```bash
python -m pytest LightGenV2/tasks/t03_saliency/tests -q
python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_viewreg_kernel13.yaml --phase all
```

运行前按实时显存设置CUDA_VISIBLE_DEVICES，不停止他人进程。
新run：`runs/simulation/moe_alpha40_viewreg_kernel13_seed42`；保留best/last，
以完整测试、专家占比及参数更新核验决定是否采用。当前未声称达到0.87。

## 原始同步弱增强三组协议

目标是完整5000张public-test平均CC达到0.87；目标不是结果承诺。
前一轮cffn三组均34轮早停，更新后最高CC为0.857841/0.857759/0.857757，
均保留epoch0的0.858120源。训练CC约0.881而测试下降，提示该续训方案泛化退步。

## 本轮不变的边界

- 冻结Qwen patch前端，不执行原生Transformer/attention；两级光电同尺度融合。
- 光router四专家Top2；alpha至少0.4；20%–30%随机相干未调制分量；pixel位移0。
- 原478有效面积、224专家、17微米、10cm传播不改；解码头85412参数不变。
- control是原电子残差；cffn只在现有两残差内部加6912个DW3×3参数，无新分支。
- 单图单次原始输入推理；不加测试增强、集成或使用真值的测试后处理。

## 数据和训练

train2014=10000，val2014=5000作为public test，无独立validation。
epoch0、1、每5轮、末轮完整测试，按CC选best，自动调速也使用public test，存在选择偏差。
只保留best/last及日志，不产生每5轮权重。测试集不进入增强、teacher缓存或反向传播。

三组共同从`moe_alpha40_refine_weakaug_seed42/best_checkpoint.pt`开始（CC约0.85468765），
这比反复无增强蒸馏续训的0.85812来源早，但并非从头训练/全新数据。
源SHA256：`de477b8c13c46c50cb9f17eb0b62bc577aaefef5512887e1e17a86d0affb5eea`。
不重置alpha，所有已有张量严格迁移，新卷积恒等初始化。

|配置后缀|电子结构|KD权重|
|---|---|---|
|viewreg_control|原结构|固定0.6|
|viewreg_cffn|增加展开空间DW3×3|固定0.6|
|viewreg_cffn_kd2|同上|2.0线性降至0.6，60轮到达|

增强在已统一224×224的训练图上执行：边长保留95%–100%的随机裁剪、缩回224、
50%水平翻转、亮度/对比度各±5%。图像、GT密度、fixation、teacher用同一空间变换。
密度双线性缩放后重新归一化为和1；fixation最近邻缩放，裁掉全部fixation时回退整幅图。
教师必须先softmax成概率密度，再变换、归一化、取log返回KD接口（温度固定1）。
**这是近似视图一致性正则化，不是教师重新推理增强图；裁剪后注视与光度不变性只是训练假设。**
不使用MixUp：现有NSS把fixation二值化，直接混合会丢失混合权重，本轮避免改动损失语义。

教师仍是同头Qwen：SHA `531c4a330fc05e2c27b037784588716d248a29d1ab8da951d443165dcce552f8`。
缓存`runs/simulation/generalize_alpha40_20260909/teacher_train_logits.pt`必须精确包含排序后10000个train ID；
代码校验身份/尺寸/来源，每个run记录缓存SHA及变换说明，推理不运行教师。

最多80轮，前5轮冻结原电子组，6–60联合训练，61–80关闭增强精修；EMA=.995。
初始LR：电子3e-5、新DW2e-4、相位5e-4、router5e-5、CCD读出3e-5、头5e-5。
weight decay=.03，相位/router/新DW为0；GT KL1+CC1.5+SIM.25−NSS.1。
从30轮开始每6次测试无至少.0001改善则额外LR减半，最多两次，之后再平台则早停。
三组增强种子、样本顺序、batch32/test48一致；训练CC是在增强图上，不能直接当作原图泛化差距。

## 操作命令

先通过Git拉取包含本文件的已发布commit；依赖同之前xml环境（Torch2.6.0+cu124、Transformers4.57.3）。
在仓库根目录，检查GPU容量后逐项启动；并行使用不同终端/受控调度，不重复同一run目录。

```bash
conda activate xml
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python -m pytest LightGenV2/tasks/t03_saliency/tests -q
TASK=LightGenV2/tasks/t03_saliency
CUDA_VISIBLE_DEVICES=6 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_viewreg_control.yaml" --phase all
CUDA_VISIBLE_DEVICES=6 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_viewreg_cffn.yaml" --phase all
CUDA_VISIBLE_DEVICES=3 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_viewreg_cffn_kd2.yaml" --phase all
```

产物：本任务`runs/simulation/moe_alpha40_viewreg_<control|cffn|cffn_kd2>_seed42/`。
查看run_manifest的commit/命令、resolved_config、teacher_cache_provenance、初始化SHA、
metrics/training_history.csv（augmentation_active/KD/LR/测试曲线）、training_report以及selected_checkpoint_test_evaluation。
最佳可视化在best_visualization。比较本轮更新后的best、历史0.85812以及Qwen0.88968，
不得把warmstart保留下来的分数标为本轮提升。获得改善后仍需完整权重复评和光router/alpha审计。
