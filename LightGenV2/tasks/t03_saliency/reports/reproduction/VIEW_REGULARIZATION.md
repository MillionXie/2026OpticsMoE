# SALICON 泛化优化：同步弱增强与早期重新适应

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
