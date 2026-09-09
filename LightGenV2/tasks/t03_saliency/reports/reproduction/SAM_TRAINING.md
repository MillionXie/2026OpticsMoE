# 电子参数子空间SAM：推理网络完全不变

目标仍为完整5000张public-test平均CC≥0.87，不是结果承诺。
依据[Foret等，SAM (ICLR2021)](https://arxiv.org/abs/2010.01412)的局部最坏损失优化思想，
以小幅权重扰动后的梯度更新参数，检验是否改善仅降低训练损失时的泛化。
此处是电子参数子空间上的SAM＋AdamW，不是整篇论文/其分类数据集的复现，
论文结果不证明本SALICON配置必然有效。

## 严格固定的对照

两组都从已经完成并独立复评的`moe_alpha40_viewreg_cffn_kd2_seed42/best_checkpoint.pt`
出发：CC=0.85953132，epoch65 EMA，SHA256
`c88e1a41febc878cf87d6c80d4e27d7ac0c6bddc11d73a29497490d2e841ad73`。
复制权重、重建optimizer，不称为精确恢复旧optimizer/RNG。
不重置alpha，不重置已学空间FFN；不加入global16、13×13、GRN或两标量校准。
推理仍是冻结Qwen patch前端、两级E/O同尺度凸融合、光Router Top2、alpha≥0.4、
478 ROI/224专家/17微米/10cm与85412参数解码头。新增推理参数为0。

共同训练：10000张train、无图像增强、50轮上限、41轮起精修，EMA=.995，weight decay=.01。
LR：E1e-5，相位2e-4，router/CCD读出/头2e-5，原空间FFN5e-5；保持KD=.6和原GT损失。
仍保留20%–30%随机相干未调制分量及既有光学噪声；标准eval关闭随机光学扰动。
相同batch32/test48、seed42、AMP设置；起点/epoch1/每5轮/末轮完整测试选最高CC，
从15轮起按原平台控制降LR。没有独立validation，选择和控制使用public-test，有选择偏差。
只保留best/last；不覆盖已验证的源run。

|配置|优化区别|
|---|---|
|moe_alpha40_sam_control.yaml|SAM半径0，直接走原legacy单次训练循环|
|moe_alpha40_sam005.yaml|电子子空间SAM半径0.05，两次前向/反向后一次AdamW更新|
|moe_alpha40_sam001.yaml|同一设置，半径0.01|
|moe_alpha40_sam010.yaml|同一设置，半径0.10|

追加两档半径用于检验敏感性，不修改正在运行的0.05或控制组；四组来源、
训练预算和推理架构相同。0.05组第1轮全5000测试CC=0.86110220、控制组0.85962192，
是开展半径对照的初步依据，不是最终选定/独立复评结果，不据此宣称达到0.87。
原始控制/0.05源码commit为`66566410e2f2cae6a1359c98c340b2c7ebdbb692`。

## 实现边界

首次计算原完整损失梯度，在electronic、saliency_head、ccd_readout和已有空间FFN组上
形成共同L2范数归一化的扰动`epsilon=.05*g/||g||`。光学相位与光Router组不加此权重扰动，
但第二次反向仍对所有可训练参数求梯度并更新，包括光学相位和Router。
alpha参数始终通过原受限sigmoid映射，不能因SAM突破0.4下限。
对同一batch复用预处理/teacher目标，恢复首次前向前的CPU/CUDA/Python/NumPy RNG，
令两次前向的光噪声与dropout实现一致；结束后的RNG推进量等于一次常规前向。
这里“相同噪声”不等于无噪声，也不修改噪声幅度/物理模型。

在第二次反向后用备份精确恢复权重，再裁剪全体梯度并只调用一次optimizer.step，
EMA钩子也只执行一次。不累加首次梯度，不给部署权重保留epsilon。
第二次异常或非有限值时恢复原权重/RNG并报错，不带着临时扰动继续训练。
拒绝含BatchNorm的模型，以免双前向悄悄双计运行统计量；当前模型不使用BatchNorm。
测试指标算法/精度不变；SAM增加训练计算量，不增加部署推理次数。
日志/历史的训练CC与loss使用首次未加SAM权重扰动的前向，
`train_sam_loss_increase`记录第二次相对首次的完整损失变化，不用它代替测试CC。

## 操作

使用GitHub已发布且测试通过的commit；按实时GPU显存选择设备，不停止他人进程。

```bash
conda activate xml
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
TASK=LightGenV2/tasks/t03_saliency
python -m pytest "$TASK/tests" -q
python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_sam_control.yaml" --phase all
python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_sam005.yaml" --phase all
python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_sam001.yaml" --phase all
python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_sam010.yaml" --phase all
```

产物分别为任务`runs/simulation/moe_alpha40_sam_control_seed42`和
`runs/simulation/moe_alpha40_sam005_seed42`（后者sam与005之间没有下划线）。
新增半径对应`moe_alpha40_sam001_seed42`/`moe_alpha40_sam010_seed42`，也不加下划线。
检查run_manifest的commit/命令、resolved_config中的rho、初始化SHA、完整测试历史、
`selected_checkpoint_test_evaluation.json`的alpha/专家占比/实际相位更新，及best可视化。
若best仍是epoch0，必须报告未超过源权重，不能记作新训练成绩。

## 训练中的候选独立核验

2026-09-10：0.05组epoch5 EMA候选已经完成独立5000张复查（训练仍继续，非最终交付）：

|指标|独立复查值|
|---|---:|
|CC（独立NumPy float64）|0.8613320359025128|
|CC（原指标累积器）|0.8613320404052734|
|KLD|0.1127731899023056|
|SIM|0.8240760560035706|
|NSS|0.9686995490074157|
|AUC-Judd|0.7701752108567858|
|MAE|0.07696553013324738|

与同batch32独立复查的来源0.8595312563按sample_id逐图配对，2947/5000张改善；
CC差均值0.00180077965，中位数0.00155842016。KLD/SIM/NSS改善，MAE较来源0.07595093退步，
不能称为所有指标均改善或跨seed显著泛化提升；仍按public-test选模，距离0.87尚有差距。
训练内batch48评估CC=0.8613320866，两种batch的差约5e-8。

- 候选checkpoint SHA256：`5aa39e30c0c04c0411138dd83464067c73ec2d240c2391b1e71a6e74bd6215f0`。
- 独立复查commit：`e97718b5ee29d32738038eef79679563d5f3508a`，训练commit仍为66566410。
- `aligned_recheck_20260910_sam005_candidate/reproduction.json` SHA256：
  `fb31f3ecc6c4aaf5d87b5adfa0581129996949f5efa3e390fdca68048ff959c6`。
- 测试ID SHA256：`625dec6bc15b2d737d39bc252cfa0c354de217fec0266dcda568913f4a3496d0`，与原候选/Qwen一致。
- `per_image_cc.csv`保留全部逐图指标，未新增周期PT；该best路径之后可能更新，须核对SHA。

可用`recheck_aligned`对当前best进行完整5000张、独立float64逐图CC复算。
该工具把一次读取的checkpoint字节同时用于反序列化和SHA256计算，
不会在评估结束时误将已更新best的SHA写入旧权重的结果。
不会额外复制/保存PT。训练中复查是暂时候选，不代替最终选定best的完整复评；
若最佳权重随后更新，最终交付必须再验证最终SHA。

```bash
TASK=LightGenV2/tasks/t03_saliency
python -m LightGenV2.tasks.t03_saliency.recheck_aligned --system optical --config "$TASK/configs/moe_alpha40_sam005.yaml" --checkpoint "$TASK/runs/simulation/moe_alpha40_sam005_seed42/best_checkpoint.pt" --run-dir "$TASK/runs/simulation/aligned_recheck_20260910_sam005_candidate" --batch-size 32
```

复查目录必须尚不存在；按实际日期/目的命名，不覆盖旧证据。若训练正同时写checkpoint，
读取可能失败，应在完整写入后重试，不把损坏/不完整文件当作有效权重。

## 直流分量差异的训练集小样本诊断

2026-09-10，在上述epoch5/SHA=5aa39e30候选上进行只读诊断，未优化参数：
从`legacy.build_loaders(..., training=False)`的train dataset中用
`sorted(random.Random(17042).sample(range(10000),128))`取固定128张，batch16，
无图像增强/外围autocast，使用`independent_cc`逐图float64计算。
完整模型保持eval、phase dropout关闭；仅在光分支原`_apply_coherent_zero_order`
方法调用期间临时设该分支training=True，调用后立即恢复。其余CCD噪声/dropout保持关闭。
这不代表完整噪声鲁棒性评估，只隔离相干未调制项。振幅/相位eta均原样0.2–0.3，
保留原随机相对相位；每batch三次抽样的torch seed为`17042+batch_index*3+draw_index`。

|条件|对同一GT的平均CC|与无扰动输出的平均CC|
|---|---:|---:|
|无光学扰动|0.8754117339|1|
|仅未调制分量，抽样0|0.8759857500|0.9941213238|
|仅未调制分量，抽样1|0.8756770201|0.9938578242|
|仅未调制分量，抽样2|0.8754149858|0.9940261410|

该小样本没有显示单独直流项造成明显性能损失，故暂不依据“训练含直流、测试关闭”
直接新增clean/noisy双损失，也不减少20%–30%训练未调制约束。
这不能证明真实硬件无域偏移、全部噪声无影响或其他样本结论相同；不是正式5000测试分数。
