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
```

产物分别为任务`runs/simulation/moe_alpha40_sam_control_seed42`和
`runs/simulation/moe_alpha40_sam005_seed42`（后者sam与005之间没有下划线）。
检查run_manifest的commit/命令、resolved_config中的rho、初始化SHA、完整测试历史、
`selected_checkpoint_test_evaluation.json`的alpha/专家占比/实际相位更新，及best可视化。
若best仍是epoch0，必须报告未超过源权重，不能记作新训练成绩。
