# 训练时空间特征监督（不增加推理网络）

## 已完成对照结果

2026-09-10，`moe_alpha40_hint_control_seed42`与`moe_alpha40_hint_cosine_seed42`
均完成50轮，源码`9bc65286e03d327c962abaa0e35a52770b2a8dab`。

| 组别 | 更新后最高test CC | 第50轮test CC | 最终保留 |
|---|---:|---:|---|
| control | 0.85780809（epoch1） | 0.85570458 | epoch0，0.85812011 |
| cosine | 0.85769613（epoch1） | 0.85596133 | epoch0，0.85812011 |

两组best的5000张完整复评均为0.85812011，alpha=0.43413550/0.44144565；
专家选择占比23.56%/26.32%/23.19%/26.93%，有效专家数3.98277/4，无未使用专家。
这不是新训练得到了同样优良的新权重，而是没有超过源权重，故保留了epoch0。
普通cosine提示未产生有效提升，不继续重复这一配置；去空间均值版本单独判断，不混为同一结果。

对应run内`selected_checkpoint_test_evaluation.json` SHA256：

- control：`0eefc5a5e3d86c376757d20862a87b038c481510d4f310e2c9d6b4fe5d4520d7`
- cosine：`aee4ac592de7d1022b0a2cba9ca671784556a1123c7a3cb122909c2080b17014`

## 方法假设

假设：只蒸馏最终单通道显著性图，可能不足以监督两层光电融合后的192通道特征。
参考[FitNets, ICLR2015](https://arxiv.org/abs/1412.6550)的中间特征提示与训练回归器思想。
本实现不是完整FitNets复现：使用SALICON空间任务、固定现有光电模型与逐像素通道cosine损失。
论文只提供方法依据，不保证本任务提升，CC≥0.87仍是待验证目标。

## 推理与训练边界

推理完全保持`moe_alpha40_adaptive_keepkd`：冻结Qwen patch前端、光router Top2、
两层同尺度光电融合、alpha≥0.4、478有效面积、224专家、17微米、10cm、20%–30%零级分量、
pixel位移0、原85412参数解码头。无新增attention、Transformer、分支或推理次数。
本对照不使用空间FFN新增模块，从原结构历史best CC=0.85812016开始。

仅训练时：教师的`AlignedReadout.decoder`输入是经过adapter+LN的`[B,192,14,14]`真二维网格。
学生的第二融合后latent也为同尺寸，添加一个**训练专用**192→192无bias的1×1投影（36864参数），
初始化为单位阵，补偿二者通道基底不同；每个像素对192通道做L2归一化，损失为平均`1-cosine`。
不强行把教师激活幅度注入光路；teacher与投影都不参与测试或部署。
原模型`core`与`saliency_head`的state_dict/architecture标签不变。
投影保存在last_checkpoint的独立`training_only_hint`字段；不是推理权重的一部分，不装入student。
best checkpoint仅含原模型有效权重；此字段为null。EMA只作用于core/head，不包含投影。
现有trainer的“重载权重+新optimizer”不等价于精确断点续训，不能把它写成恢复optimizer/RNG。

## 缓存与数据安全

仅10000个train2014 ID，顺序必须与manifest完全一致；不缓存test特征。
教师仍为同规格头Qwen baseline，SHA256：
`531c4a330fc05e2c27b037784588716d248a29d1ab8da951d443165dcce552f8`。
float16缓存约752640000字节（约718MiB）；一个共享缓存，不复制到每个run。
记录教师SHA、数据/标注SHA、源码commit、缓存SHA、真实网格合同；拒绝错序、错误尺寸或非有限值。
本轮特征监督禁用增强以确保严格对齐，不把插值后的特征伪称为教师真实增强输出。
正常公开测试5000张、无独立validation；起点/epoch1/每5轮/末轮选最高CC，存在测试选模偏差。

## 两组受控训练

`hint_control`与`hint_cosine`的源、GT损失、map KD=.6、学习率、EMA、样本顺序相同。
唯一差异是cosine组附加hint权重.1→.02，30轮线性退火，投影LR2e-4。
共同源SHA：`36333e6ba013cac3a2801bce4babcc2fb5cd36adf02969e019437b40ceebaffd`。
最多50轮，联合训练，41起精修；电子1e-5、相位2e-4、router/CCD读出/头各2e-5。
其余采用历史原结构策略；保留best/last，自动调速，不改变正在运行的viewreg对照。
原legacy epoch继续用于control，hint epoch是任务内等价损失+显式hint；有零hint梯度更新等价测试。
所有包含训练专用投影的参数都参与梯度裁剪；投影不使用weight decay。

### 基于缓存审计追加的空间去均值对照

固定缓存SHA `39aae6b13452975186266672e51216e72391112df65b23c453de030033826304`，
10000张教师特征`F[N,192,14,14]`，float32累加：
`sum_n mean_c(mean_hw(F)^2) / sum_n mean_chw(F^2) = 0.931409527`。
即93.14%的特征平方能量属于每个样本各通道的空间常量部分。
这是电子特征的共同分量，**不是物理未调制光的比例**，不改变20%–30%光学DC。
普通cosine可能主要拟合共同分量，故增加`hint_centered`：
投影后学生特征和教师特征分别减去各自空间均值，再做同样的逐像素通道cosine。
仅训练损失变换，实际送给解码头的学生特征不变；其余初始权重、系数、数据、优化器完全相同。
有共同大偏置/不同局部结构的单测，证明新损失对局部差异敏感、对空间常量偏置不敏感。
这只是受控训练假设，是否提高完整测试CC仍需实际验证。

## 可复现命令（仓库根目录）

先Git拉取已测试发布的commit；先查GPU空闲容量，以下GPU编号是示例。

```bash
conda activate xml
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
TASK=LightGenV2/tasks/t03_saliency
python -m pytest "$TASK/tests" -q
CUDA_VISIBLE_DEVICES=6 python -u -m LightGenV2.tasks.t03_saliency.export_teacher_features --config "$TASK/configs/moe_alpha40_hint_cosine.yaml" --checkpoint "$TASK/runs/simulation/qwen_aligned_head_staged_seed42/best_checkpoint.pt" --output "$TASK/runs/simulation/teacher_features_20260910/teacher_train_features.pt" --batch-size 16
CUDA_VISIBLE_DEVICES=1 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_hint_control.yaml" --phase all
CUDA_VISIBLE_DEVICES=5 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_hint_cosine.yaml" --phase all
CUDA_VISIBLE_DEVICES=6 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_hint_centered.yaml" --phase all
```

勿重复写同一输出目录；缓存完成且SHA验证后再训练hint。环境沿用xml/Torch2.6.0+cu124/Transformers4.57.3。
产物固定`runs/simulation/moe_alpha40_hint_<control|cosine|centered>_seed42/`。
查看feature_hint_provenance、resolved_config、初始化SHA、run_manifest的commit/命令、
metrics/training_history的hint/lr/CC及selected_checkpoint_test_evaluation。
本文件为方法与操作协议，不是已达到0.87的成绩声明。
