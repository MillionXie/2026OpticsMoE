# SALICON baseline 复现与公平性

## 口径（先读）

表中 0.8810 是 **冻结 Qwen3-VL-Embedding-2B 视觉主干 + 有监督训练的显著性解码头**，
不是零样本 Qwen，也不是 SALICON 官方隐藏测试榜单成绩。输入没有文本，语言 Transformer 不执行。
原 checkpoint 为历史 `salicon_vision_optical_saliency/checkpoints/teacher_best.pt`，SHA256：
`aefd5c6cab81c4d720ada9e93935cf90b16d8653771f081bf933a62aba6bc644`。
历史完整5000张记录 CC=0.88104494；5090D记录=0.88105177（不同硬件的微小数值差异）。
本轮重新评估/重新训练的结果以本目录后续 `RESULTS.md` 为准，不把旧记录当作新复现。

数据：SALICON2015r1 train2014=10000，val2014=5000。本项目把 val2014 用于周期测试及选模，
不另分验证集，因此存在测试选模偏差。输出密度图224×224，sigma=19指**原图像素**，
映射到224时按原图宽/高分别缩放，两系统相同。fixation按代码的1-based `[y,x]`映射并去重成二值图，
Gaussian滤波后保存为最大值归一化的8bit密度PNG；评估再归一化为和为1的密度。
CC 为每张预测概率密度与真值密度的 Pearson，再平均5000张，不是把所有图片拼起来算。
预测用空间 softmax，不做拟合真值的逐图后处理。脚本额外以 NumPy float64 独立核验 CC。
这个标签生成口径必须披露，不能直接与别的标签分辨率/模糊半径的论文成绩横比。

## 架构与公平性

| 项目 | 光电 MoE | Qwen baseline |
|---|---|---|
| 输入 | 相同224×224 RGB及Qwen处理器 | 相同 |
| 前端 | 冻结Qwen patch embedding与位置编码 | 相同 |
| 主体 | 两层光电融合；光router选4专家中的2个 | 完整冻结原生视觉Transformer |
| 中间特征 | 196空间token×192维 | 196空间token×1024维（merger之前） |
| 输出头 | 逐级上采样+深度可分离卷积显著性头 | 有监督训练的230257参数卷积显著性头 |
| 文本/语言网络 | 无 | 无 |
| 输出 | 1×224×224密度图 | 同左 |
| 训练 | 光相位/router、电子残差、解码头 | 只训练解码头 |
| 原训练预算 | 60epoch、每5epoch选test CC | 30epoch、每epoch选test CC |

光电每层为同尺度 `(1-alpha)E + alpha O`；不是把完整冻结视觉Transformer藏在电残差里。
更精确地说，E和O先分别按样本RMS归一化，凸融合后再共同缩放回原E的RMS；alpha是融合系数，
不是准确率/物理能量贡献百分比。电子残差是无attention的token mixer与通道MLP。
必须披露一个历史实现细节：`RobustCCDNormalizer` 做了非负检查、除整幅均值、上限12裁剪，
然后 `log1p(relative_intensity)`。因此当前模型**不是CCD之后只有线性归一化**；
这里的log不是Qwen或图像查看器加的。本轮受控续训不改这一历史算子，也不把它隐瞒成纯线性读出。
若要求严格取消CCD后非线性，应另立架构合同并重新训练/复测，不能直接将log系数设0（那会输出全零）。
光路参数：17μm、10cm、4专家Top2，一次router读出、两次特征光传播读出；20%–30%随机相干零级分量。
已有专家选择占比23.54/26.80/23.38/26.28%，没有全局坍缩。
比较属于**端到端系统比较**，不是只替换一种模块的严格参数量/训练预算受控消融：主干、解码头、
训练轮数和测试频率均有差异。另有D2NN CC=0.83456；其匹配的是两位激活专家的相位参数，
不包括MoE的额外router/global相位，不应写成“总参数完全相同”。

## 复现步骤（仓库根目录运行）

需要完整仓库及其 `experiments/` 兼容后端、SALICON原始图像与fixation JSON、完整本地Qwen模型/处理器、
上述teacher checkpoint。原始数据/模型不提交Git。环境锁定清单由每个run的 `environment.txt` 提供；
至少需要可用的PyTorch/CUDA、transformers（支持Qwen3VL）、NumPy、SciPy、Pillow、Matplotlib、PyYAML、pytest。
禁止用未记录的另一套标签缓存；复现脚本在新run中重新生成密度图。

Linux服务器准备路径（其他电脑只替换这两个路径）：

```bash
cd /DATA/DATA1/guest3/2026OpticsMoE
conda activate xml
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MODEL=/DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda
```

1. 固定权重重新评估所有5000张（任意支持的CUDA卡；不测速度/功耗）：

```bash
python -m LightGenV2.tasks.t03_saliency.reproduce_baseline --model "$MODEL" --run-dir LightGenV2/tasks/t03_saliency/runs/simulation/baseline_recheck_20260908
```

2. 从新初始化的解码头重新训练30epoch，Qwen保持冻结（这是比步骤1更强的训练可复现性检查）：

```bash
python -m LightGenV2.tasks.t03_saliency.reproduce_baseline --model "$MODEL" --retrain-head --run-dir LightGenV2/tasks/t03_saliency/runs/simulation/baseline_retrain_seed42_20260908
```

每次必须使用空run目录。查看 `reproduction.json`（性能、独立CC、权重/模型/标签SHA、commit）、
`per_image_cc.csv`、`resolved_config.json`、`environment.txt`；训练模式另含teacher_history和best/last。
检查独立CC与原实现差异接近浮点误差，样本数必须5000。单次新训练不保证逐位等同历史值；
若要声明训练方差，应后续补多seed，不能把一次成功复评说成多次重训成功。

3. 光电原版本复评（与历史数据独立保留）：

```bash
python -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --phase evaluate --checkpoint LightGenV2/tasks/t03_saliency/runs/simulation/moe_router_scale_dc20_seed42/best_checkpoint.pt --run-dir LightGenV2/tasks/t03_saliency/runs/simulation/moe_recheck_20260908
```

## 本轮优化（不改网络结构，不取消光路由/DC）

先检验训练而不是增加分支。保留原baseline与原0.8291，使用原best权重续训并重建优化器，
仅保留best/last；epoch0先复评并纳入best，防止续训退化覆盖好权重。
候选A：关闭输入/相位/CCD及router的16px位置扰动，保留DC与其他噪声，续训100epoch。
候选B：在A基础上CC损失权重0.5→1.0。均为每5epoch测public test选best，结果具有选模偏差。

```bash
python -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config LightGenV2/tasks/t03_saliency/configs/moe_dc20_no_shift_continue.yaml --phase all
python -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config LightGenV2/tasks/t03_saliency/configs/moe_dc20_cc_continue.yaml --phase all
```

训练产物在配置同名run。最终除CC/KLD/SIM/NSS外还需查看router占比、alpha、相位与光场。
关闭位置扰动的新候选不能宣传为已经验证具有与旧版相同的位置鲁棒性；实测需重新核验。
