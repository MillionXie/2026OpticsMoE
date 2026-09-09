# SALICON baseline 复现与公平性

[SAM训练对照](SAM_TRAINING.md)：不增加推理结构，在电子参数子空间进行训练时扰动，
有同源普通续训组、随机噪声配对和单次optimizer/EMA更新的实现检查。

当前完成并独立核验的光电候选：SAM.05完成50轮、best epoch5 EMA，
重载完整5000测试CC=0.86133209，独立float64 CC=0.86133204；最终best字节与独立复查一致。
alpha≥0.4且四专家无明显坍缩；权重SHA、相位更新、命令和差距见[SAM完成结果](SAM_TRAINING.md)。
来源强蒸馏.85953132的历史结果保留在[泛化优化](VIEW_REGULARIZATION.md)。
同规格头Qwen为0.88968469，0.87目标仍未达到。

[两参数读出校准诊断](READOUT_CALIBRATION.md)：冻结原光电网络，只用训练数据拟合两个全局标量；
不添加特征分支，保留原指标，并成对执行完整测试。尚未证明有效，不替代正式模型。

[训练专用空间特征提示](FEATURE_HINTS.md)：固定原推理结构，以教师中间特征提供额外监督；
投影仅训练使用，不计入/不加入部署网络，含严格缓存身份与无hint等价测试。

[同步弱增强与早期重新适应](VIEW_REGULARIZATION.md)：目标CC≥0.87，三组受控实验；
只改变训练视图/蒸馏策略，保留光学约束，明确增强教师目标的近似假设和复现命令。

[电子残差内部空间FFN](SPATIAL_FFN_RESIDUAL.md)：原结构/普通3×3/空洞3×3三组，新增6912参数，
保留光学约束与同规格头；含论文依据、初始化/坐标合同和训练命令。

[2026-09-09同规格头独立复评](ALIGNED_RECHECK_20260909.md)：Qwen CC=0.88968469，光电CC=0.85812014，
同5000张清单与独立float64逐图CC，参数审计、差距分析和后续建议。

当前同规格头的固定权重复评入口：`python -m LightGenV2.tasks.t03_saliency.recheck_aligned --help`。
支持Qwen/光电，完整5000张public-test、逐图float64独立CC及样本ID清单SHA；不训练、不测速度功耗。
使用`--system qwen --config LightGenV2/tasks/t03_saliency/configs/moe_staged_alpha_free.yaml`
或`--system optical --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_sam005.yaml`，
并显式提供`--checkpoint`和新的`--run-dir`。原同头Qwen权重为
`runs/simulation/qwen_aligned_head_staged_seed42/best_checkpoint.pt`（路径相对于本任务），
光电已核验候选为`runs/simulation/moe_alpha40_sam005_seed42/best_checkpoint.pt`。

[平台期受控精修](ADAPTIVE_REFINEMENT.md)：从历史best启动，比较KD约束与GT CC目标，
含自动降学习率/早停机制；不增加推理结构。

[早期起点与论文依据的轻量电子残差试验](EARLY_LIGHTWEIGHT_RESIDUAL.md)：包括ConvNeXt/GRN借鉴范围、
五组对照、权重迁移、参数预算及完整命令；不改变本页baseline的历史含义。

[Baseline????????](BASELINE_METHODS.md)?????baseline??????????????????????2026-09-09?

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
这里的log不是Qwen或图像查看器加的。历史配置保留用于复现；本轮正式优化已改用独立的
`mean_only`合同，仅除整幅均值，无log/gamma/上限裁剪；末端电子读出网络仍有其常规激活。
不能直接将旧log系数设0（那会输出全零）。新合同有独立checkpoint架构标签，旧权重仅作为显式迁移初始化。
光路参数：17μm、10cm、4专家Top2，一次router读出、两次特征光传播读出；20%–30%随机相干零级分量。

**噪声与指标口径（2026-09-10源码核验）**：上述20%–30%是训练增强，不是下表测试时持续注入的漏光。
共享光学后端`_apply_coherent_zero_order`和`_perturb_ccd`在`not self.training`时直接返回未扰动值，
标准`evaluate_model`使用eval模式；位置扰动和相位dropout也在标准测试关闭。
因此当前光电CC及CC≥0.87目标均指5000张public-test上的**理想光学仿真评估**，
不能描述成“20%–30%漏光环境下已取得该分数”，也不能用随机带噪测试替换原列来宣称达标。
训练仍保留既定20%–30%未调制扰动；实测/固定漏光强度下的鲁棒性须另列协议与结果。
已有专家选择占比23.54/26.80/23.38/26.28%，没有全局坍缩。
比较属于**端到端系统比较**，不是只替换一种模块的严格参数量/训练预算受控消融：主干、解码头、
训练轮数和测试频率均有差异。另有D2NN CC=0.83456；其匹配的是两位激活专家的相位参数，
不包括MoE的额外router/global相位，不应写成“总参数完全相同”。

## 复现步骤（仓库根目录运行）

需要完整仓库及其 `experiments/` 兼容后端、SALICON原始图像与fixation JSON、完整本地Qwen模型/处理器、
上述teacher checkpoint。原始数据/模型不提交Git。环境锁定清单由每个run的 `environment.txt` 提供；
本次验证环境已收录为 [baseline_environment.txt](evidence/baseline_environment.txt)，
其中PyTorch2.6.0+cu124、transformers4.57.3、NumPy1.26.4。该文件是完整环境审计记录，
不是要求把无关包也全安装；换显卡需要选择兼容的PyTorch构建，再重新执行性能复评。
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

## 本轮优化（不增加网络分支，不取消光路由/DC）

先检验训练而不是增加分支。保留原baseline与原0.8291，使用原best权重续训并重建优化器，
仅保留best/last；epoch0先复评并纳入best，防止续训退化覆盖好权重。
候选A：CCD改用mean_only；关闭输入/相位/CCD及router的16px位置扰动，保留DC与其他噪声，训练100epoch。
候选B：在A基础上CC损失权重0.5→1.0。均为每5epoch测public test选best，结果具有选模偏差。

```bash
python -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config LightGenV2/tasks/t03_saliency/configs/moe_dc20_mean_only_continue.yaml --phase all
python -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config LightGenV2/tasks/t03_saliency/configs/moe_dc20_mean_only_cc_continue.yaml --phase all
```

训练产物在配置同名run。最终除CC/KLD/SIM/NSS外还需查看router占比、alpha、相位与光场。
关闭位置扰动的新候选不能宣传为已经验证具有与旧版相同的位置鲁棒性；实测需重新核验。
早先两份保留log的诊断续训 `moe_dc20_no_shift_continue_seed42` / `moe_dc20_cc_continue_seed42`
已经人工停止，保留现有日志和best/last作为审计记录，不属于正式候选，也未删除。

并行时先检查GPU空闲情况，设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 再指定 `CUDA_VISIBLE_DEVICES`，
或者直接指定GPU UUID；不要假定默认CUDA序号总与nvidia-smi物理序号一致。
