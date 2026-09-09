# 同规格头baseline与光电最优：独立固定权重复评

本次不是重新训练，也不是准确分类比例；CC是逐图Pearson相关系数的样本平均。
复评代码`8e093743bf4756a2fedd598fec3ba52f010e8a3b`，40项任务测试通过。
Qwen运行于A100，光电运行于4090，torch2.6.0+cu124；本次不测速度/功耗。

| 系统 | CC | 独立float64 CC | SIM | NSS |
|---|---:|---:|---:|---:|
| Frozen Qwen24 + aligned head | 0.8896846851 | 0.8896846851 | 0.83813736 | 1.00168349 |
| Optical Router Top2 best | 0.8581201437 | 0.8581201387 | 0.82209839 | 0.96558752 |

差距0.03156455 CC。5000张val2014作为public test，未新划validation；原权重按public-test选定，
仍有选择偏差。不能把0.88968写成88.968%分类准确率，或宣称Qwen零样本。
两组有序test IDs SHA256相同：`625dec6bc15b2d737d39bc252cfa0c354de217fec0266dcda568913f4a3496d0`。
Qwen权重SHA256：`531c4a330fc05e2c27b037784588716d248a29d1ab8da951d443165dcce552f8`，epoch80。
光电权重SHA256：`36333e6ba013cac3a2801bce4babcc2fb5cd36adf02969e019437b40ceebaffd`，续训epoch5。

证据在本任务`runs/simulation/aligned_recheck_20260909_qwen/`及
`aligned_recheck_20260909_optical/`：`reproduction.json`、`per_image_cc.csv`、实际配置和数据清单。

## 解码头参数与公平性

| 部分 | Qwen | 光电 |
|---|---:|---:|
| Linear1024→192 + LayerNorm192 | 197184 | 197184 |
| SaliencyDensityDecoder | 85412 | 85412 |
| 上述合计 | 282596 | 282596 |

适配器在Qwen冻结主干之后，在光电主体之前；因此它们是同规格而非同位置。
光电还有可训练电子残差、CCD读出和相位，不能把上表合计当成整套模型参数量。
两者训练历史/预算也不相同。Qwen执行完整24个冻结视觉Transformer层，光电执行零个原生视觉层。

同一解码器：`[B,192,14,14]`→LN/逐点Linear192→128→两个DW3×3+PW1×1的局部残差单元→
双线性上采样28/56/112/224，每级DW3×3+PW1×1+GroupNorm+GELU，通道96/64/32/16→
16通道局部细化→1×1输出单通道logits→空间softmax得到密度图。
无attention。参数分项：LN384，投影24704，两个局部单元35586，上采样24288，细化433，输出17。

## 逐图差距与下一步（尚未启动训练）

光电在1422/5000张上高于Qwen。按测试后观察到的差距排序，最大20%平均差距0.11914339，
其余80%平均差距0.00966984。这是描述性分组，不是预先定义的子集或独立泛化结果，
不可据此挑测试样本训练。应在train上重新分析教师/学生与GT的误差分布。

建议顺序：

1. 冻结已有光电主体，只重训同规格85412参数头，诊断特征可读性/共同适应问题。
   两边头尺寸已相同，因此不能先认定放大头就能消除差距。
2. 用train的GT评估教师质量，尝试有上限的困难样本加权与选择性KD；
   教师在该训练样本不优于学生时降低其约束，而不是像上一轮对全部样本统一撤KD。
   当前全局撤KD已验证无收益；选择性方案只是待检验假设。
3. 尝试中间空间特征蒸馏：教师的14×14×192特征提供更直接的监督，
   学生使用训练期对齐投影而非要求独立训练的通道逐项天然对应；先适配后联合微调。
   教师及额外对齐投影只用于训练，推理结构/光路/Top2/alpha≥0.4/DC20–30%均不变。
   参考[FitNets, ICLR2015](https://arxiv.org/abs/1412.6550)的中间提示思想，非照搬整套架构。

不保证达到0.88968；先验证能否稳定超过0.85812，不再重复已失败的统一弱KD、小LR长续训和5×5/GRN组合。

## 复评命令

在仓库根目录，激活xml，设置本机Qwen/data路径；输出目录必须不存在：

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
TASK=LightGenV2/tasks/t03_saliency
CUDA_VISIBLE_DEVICES=6 python -m LightGenV2.tasks.t03_saliency.recheck_aligned --system qwen --config "$TASK/configs/moe_staged_alpha_free.yaml" --checkpoint "$TASK/runs/simulation/qwen_aligned_head_staged_seed42/best_checkpoint.pt" --run-dir "$TASK/runs/simulation/aligned_recheck_20260909_qwen"
CUDA_VISIBLE_DEVICES=3 python -m LightGenV2.tasks.t03_saliency.recheck_aligned --system optical --config "$TASK/configs/moe_alpha40_adaptive_keepkd.yaml" --checkpoint "$TASK/runs/simulation/moe_alpha40_generalize_kd060_seed42/best_checkpoint.pt" --run-dir "$TASK/runs/simulation/aligned_recheck_20260909_optical"
```
