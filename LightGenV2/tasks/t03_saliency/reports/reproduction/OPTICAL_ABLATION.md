# 同一权重去光复评（不重新训练纯电子模型）

目的：测量当前权重对光学分支的依赖，不把alpha或相位更新幅度直接当成性能贡献。
标准仍为完整SALICON官方val2014的5000张，每图CC取平均，公开测试参与过选模。
不改变图像预处理、标签、读出归一化或评价精度；正常测试关闭随机光学扰动。

`recheck_aligned --ablation remove_optical`调用已有运行时消融：
跳过光router、专家传播及全局传播，两级融合分别直接返回原电子残差E，系数为1。
不是将O置零后仍保留(1-alpha)衰减，不重训电子分支、不改相位文件、不修改保存的alpha。
正常方法仍要求alpha≥0.4；旁路仅用于这个明确标记的消融，不能用于满足正常模型的0.87目标。
Qwen不接受去光选项；默认`--ablation none`保持原复评行为。

## 命令与身份

在已配置SALICON数据/Qwen本地缓存的Git工作树运行，选择可用GPU，保留已存在的复评目录。
每次命令的输出目录必须尚不存在。示例候选是空间CC KD2的epoch5；两次必须核对相同SHA。

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
TASK=LightGenV2/tasks/t03_saliency
CONFIG="$TASK/configs/moe_alpha40_sam_spatialcc_kd2.yaml"
CHECKPOINT="$TASK/runs/simulation/moe_alpha40_sam_spatialcc_kd2_seed42/best_checkpoint.pt"
python -m LightGenV2.tasks.t03_saliency.recheck_aligned --system optical --config "$CONFIG" --checkpoint "$CHECKPOINT" --run-dir "$TASK/runs/simulation/aligned_recheck_20260910_kd2_ablation_normal" --batch-size 32 --ablation none
python -m LightGenV2.tasks.t03_saliency.recheck_aligned --system optical --config "$CONFIG" --checkpoint "$CHECKPOINT" --run-dir "$TASK/runs/simulation/aligned_recheck_20260910_kd2_remove_no_ccd" --batch-size 32 --ablation remove_optical
```

报告的`checkpoint_sha256`绑定实际读入的字节。示例候选应为：
`87ad4db51e3f58f9a41d6df09092439e5a09e81e93f88a3bfe2d3d008fafb29a`。
若活动训练已更新best，请选择新目录，并明确这是另一个候选；不可沿用旧SHA/指标。
两份`reproduction.json`中的数据ID SHA也须一致，逐图差异按`per_image_cc.csv`中的sample_id连接。

报告CC绝对差`normal-removed`，若给相对下降则为`100*(normal-removed)/normal`。
相对下降不是“光学参数/功耗/信息贡献占比”，也不能由消融断言光学硬件优于重新训练的纯电子网络。
不测速度/功耗，不将旁路结果填入普通D2NN baseline。

正式数值待入口完整复评写入；临时终端诊断不替代含逐图指标和SHA的正式产物。

## 首次独立旁路发现的接口问题

初版入口0c185bae在全新进程运行去光时，被继承的LSP `forward` 拒绝：
`Vision global CCD readout is unavailable`。正常预测`head(spatial)`不消费该CCD返回值，
但旧接口仍要求它存在；先正常再去光的同进程诊断会残留旧CCD缓存，不能据此宣称独立旁路验证通过。
失败目录`aligned_recheck_20260910_kd2_ablation_remove`保留，不包含成功的reproduction.json，不能作结果引用。

修复限定于T03的去光forward：主动清空旧CCD诊断值，保持电子latent/读出计算，第三返回值明确为None。
不伪造全零CCD，不执行一次“隐藏的正常光学预热”；正常模式直接调用原父类forward，不改正常推理。
新增测试覆盖全新进程等价的无CCD状态、旧NaN缓存、变化batch大小和正常路径直接委托。
去光模式的调用方必须接受“没有CCD诊断图”；它不是硬件CCD为空的容错开关。
