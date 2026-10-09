# Temporal 0.8044：未调制光场对照

## 固定版本，不替换原正式权重

- Temporal：`runs/simulation/multivideo16x4_rank_s163/best_checkpoint.pt`，SHA256
  `5303b574b200e14bf943af21c60a246720be453b9cf8847cd93c0eaaa243a77c`，SRCC `.8043868643`。
- Spatial：`runs/simulation/spatial_readout_1m_srcc067/best_checkpoint.pt`，SHA256
  `95e12397ccf8c960fa30ba9dfb400b69d2c2ebd02288ac6a4879870ab592828b`，
  封存复评 SRCC `.6710968960`（历史训练记录 `.6710079009`）。本次不重训 Spatial。
- Temporal：训练 rho 随机 `[.20,.35]`，推理 `.20`；Spatial：训练/推理均 `.20`。
- 这些是开发期 TEST 选模指标，不声称独立泛化结果。TRAIN2250 / TEST558，无验证集。

## 物理意义与消融边界

原实现位于 `experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/modeling.py`
的 `_phase_modulation`，T06 的 `models/multivideo9x4.py` 在六次传播中调用它：

`U_out = Fresnel[A * (sqrt(1-rho)*exp(i*phi) + sqrt(rho))]`。

rho 是混合前名义功率系数；相干干涉后，CCD 的实际局部能量占比不是固定 rho。
这不是额外一次光传播，也不是电子 alpha；更不是严格的数字 FFN `x+F(x)`：
未调制场同样经过 Fresnel 传播，并与调制场在强度探测前相干叠加。
如要研究严格恒等跳连，须另立合同，不能将当前结果如此命名。

两组保持光学 Top2、16视频×4帧、478有效面积、六次全场传播、所有参数形状和
电子 RMS 同尺度融合不变。rho=0 **仍保留全部光路**，不是去光分支。
本轮也不更改封存版本原有的 k-space / 像素扰动等设置。

1. 固定 PT 干预：同权重对比推理 rho=.20 / 0，先验证原 .8044 复现。
2. 匹配续训：两组从上述同一 PT 严格加载；100 epoch，seed163，batch16个光场，
   电子 LR3e-5、相位 LR.002、Router LR.0032，同一损失、余弦学习率、每5epoch测TEST。
   dc20 训练 rho 随机20%~35%、推理20%；dc0训练/推理均0。
   零系数组补齐原随机rho的随机数消耗，避免仅因关闭rho改变后续扰动序列。
   只保留 best/last PT，不保存每5epoch的mask。
   这是从含DC的已训练权重出发的续训对照，不能冒称两组独立从头训练。

## 服务器执行与数据恢复

从主仓库根运行。不要创建分支、worktree或新独立源码副本。先确认 Git 干净且
本地/服务器 main 同 SHA，再启动；最多两张空闲GPU，只管理自己启动的PID。
下面路径为 guest3 已核实路径。运行输出不进Git，摘要报告进Git。

原缓存路径已失效。后来重建缓存只有在全部558个测试视频的 vision/quality 和
文本输入与原发布包逐元素相等、且 loader 的来源身份及划分检查通过后才允许用于续训。
训练样本来源仍须注明“重建缓存”，不声称其字节与已丢失原文件相同。

```bash
cd /DATA/DATA1/guest3/2026OpticsMoE
PY=/home/guest3/miniconda3/envs/xml/bin/python
TASK=LightGenV2/tasks/t06_video_quality_assessment
CACHE=/DATA/DATA1/guest3/2026OpticsMoE/.worktrees/t13_temporal_robust_20260927/LightGenV2/tasks/t13_temporal_robust_training/assets/cache/rebuilt_4f_20260927
CKPT=$TASK/runs/simulation/multivideo16x4_rank_s163/best_checkpoint.pt
AUDIT=$TASK/runs/simulation/temporal_dc_ablation_audit_20261009
CUDA_VISIBLE_DEVICES=0 $PY -m LightGenV2.tasks.t06_video_quality_assessment.dc_ablation \
  --phase audit --checkpoint "$CKPT" --output "$AUDIT" \
  --package "$TASK/releases/20260914_shs_temporal08044.zip" \
  --vision-cache-path "$CACHE/vision_49x1024_quality14.pt" \
  --language-cache-path "$CACHE/language_temporal_2048.pt"

# 各用一张当前空闲GPU；不要在未完成audit时执行。
CUDA_VISIBLE_DEVICES=0 $PY -m LightGenV2.tasks.t06_video_quality_assessment.dc_ablation \
  --phase train --arm dc20 --checkpoint "$CKPT" --audit-output "$AUDIT" \
  --output "$TASK/runs/simulation/temporal_dc_ablation_dc20_20261009" \
  --vision-cache-path "$CACHE/vision_49x1024_quality14.pt" \
  --language-cache-path "$CACHE/language_temporal_2048.pt"
CUDA_VISIBLE_DEVICES=1 $PY -m LightGenV2.tasks.t06_video_quality_assessment.dc_ablation \
  --phase train --arm dc0 --checkpoint "$CKPT" --audit-output "$AUDIT" \
  --output "$TASK/runs/simulation/temporal_dc_ablation_dc0_20261009" \
  --vision-cache-path "$CACHE/vision_49x1024_quality14.pt" \
  --language-cache-path "$CACHE/language_temporal_2048.pt"
```

audit生成 `comparison.json`、逐样本 `test_predictions.csv`、数据SHA和缓存恢复核验。
训练各生成 `train_history.json`、best/last、指标和run_manifest。输出目录已存在时拒绝覆盖。
初始权重和数据不上传Git。结论必须区分固定权重干预与匹配续训，不只挑一组最高分。

## 2026-10-09 已完成的固定权重干预

代码提交 `76643e200`，服务器 xml 环境；新物理公式与设置合同测试5项通过。
扩大运行T06整组测试为51通过、2失败；失败均为历史Temporal36/旧Spatial profile
的backend来源hash检查，涉及未改动的旧modeling/settings/training/run/data文件。
README已标注旧profile未通过预检；本次固定PT入口不经这些profile，不绕过旧来源检查。
原完整TEST558发布包原预测复现为 `.8043868643075132`，不是另换一个Temporal版本。

| 同一原始PT，全部光分支开启 | SRCC | PLCC | RMSE | MAE |
| --- | ---: | ---: | ---: | ---: |
| 显式未调制场 rho=.20 | .8043868643 | .8180329374 | 7.99108 | 5.99206 |
| 显式未调制场 rho=0 | .8003881881 | .8134329592 | 8.10588 | 6.13661 |

SRCC差 `.0039986762`。同视频配对bootstrap（2000次，seed163）差值95%区间
`[-.00259043, .01083649]`，含0；当前不能声称增益统计显著，更不能推广成所有任务。
bootstrap只描述本组测试样本的不确定性，不是跨训练seed的误差，也不是独立泛化评估。

原四个同尺度融合alpha为 `.56113535/.55998361/.56789732/.56950724`；
固定PT干预两组完全相同。未调制比例rho与这些alpha必须分开报告。
训练仍保留原共同的5% phase-dropout、原k-space和位移扰动设置；dc0只关掉显式
相干混合项，不能说训练中不存在任何零相位像素或背景未调制区域。

重建缓存全部558个测试视频的两类视觉输入、文本输入逐元素核对一致，
TRAIN2250/TEST558与冻结前端来源校验通过。恢复资产为只读引用：

- Vision SHA256 `17b29fc12b17268b7d94fb2c6bbead6650d82899b33bcc1c3ee2a8608a64419c`。
- Language SHA256 `57b9f8463a88b5c71a917422bcbbb48d8ac156a16dc460ffccca8a56d10feb4c`。

两组100epoch续训已启动，各自独立进程占GPU0/1，数据加载workers=0，不产生常驻
GPU子进程；退出后CUDA上下文随进程释放。仅保存best/last。训练完成前不填最终成绩。
