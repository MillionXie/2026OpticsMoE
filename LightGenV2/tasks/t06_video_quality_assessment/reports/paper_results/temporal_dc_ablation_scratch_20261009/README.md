# Temporal 光学 FFN 残差：同一随机初始化从头训练

2026-10-09用户修正实验口径：只继承原 `.8044` 架构，不从旧学生PT继续训练。
本页独立于上一轮固定PT干预/续训报告，不把两者的成绩混作从头训练结果。

## 两组合同

| 设置 | 有光残差 dc20 | 无显式光残差 dc0 |
| --- | --- | --- |
| 学生初始化 | seed163，同一随机状态 | seed163，同一随机状态 |
| 旧学生PT加载 | 无 | 无 |
| 训练rho | 固定.20 | 固定0 |
| 测试rho | 固定.20 | 固定0 |
| 训练轮数 | 100 | 100 |

不在同一次训练中切换开关。两组始终开启完整光分支，均保持原光Router Top2、
16视频×4帧、478有效面积、六次全场传播、原电子残差与读出头。
“光学FFN残差”在本项目中采用既有的
`sqrt(1-rho)*exp(i*phi)+sqrt(rho)` 光场混合公式，探测/归一化流程保持不变。
原有phase-dropout、k-space、位置扰动等训练设置均相同，不额外改计算图。

冻结Qwen输入前端及其已核验缓存保留；它们不是学生已训练权重。
新入口从原审核 `resolved_config.json` 读取架构配置，完全不读取学生checkpoint PT。
旧的 `initialization_checkpoint` 强制清空。两组记录全部初始参数/buffer SHA256，
必须一致，不能仅凭相同seed宣称初始化一致。额外初始PT不保存，只保留best/last。

TRAIN2250/TEST558、prompt、输入缓存、原损失（含训练集教师软目标）、batch16个物理场
和训练顺序相同；未新加学生分支或参数。使用从头训练学习率：电子.0003、相位.02、
Router相位.032，余弦退火。原mask初始化方法保留，各组构建后再以同一seed重置训练RNG。
固定rho不消耗随机rho抽样，因此两组无需上一轮随机rho的额外RNG补齐。
每5epoch测试并选best，另保存第100epoch last；无VAL，TEST选模属于开发指标。

## 命令（服务器主仓库根目录）

源代码只能经Git同步main，最多两张当前空闲GPU。原audit已完成全558测试输入
逐元素缓存验证，此处复用它的数据身份，不复用原学生参数。

```bash
cd /DATA/DATA1/guest3/2026OpticsMoE
PY=/home/guest3/miniconda3/envs/xml/bin/python
TASK=LightGenV2/tasks/t06_video_quality_assessment
AUDIT=$TASK/runs/simulation/temporal_dc_ablation_audit_20261009
CACHE=/DATA/DATA1/guest3/2026OpticsMoE/.worktrees/t13_temporal_robust_20260927/LightGenV2/tasks/t13_temporal_robust_training/assets/cache/rebuilt_4f_20260927

CUDA_VISIBLE_DEVICES=0 $PY -m LightGenV2.tasks.t06_video_quality_assessment.dc_ablation \
  --phase train --initialization scratch --arm dc20 \
  --architecture-config "$AUDIT/resolved_config.json" --audit-output "$AUDIT" \
  --output "$TASK/runs/simulation/temporal_dc_ablation_scratch_dc20_s163_20261009" \
  --vision-cache-path "$CACHE/vision_49x1024_quality14.pt" \
  --language-cache-path "$CACHE/language_temporal_2048.pt"

CUDA_VISIBLE_DEVICES=1 $PY -m LightGenV2.tasks.t06_video_quality_assessment.dc_ablation \
  --phase train --initialization scratch --arm dc0 \
  --architecture-config "$AUDIT/resolved_config.json" --audit-output "$AUDIT" \
  --output "$TASK/runs/simulation/temporal_dc_ablation_scratch_dc0_s163_20261009" \
  --vision-cache-path "$CACHE/vision_49x1024_quality14.pt" \
  --language-cache-path "$CACHE/language_temporal_2048.pt"
```

输出目录存在时拒绝覆盖，scratch模式传 `--checkpoint` 时直接拒绝。
各run保存 `initialization_identity.json`、配置、数据SHA、命令、commit、环境、状态、
逐样本预测、best/last；最终同时报告两组best与last指标，不将旧.8044成绩填作新结果。
原正式PT、原消融与续训数据不改、不删。

## 完成结果

两组均从头训练100epoch，训练源码commit
`c1d616f6ecfd22cb04b71785bd1eab9ab60e5834`。相关CPU测试7项通过。
实际配置仅输出目录和三项rho参数不同；二者共同初始state SHA256：
`86c80d5fbf538730f58b3d0893dde1e7d006fc880fe56ba85fb7dc251ce179c5`。
各自 `initialization_identity.json` 均记录 `pretrained_student_weights_loaded=false`。

| 从头训练条件 | 最佳epoch | 最佳SRCC | 最佳PLCC | 最佳RMSE | 第100epoch SRCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 有光学FFN残差，训练/测试rho=.20 | 100 | .7970022913 | .8036683088 | 8.353568 | .7970022913 |
| 无显式光学FFN残差，训练/测试rho=0 | 75 | .8003919178 | .8102487203 | 8.327778 | .7974375292 |

最佳SRCC的有减无差值为 `-.0033896265`，第100epoch差 `-.0004352378`。
最佳PT逐视频配对bootstrap（2000次，seed163）差值95%区间
`[-.0226477797, .0146697694]`，包含0。单seed、TEST选模的开发实验：
当前只能说性能接近，本轮无残差组略高，不能声称某一条件显著更好。
这不是旧PT续训，不将原 `.8044` 纳入此表，也不替换原部署权重。

服务器权重均在本任务 `runs/simulation/` 下：

- `temporal_dc_ablation_scratch_dc20_s163_20261009/best_checkpoint.pt`：
  SHA256 `2746a94ddd06b8d1ab355ebb9ed1913b1eb83f2f48faddfb0a94a05da0dccc9d`。
- `temporal_dc_ablation_scratch_dc0_s163_20261009/best_checkpoint.pt`：
  SHA256 `d027a52e2abd249e52d9dc2d9501967c1f563fec0a2f38f128a43522527b8e48`。

两份status均complete，只保留best/last两份PT及必要指标、逐视频预测。
训练PID2825126/2825127的 `/proc` 条目均消失，`nvidia-smi` 中无对应CUDA进程；
GPU资源随训练退出释放，未停止其他用户/任务的进程。
