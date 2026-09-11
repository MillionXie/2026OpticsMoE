# LSP alpha≥0.4：冻结低alpha教师热图蒸馏

## 起点和约束

上一轮40轮小步精修完成，best epoch5的1000张test PCK=0.72792857，
PCKh=0.84621429，alpha=0.41814655/0.41812393；尚未达到0.73。
同权重去光PCK=0.70357143，5张随机global平均0.72318571。
PCK仅比来源提高0.001，PCKh/NME略差，不宣称全面改善。
证据：`runs/simulation/alpha40_polish_seed42_20260910/final_report.json`和
`global_noise_alpha40_polish_20260910/final_report.json`。

蒸馏学生从这个best继续，SHA256：
`6519891703432d049d8c0efdbe89a522912d06f607c3d46cfc8c49234dbc8c23`。
教师为旧低alpha PCK=0.73478571的epoch50 EMA，SHA256：
`495b9c2c4e3df15d3715f1ce8f2faea7cb9156275b31103ec684f4e96a328518`。
教师和学生历史均使用周期test选模，应披露这种选择偏差。

学生推理完全不变：电子836248参数，alpha硬范围[0.4,0.95]，保留来源logit；
Top2光router、相位/噪声/10cm光路/478有效场、姿态头均不改。
不加层、attention、测试时增强或后处理。教师不保存进学生checkpoint，不部署。

## 教师缓存与损失

仅对10428张train做canonical crop，教师eval/no_grad，一次生成FP32 [10428,14,56,56]
热图。存于新run的`training_cache/teacher_heatmaps.npy`，约1.83GB；这不是额外模型checkpoint。
记录教师SHA、缓存SHA、每行sample ID、图像路径和关键点摘要SHA；强制检查train/test ID无交叉。
不缓存test教师输出，不增加任何人工标签。导出结束释放教师，只保留CPU热图。
每个训练batch按dataset index取缓存，通过原有warp函数匹配随机crop/flip及左右关节置换。

复用现有可见关节masked heatmap MSE：
`L = 原GT热图MSE + 原路由/相位约束 + lambda * MSE(student, warped_teacher)`。
GT权重不下降，原coordinate loss权重仍0；不使用空间softmax/KL。
lambda从1.0线性减到0.2，避免后期只跟随教师误差。
前30轮与上一轮精修同LR日程，后10轮固定光学/主支，只微调原CCD读出/头。
router探索噪声从来源第5轮对应的原始7/12强度起递减，不改变测试过程。

启动时保存`distillation_gradient_audit.json`：在4张训练图上、不做更新，分别测GT/KD
对电子/相位/router/CCD/头的梯度范数，要求KD到相位梯度非零且有限。
每epoch记录实际KD权重与原始蒸馏损失。17um/原噪声与损失保留，未把更干净仿真冒充鲁棒提升。

## 命令

干净源码worktree中运行，先检查GPU空闲；只用一张GPU：

```bash
ROOT=/DATA/DATA1/guest3/2026OpticsMoE
TASK=$ROOT/LightGenV2/tasks/t02_keypoint_detection
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
/home/guest3/miniconda3/envs/xml/bin/python -m LightGenV2.tasks.t02_keypoint_detection.refine \
  --profile alpha40_distill --seed 42 --batch-size 24 --workers 4 \
  --source "$TASK/runs/simulation/alpha40_polish_seed42_20260910/best_checkpoint.pt" \
  --teacher "$TASK/runs/simulation/refinement_20260909/staged_heatmap/best_checkpoint.pt" \
  --data-root "$ROOT/data/lsp_pose" --cache-dir /DATA/DATA1/guest3/.cache/huggingface/hub \
  --run-dir "$TASK/runs/simulation/alpha40_distill_seed42_20260911"
```

新目录不可已存在；`--smoke --batch-size 4 --workers 0`仅检查4张训练/4张测试，不算正式性能。
完整train10428/test1000，40epoch，每1/5/末轮评估并按PCK选EMA best，epoch0作为回退候选。
完成自动重载best、去光测试；随机global用`global_noise --profile alpha40`，推理合同相同。
结果在final_report中标记target_met，达不到0.73如实报告，不修改指标或alpha下限。
