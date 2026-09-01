# P12 下游 Fixed-Feedback 50e 命令手册

以下命令均从仓库根目录执行。唯一正式配置为：

```text
experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/configs/base_50e.yaml
```

完整矩阵是 3 tasks x 3 seeds x 4 methods = 36 个 run。每个 run 都训练
50 epochs：9 个 NoFT run 先产生各自的 common start，随后 27 个适配 run
各自再训练 50 epochs。首轮预注册 pilot 先做全部 9 个 NoFT/common start，
但只放行 seed 2026 的 9 个 BP/FA run；通过性能和梯度闸门后，再放行
seed 2027/2028 的 18 个适配 run。

## 1. Git 同步与只读检查

本地代码改完后先提交和 push，再在服务器仓库做 fast-forward 同步。不得
用 reset/checkout 覆盖服务器已有的无关 dirty 文件。

```powershell
git status --short
git push origin main
ssh -p 24096 guest3@202.120.62.181
```

服务器端：

```bash
cd /DATA/DATA1/guest3/2026OpticsMoE
git status --short
git pull --ff-only origin main
git rev-parse HEAD
```

检查候选五卡和实际 compute-app。必须同时看 UUID/PID；低利用率不代表
空闲。

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
```

正式资产检查：

```bash
test -f experiments/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/runs/p11_imagenet1k_pretrain_bs96_90e/checkpoints/backbone.pt
test -f experiments/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/assets/qwen3_vl_static_stem_224.pt
test -d data/Caltech101
test -d data/ISIC2016
test -d data/lsp_pose
```

## 2. 单元测试与隔离 GPU smoke

```bash
/home/guest3/miniconda3/envs/xml/bin/python -m pytest \
  experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/tests -q

/home/guest3/miniconda3/envs/xml/bin/python -m \
  experiments.qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa.run --help

/home/guest3/miniconda3/envs/xml/bin/python -m \
  experiments.qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa.queue --help
```

真实 batch、真实 P11 checkpoint 的 smoke 必须使用独立 `--output-root`，
绝不能指向正式 `runs/p12_downstream_fa_50e`，否则会污染正式 manifest 或
common-start 身份。以下示例在确认物理 GPU 4 空闲后，对 12 种 task/method
组合各执行一个完整训练 batch 和一个验证 batch：

```bash
SMOKE_ROOT=experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/runs/p12_smoke_20260831
for task in caltech101 isic2016 lsp; do
  for method in noft bp fa_pretrained fa_random; do
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
    /home/guest3/miniconda3/envs/xml/bin/python -m \
      experiments.qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa.smoke \
      --config experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/configs/base_50e.yaml \
      --task "${task}" \
      --method "${method}" \
      --seed 2026 \
      --output-root "${SMOKE_ROOT}" \
      --max-train-batches 1 \
      --max-validation-batches 1 \
      --max-test-batches 1
  done
done
```

`smoke.py` 只做单次工程验收并写 `smoke_result.json`。更新组会在隔离的
smoke root 中写一个明确标记为 `synthetic_smoke_only` 的临时 common start，
用于实际走通严格加载链；它不会触碰正式 `runs/p12_downstream_fa_50e`。
必须检查：配置 batch 能装入显存、峰值显存、loss
有限、所有预期梯度非零、BP/FA-pretrained 初始化梯度关系及数据 manifest。
smoke 指标没有科学性能含义。

## 3. 五 GPU 首轮正式启动

默认物理 GPU 为 `1,2,3,4,5`，common-start seed 为
`2026,2027,2028`，首轮 adaptation seed 仅为 `2026`。Linux 包装器通过
`nohup` 启动依赖感知队列并保存 wrapper PID/log：

```bash
bash experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/commands/p12_downstream_fa_50e.sh launch
```

核心队列调用为：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
/home/guest3/miniconda3/envs/xml/bin/python -m \
  experiments.qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa.queue \
  --config experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/configs/base_50e.yaml \
  --gpus 1,2,3,4,5 \
  --seeds 2026,2027,2028 \
  --adaptation-seeds 2026 \
  --python /home/guest3/miniconda3/envs/xml/bin/python \
  --repo-root /DATA/DATA1/guest3/2026OpticsMoE \
  --poll-seconds 20 \
  --max-retries 2
```

队列的安全与依赖约束：

- 每次派发前读取 `nvidia-smi --query-compute-apps`，只使用没有外部
  compute process 的候选卡；
- 每张物理 GPU 同时最多一个 P12 worker，并显式设置
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` 和对应可见设备；
- 先并行安排不同 task/seed 的 NoFT；合法 `common_start.pt` 完成后，
  相应 task/seed 的 BP/FA 才能进入 ready；
- 已有 `result.json(status=complete)` 的 run 被跳过；未完成 run 默认从
  `last.pt` resume；失败最多按统一策略重试两次；
- 队列有单例锁，状态写入 `<output_root>/queue/queue_state.json`。每个 run
  另有 `process.json` 和 `logs/attempt_XX.{stdout,stderr}.log`；
- 某卡后来被别人占用时，不再给它派发新作业。队列尾部若少于五个 ready
  job，可以自然少于五卡，不能为“满卡”破坏依赖或在一卡塞多个 run。

本地 Windows 可用同一包装器远程启动。脚本不保存密码，SSH 会在需要时
提示输入：

```powershell
powershell -ExecutionPolicy Bypass -File `
  experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/commands/p12_downstream_fa_50e.ps1 `
  -Action Launch
```

## 4. 状态、GPU 与日志

结构化队列状态：

```bash
bash experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/commands/p12_downstream_fa_50e.sh status
```

wrapper log 与 GPU 快照：

```bash
bash experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/commands/p12_downstream_fa_50e.sh tail
```

Windows 远程查看：

```powershell
powershell -ExecutionPolicy Bypass -File `
  experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/commands/p12_downstream_fa_50e.ps1 `
  -Action Status
```

同时直接核对 GPU compute process，不能只看队列自报状态：

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
```

单个 run 的完成必须同时满足：

1. `result.json` 存在且 `status=complete`；
2. `epochs_completed_this_run=50`，history 有 50 个完整 epoch；
3. `best.pt` 与 `last.pt` 可加载；
4. source、manifest、config 和适配组 common-start 的 SHA 齐全；
5. NoFT 的 common 目录有合法 `common_start.pt`；
6. sealed test 与 normal、optical-off、phase-random、
   electronic-skip-off 诊断均完成。

队列退出、GPU 归零或日志暂时不增长都不能单独证明完成。

## 5. Pilot 闸门与三 seed 放大

首轮 18 个 run（9 NoFT + seed-2026 的 9 个适配）完成后，先汇总并检查：

1. 三任务 BP-current 能正常学习，并相对 NoFT 具有可解释的适配空间；
2. common start 的 BP-current/FA-pretrained 初始化梯度 cosine 接近 1；
3. FA-pretrained 与 FA-random 的梯度几何确有分离；
4. 无数据泄漏、NaN、stem 漂移、门下界违规或错误 checkpoint；
5. batch、scheduler 和任务头在四组完全一致。

通过后，不改变配置和已完成结果，只扩大 adaptation seed。待首轮队列已经
退出后执行：

```bash
P12_ADAPTATION_SEEDS=2026,2027,2028 \
bash experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/commands/p12_downstream_fa_50e.sh launch
```

新队列会识别并跳过已经完成的 18 个 run，只调度 seed 2027/2028 尚缺的
18 个适配 run。若 pilot 未过，不应靠盲目扩大 seed 掩盖 BP 或实现问题；
先按 `OPTIMIZATION_LOG.md` 记录诊断和统一 recipe 决策。

## 6. 单 run 受控恢复

方法标识固定为 `noft`、`bp`、`fa_pretrained`、`fa_random`；任务标识固定
为 `caltech101`、`isic2016`、`lsp`。以下示例在确认物理 GPU 4 无人使用
且对应 common start 已完成后，恢复 Caltech seed-2026 的 BP run：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
/home/guest3/miniconda3/envs/xml/bin/python -m \
  experiments.qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa.run \
  --config experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/configs/base_50e.yaml \
  --task caltech101 \
  --method bp \
  --seed 2026
```

直接单 run 会绕过五卡队列的安全占用和依赖调度，只用于审计或受控恢复。
运行器默认 resume。`--no-resume` 只能用于全新、且旧目录已另行归档的实验
identity，不能在已有正式 run 上使用。

## 7. 汇总

任意阶段都可生成当前结果的机器可读汇总；缺失 run 应显示为
pending/missing，不能填零：

```bash
bash experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/commands/p12_downstream_fa_50e.sh summarize
```

核心调用为：

```bash
/home/guest3/miniconda3/envs/xml/bin/python -m \
  experiments.qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa.summarize \
  --config experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/configs/base_50e.yaml \
  --seeds 2026,2027,2028 \
  --adaptation-seeds 2026 \
  --output experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/runs/p12_downstream_fa_50e/summary.json
```

首轮保留 `--adaptation-seeds 2026`，因此 expected run 数为 18。正式放大后
将其改为 `--adaptation-seeds 2026,2027,2028`，expected run 数为 36；
包装器会自动使用当前 `P12_ADAPTATION_SEEDS`。

当前 `summarize.py` v1 负责完整性、逐 run 主指标/NoFT 差值、aggregate
mean/sample std 和 CSV，不应把它描述成完整论文统计器。三 seed 完成后，
paper-facing 汇总还需补充逐 seed 配对差、配对 bootstrap CI、FA 的
BP-gain recovery、次指标、逐层梯度 cosine/norm ratio、phase 漂移、
光学/电子门、破坏性消融和吞吐。若 BP-current 不优于 NoFT，recovery
标为不稳定，不能强行给出支持性百分比。

## 8. 记录纪律

- 启动、恢复、失败、OOM、recipe 变更和完成均追加到
  `../OPTIMIZATION_LOG.md`，保留时间、commit、GPU UUID/PID、epoch、
  checkpoint/hash 和处置。
- 若因 OOM 下调某任务 batch，必须对该任务四组统一修改并形成新 config
  digest，不能只降低一个 FA 组。
- 单 epoch 日志只是中间值；正式数字来自验证最优 checkpoint 的 sealed
  test 评估。
- 每次代码修改均先 Git 提交并同步服务器，再启动/恢复；命令保存在本
  目录，不依赖 shell history。
