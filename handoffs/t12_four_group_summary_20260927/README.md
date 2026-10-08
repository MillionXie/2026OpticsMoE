# T12 四组最终结果（2026-09-27）

统一TEST2304条，256×256原生输出，RGB[0,1]逐图计算PSNR再平均；SSIM仅辅助。每种编辑模式768条，灯、桌、靠垫各有样本。有限商品目录和程序合成场景任务，不宣称开放域文生图。

| 模型 | 参数/固定条件预算 M | PSNR↑ dB | SSIM↑ |
|---|---:|---:|---:|
| 小版原权重仿真 |17.026642|34.277751|.927140|
| 小版电子decoder微调后实测 |17.026642|31.552886|.906013|
| 大版仿真 |149.755866|31.428599|.882211|
| Qwen28+decoder同任务补训baseline |1834.345963|27.260453|.837653|

**主表小版两行不是同权重对照。**原权重5b4f的仿真/实测为34.277751/27.551289dB；微调eeec的仿真/实测为28.890612/31.552886dB。微调仅改电子decoder，光学上游未变。微调干净仿真有所下降，不能声称31.55实测同时保住34.28干净仿真。小版微调原报告保留validation warm-start caveat，详见原始报告。

## 图片从哪里看

新增 `baselines/`：Qwen28 与 pix2pix-Turbo 两组完整 TEST 输出已放在同一子目录，各 2304 张；使用说明见 `baselines/README.md`。原有四组结果与指标未改动。

- `figures/matched_background.png`：换背景，Input/GT/小版EXP/大版SIM/baseline，同一样本。
- `figures/matched_object.png`：换目标，同上。
- `figures/matched_joint.png`：同时修改，同上。
- `small_exp/`、`large_sim/`、`baseline/`：相同整理形式，每目录有三张overview和逐图CSV/JSON。
- `small_sim/`：原小版仿真逐图结果。
- 每个模型 `images/reference/`、`images/target/`、`images/generated/` 各2304张原生256 PNG，文件名test_00000至test_02303；prompt与身份查对应sample_metrics.csv。
- `figure_indices.json`：每类别×模式取首个和中间样本，固定18个，无质量筛选。

输入/GT在四个目录统一使用同一导出；小版原始导出floor、大版导出round，二者GT/输入差不超过1/255。生成PNG均保留原生结果，不锐化、不超分、不贴回GT。不包含原瘦包未提供的微调后仿真PNG；其28.89仅引用完整2304条浮点指标。

## 指标、训练与身份

`summary.json`、`all_test_metrics.csv/json` 为完整汇总与逐图指标；`integrity_audit.json` 校验ZIP CRC、SHA、2304条身份/prompt/模式及PNG大小/哈希。指标来自PNG量化前的浮点输出，不用整体MSE取log替代逐图平均PSNR。PSNR衡量与指定GT的保真度，不能单独说明画面真实感或生成创新性。

baseline保留完整28层Qwen和原电子BK-SDM tiny UNet、adapter、SD-Turbo VAE；Qwen/VAE冻结，UNet/adapter用当前TRAIN20736补训3epochs，VAL2304按PSNR选择step15000，然后固定权重测TEST。best VAL27.404973dB，不是最终TEST指标。原baseline历史权重未覆盖。四组训练历史/损失不完全相同，结果不支持将优势单独归因于光。

预算含固定条件缓冲值；冻结词嵌入311.164928M按约定另列排除，不是无参数tokenizer。大版NN144.630618M+固定5.125248M；baseline NN1824.174315M+固定10.171648M。没有测本轮新延迟，不沿用旧9.96M小版速度作为17M速度。

权重SHA256：

- 小版原仿真：5b4f9a37f19ce95cf23e4b874badc8e88955559553d47929fb80cd3cbf527cac
- 小版微调实测：eeeca764b11a23613459fdae0a190fdba362a6e3bdeed9180279d9031c297a14
- 大版：2a91d8129253c0eb060ae88a3f3f671ef51e2a6e153d95e8828962c63ee0c5b5
- baseline：599805ad3062bebf67acf3b515f0a812fb843506643e18251004f180131b4b52

源码训练b813628ea、固定权重导出c657f959d、结果文档7c533b521，分支codex/t12-physical-robust-v2-20260927，已同步GitHub和服务器。服务器工作树 `/DATA/DATA1/guest3/2026OpticsMoE/.worktrees/t12_physical_robust_v2_20260927`，权重在任务runs/simulation/20260927_matched_qwen28_baseline/best_checkpoint.pt；完整导出run为20260927_four_group_matched_baseline。PyTorch2.6.0+cu124、RTX4090。

本任务仅一张GPU串行训练/导出，自身进程已结束，未操作真实光路或停止其他AI进程。详细复现说明：隔离工作树LightGenV2/tasks/t12_text_to_image/reports/reproduction/README.md。
