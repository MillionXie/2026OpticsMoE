# Qwen3-VL LGVQ Temporal 抽帧速度实验记录

日期：2026-09-04

服务器：NVIDIA GeForce RTX 5090 D，32,607 MiB

模型：`Qwen3-VL-2B-Instruct`

软件：PyTorch 2.8.0+cu128，Transformers 4.57.6，SDPA

## 1. 这次到底测的是什么

一个视频只输入一次 Temporal prompt，并只输出一个连续时间质量分数，不存在所谓“双
prompt 时间”。逐视频计时从打开原始 MP4 开始，到 `Linear(2048,1)` 的标量回到 CPU
结束，包含：

```text
逐目标帧随机 seek + 解码
→ 65% 中心裁剪和 448×448 缩放
→ Qwen processor 与 tokenizer
→ CPU→GPU
→ 完整 Vision tower + learned main merger + Language backbone
→ 最后有效 token + Linear(2048,1)
→ CPU 标量
```

模型加载和两条 shape warmup 样本不计入逐视频时间。模型加载本次为 1.330 s。

## 2. 没有使用的加速

- 没有顺序解码整段视频；
- 没有多线程解码；
- 没有帧缓存或 Qwen feature cache；
- 没有量化、`torch.compile` 或手动 FlashAttention override；
- batch size 为 1 个视频；
- 每个目标帧都执行原工程相同的 `CAP_PROP_POS_FRAMES + read()`。

完整 Qwen 共 2,127,532,032 个参数，全部冻结。输出边界只有一个
`Linear(2048,1)`，共 2,049 个参数；本实验只测速度，不使用其数值计算性能指标。

## 3. 正式全量结果

558 个 test 视频全部参加；每个视频在 4、9、16、25、36、49 帧下各测试一次，共
3,348 条原始记录。
主表中所有时间已经是“抽完这一档全部帧后的每视频总时间”，不是单帧时间。

| 抽帧 | 总耗时均值 ms/video | 中位数 | P95 | 随机seek+解码 | Processor | CPU→GPU | Qwen主干 | Pool+Linear | video/s | 峰值显存 GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 134.058 | 95.009 | 378.928 | 89.364 | 4.917 | 1.657 | 37.895 | 0.214 | 7.459 | 4.052 |
| 9 | 278.559 | 204.352 | 737.544 | 188.412 | 21.985 | 10.451 | 57.458 | 0.242 | 3.590 | 4.169 |
| 16 | 476.591 | 344.483 | 1277.810 | 348.603 | 36.535 | 8.137 | 83.054 | 0.252 | 2.098 | 4.283 |
| 25 | 773.266 | 565.622 | 1929.744 | 557.863 | 62.112 | 10.604 | 142.432 | 0.243 | 1.293 | 4.479 |
| 36 | 1133.494 | 832.388 | 2776.558 | 853.029 | 81.342 | 14.105 | 184.746 | 0.259 | 0.882 | 4.678 |
| 49 | 1532.186 | 1144.709 | 3732.220 | 1154.196 | 106.086 | 18.664 | 252.985 | 0.243 | 0.653 | 4.945 |

相对4帧：9帧端到端约2.078倍，16帧约3.555倍。Qwen主干本身分别约为4帧的
1.516倍和2.192倍；端到端增长更快，是因为按要求保留了逐位置随机seek。

Qwen 序列长度也随输入增长：4帧为446 token，9帧为1,058 token，16帧为1,670
token，25帧为2,690 token，36帧为3,710 token，49帧为5,138 token。线性层约0.2 ms，
不构成主要成本。

36 帧平均总耗时已超过预设的 910 ms/video，因此第一轮按条件没有继续测试 49 帧。
用户随后明确要求补测，故正式结果现已包含 25 帧和 49 帧。25 帧均值为 773.266 ms，
低于 910 ms；49 帧均值为 1532.186 ms。详细解释见 [TIMING_REPORT.md](TIMING_REPORT.md)。

## 4. 为什么P95明显高于中位数

LGVQ并非每段都约96帧。test的源文件分布为：8帧93个、16帧186个、24帧93个、
36帧93个、96帧93个。随机seek的成本随源视频长度明显增加：

| Qwen输入帧数 | 源8帧均值 | 源16帧均值 | 源24帧均值 | 源36帧均值 | 源96帧均值 |
|---:|---:|---:|---:|---:|---:|
| 4 | 92.729 | 77.371 | 84.006 | 130.996 | 341.874 |
| 9 | 183.877 | 166.159 | 181.597 | 289.322 | 684.240 |
| 16 | 287.663 | 286.896 | 309.924 | 529.315 | 1158.849 |
| 25 | 462.788 | 496.215 | 524.321 | 884.445 | 1775.609 |
| 36 | 641.030 | 746.081 | 775.757 | 1382.821 | 2509.190 |
| 49 | 862.716 | 1032.448 | 1075.876 | 1838.861 | 3350.767 |

高帧数配置会向Qwen输入指定数量的图，但在短视频上，时间比例换算后的整数帧号可能
重复。4/9/16/25/36/49 帧配置对应的平均实际唯一帧号数分别为
4.0、8.5、13.333、17.0、19.5、21.667。这里没有去重，因为去重
会改变正式工程既有输入合同。

## 5. 三个旧工程的正确理解

- `qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa`：4帧；Vision只复用Qwen
  patch+position前端，不执行24层Vision及learned merger；固定prompt经过冻结Language
  后缓存。
- `qwen3_vl_2b_lgvq_o2_109_highalpha_vqa`：4帧；执行完整24层Vision和learned main
  merger，并缓存完整Qwen边界。这是三个工程中Qwen视觉复用最完整的一版。
- `qwen3_vl_2b_lgvq_single_metric_o2_16frame_54`：Temporal主线16帧、4×4 lane；只保留
  patch+position和词embedding，不执行Qwen Vision/Language Transformer。该目录后来
  又加入独立的9帧、3×3 compact分支。

因此3×3就是9个视频帧；看起来“超过9”的部分是router和四个feature层的光路pass，
不是额外抽帧。这三个旧工程执行的Qwen深度不同，不能直接拿其总时间比较Qwen帧数。
本次实验固定完整Qwen不变，只改变输入帧数，六档结果才是同口径速度比较。

## 6. 证据

- `results/timing_summary.json`：完整均值、中位数、P05/P95、形状、环境和合同；
- `results/timing_summary.csv`：主表机器可读版；
- `results/per_video_measurements.csv`：1,674条逐视频原始记录；
- `results/run_identity.json`：manifest/config哈希与固定样本ID；
- `results/RESULTS.md`：服务器自动生成的原始摘要。
- `results/per_video_measurements_36.csv`：558 条 36 帧逐视频原始记录；
- `results/timing_summary_extended.json/csv`：4/9/16/36 汇总和 49 帧条件判定；
- `results/RESULTS_EXTENDED.md`：服务器自动生成的追加测试摘要。
- `results/per_video_measurements_25.csv`、`per_video_measurements_49.csv`：各 558 条追加原始记录；
- `results/timing_summary_all.json/csv`：最终六档统一汇总；
- `results/RESULTS_ALL.md`：最终服务器自动生成摘要。

服务器原始目录：

```text
/root/autodl-tmp/qwen3vl_lgvq_temporal_framecount_timing/
```
