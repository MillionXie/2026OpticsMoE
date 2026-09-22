# DeepSeek-VL2-Tiny 冻结主干 LGVQ baseline（2026-09-22）

## 结论

在固定 LGVQ prompt-group 2250/558 划分上，冻结完整 DeepSeek-VL2-Tiny
Vision、projector 和 MoE language backbone，只训练 5×1280 个质量词输出参数，得到：

| 任务 | DeepSeek-VL2-Tiny SRCC | PLCC | KRCC | RMSE | best epoch | Qwen3-VL-2B SRCC | SRCC 差值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Temporal | **0.5688169** | 0.5810655 | 0.4000400 | 11.6038570 | 50 | 0.7663 | -0.1975 |
| Spatial | **0.5974811** | 0.6079283 | 0.4194471 | 9.0147915 | 48 | 0.6908 | -0.0933 |

这是首个可复现结果，不是超参数搜索。它说明在当前“最终语言 token + 五质量词行”读出
协议下，DeepSeek-VL2-Tiny 的空间质量迁移明显好于时间质量，但两项均低于当前 Qwen
基线。模型总参数为 3,370,501,440；运行时确认 backbone 可训练参数为 0，每个任务实际
训练参数为 6,400。

## 配对协议与不可混淆项

- 与 Qwen 相同：四帧位置 10%/37%/63%/90%，短边 65% 中心正方形裁剪，OpenCV
  `INTER_AREA` resize，相同 temporal/spatial prompt，相同 2250/558 样本身份，五档等宽
  硬标签，AdamW、50 epoch、batch 512、lr 0.001、余弦衰减、seed 42、按完整 test SRCC
  选 epoch。沿用 Qwen 协议意味着 test 参与选模，不能称为未触碰 test。
- DeepSeek-VL2 不是原生视频模型，四帧按时间顺序作为同一对话内四个 `<image>` 输入；
  Qwen 使用原生 video message。该差异是模型能力边界的一部分。
- DeepSeek 使用官方原生 384×384 前端；Qwen 基线为 448×448。因此这是任务/训练协议
  配对，而不是完全相同的视觉 token 几何。表格中必须保留各自分辨率。
- 五个质量词在 DeepSeek tokenizer 中全部是单 token：Bad 44379、Poor 82916、Fair
  72572、Good 17259、Excellent 81229；五行均直接复制原生 LM-head 行初始化，没有使用
  多 subtoken fallback 或随机初始化。
- 两种 prompt 的所有 2,808 条序列长度均为 1,743；冻结特征均为 `[2808,1280]`。
- 本轮没有做独立端到端 latency/power benchmark。下述 extraction wall time包含视频解码、
  处理器和特征写盘，不能与 Qwen 的 Vision block-0→score CUDA latency 直接比较。

## 可复现身份

- 运行 Git commit：`4178b761d2dc4b48ba753a9d91235297d9148e8a`，启动时 worktree clean。
- 配置：`configs/baselines/deepseek_vl2_tiny_lgvq_4f_r384.yaml`；SHA256
  `cc8224f67be768491282019fe96811af97b833f6ef4aa09cf9adcd1d64d31b00f`。
- LGVQ manifest SHA256：`607c50d20662a47795c23cd20380811b30368ed108ba9be8a1bb8a9d250f8e7fc`。
- 官方模型权重 SHA256：`cc1e5047280253e224b299677bea16b7960c2436ec676d28911ebd9de3bb0074`；
  `config.json` SHA256：`377e10dceb803d9a843d9825d1aa7cba8121c2b825888c632ecf807d0834ce97`。
- 初始五行 SHA256：`66696e2966da5db4afb82a22ec55fab998e1a29fca64e7c0b6f8ade262150123`。
- Temporal feature SHA256：`a5bd55f24e5d8b9d9f7b8047b2899338ba18fc899892e1ee762a19c2eefffd6c`；
  提取 1,077.8515 秒。
- Spatial feature SHA256：`105662dee85510ea46748e205e768d91c7fcdf23ed9cc015cb5e9971c8243bae`；
  提取 1,026.9855 秒。
- Temporal best checkpoint SHA256：`6540addb760cc8bf61ff3f83af75d162fa2dd2ebbf29f938a0762767d2b0aef6`；
  训练后五行 SHA256：`2b77bdd3aa316b98dd384127c8874c231d0d4d502e2995ddd8545d443a0442cde`。
- Spatial best checkpoint SHA256：`e2a195c391acb1669359cc1ddb6f20dda2ca46f6a0390b382059a06127096402`；
  训练后五行 SHA256：`2076bdabe8a820e9d31e8edecfeb8e579e98c6df07401b9ffc98b0b245b31857d`。
- 环境：Python 3.11.15、PyTorch 2.6.0+cu124、CUDA 12.4、Transformers 4.38.2、
  xFormers 0.0.29.post2；单张 RTX 4090。结束后该卡为 12 MiB / 0% utilization。

服务器运行目录为
`LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/deepseek_vl2_tiny_lgvq_4f_r384`。
其中只保留两项的 best/last checkpoint、训练历史、逐视频 test 预测、冻结特征与 32 条一组的
断点 shard；这些大产物受 `.gitignore` 管理，不提交 Git。入口实现为
`deepseek_vl2_quality.py`，运行方式见任务 README。
