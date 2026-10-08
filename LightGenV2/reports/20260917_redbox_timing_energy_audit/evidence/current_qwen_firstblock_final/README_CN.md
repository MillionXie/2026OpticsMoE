# 最新 Qwen3-VL baseline：A100 正式性能、速度、功率与能耗

## 主结果

| 任务 | 性能 | 单次/批次 Wall mean | 平均功率 | 单次/批次能耗 | 等效16视频时间 | 等效16视频能耗 |
|---|---:|---:|---:|---:|---:|---:|
| LGVQ temporal quality, batch=1 | SRCC=0.7663 | 53.160 ms | 160.545 W | 8.535 J | 850.556 ms | 136.553 J |
| LGVQ temporal quality, batch=2 | SRCC=0.7664 | 63.466 ms | 198.540 W | 12.600 J | 507.724 ms | 100.804 J |
| LGVQ spatial quality | SRCC=0.6908 | 81.118 ms | 86.180 W | 6.991 J | — | — |
| ABO image-to-text retrieval | R@1=0.7358 | 40.242 ms | 80.685 W | 3.247 J | — | — |
| ABO image-to-image enrolled-SKU retrieval | R@1=0.8513 | 48.628 ms | 71.779 W | 3.490 J | — | — |
| LSP keypoint detection | PCK@0.2=0.7226 | 16.971 ms | 63.076 W | 1.070 J | — | — |
| SALICON saliency | CC=0.8748 | 20.038 ms | 54.963 W | 1.101 J | — | — |
| OpenMoji semantic interaction | changed-cell accuracy=0.8420 | 47.147 ms | 71.428 W | 3.368 J | — | — |

LGVQ 时间质量按相同的 **16 个视频、每视频 4 帧** 工作量换算：batch=1 需要16次调用，batch=2需要8次调用。换算不是把单视频延迟除以batch，而是 `每批实测均值 × ceil(16/batch)`；能耗同理。

## 统一计时边界

正式时间均使用 **Synchronized Wall mean**：在第一个原生 Vision Transformer block 的 pre-hook 开始，到任务最终结果已经在GPU上生成并完成CUDA同步为止。包含所有后续必须执行的Vision/Language block及任务读出；不包含模型加载、文件I/O、图像/视频解码、processor/tokenizer、H2D，以及第一个block之前的patch embedding。

计时不是整段数据加载耗时，也不是只看CUDA Event。每次调用的CUDA Event仍保留在原始CSV中作审计。

## 功率与能耗

单次/批次能耗按 `E = P_active_mean × T_wall_mean / 1000` 计算，单位J。功率是A100整卡 `power.draw`，不是仅任务头功率，也不是250 W额定上界。额定上界能耗另存于CSV。

2026-09-14新测的T02/T03/T04/T07/T08采用独立连续推理功率段，10 ms采样，同时保存GPU利用率、显存和SM时钟；正式计时段不运行遥测采样器。每个新测报告的process_audit均证明加载前没有其他计算进程，加载后只有测量脚本本身。

T06 temporal使用已有的同一模型A100完整558视频原始记录（含batch=1/2功率、利用率和逐批时间）；T06 spatial使用同一模型558视频原始记录。它们的性能值与当前表格的0.7663/0.6908完全对应，未用旧checkpoint冒充。

## 输入与性能口径

- LGVQ：4帧/视频，每帧448×448；时间/空间分别使用对应五质量词baseline。时间质量batch=1为558批，batch=2为279个完整批次。
- ABO图搜文：RGB 224×224，冻结Qwen3-VL-Embedding-2B，2048维查询，对100个离线标题向量排序。
- ABO图搜图：新 enrolled-SKU 协议，800 query、1600 gallery、每query有8个同SKU正样本；native aspect、50176像素约束、64维前缀，对1600项完整稳定排序。它不是旧的120类中心协议。
- LSP：RGB 224×224，输出14×56×56热图；性能为1000张测试图的PCK@0.2。
- SALICON：RGB 224×224，冻结Vision主干，加197,184参数adapter与85,412参数density decoder；性能为5000张测试图的CC。
- OpenMoji：RGB 224×224 + 原始编辑指令；完整冻结Qwen Vision/Language，训练读出共972,952参数（image adapter 197,184、text adapter 393,792、shared grid readout 381,976），输出6×6类别与编辑网格；性能为1000样本changed-cell accuracy。

## 原始数据

- `fresh/`：五个2026-09-14空卡复测，每项含逐样本时间、10 ms功率/利用率/显存/时钟、进程审计、命令和JSON报告。
- `evidence/T06_temporal_batch1_batch2/`：batch=1/2的558视频逐批时间、预测、预处理轨迹和遥测。
- `evidence/T06_spatial/`：558视频逐视频时间、预测和功率。
- `evidence/performance_sources/`：表中最新版性能的原始评估报告副本。
- `source_snapshot/`：本次计时与汇总脚本；`SHA256SUMS.txt`固定全部证据字节。

图：`qwen_a100_latency_power_energy.png` 与 `lgvq_temporal_batch1_batch2_equivalent16.png`。
