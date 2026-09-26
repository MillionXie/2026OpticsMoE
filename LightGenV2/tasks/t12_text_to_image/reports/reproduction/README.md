# T12 最终版本与复现入口

日期：2026-09-26。正式代码分支：`codex/t12-audited-editors-20260926`。本轮训练源代码分别记录于报告 execution 字段；封装提交 bb7bfba2253108ef5da3d81203f4edc7f1510dc9，运行记录提交 15537441f，清理提交 dbda11c62179e62967f5169759437ee35093d978。

## 两套主权重

本地统一目录：
`C:/Users/Xml12/OneDrive/2026OpticsMoE/LightGenV2/tasks/t12_text_to_image/runs/simulation/20260926_consolidated/`

- large.pt / large_test.json / large_audit.json / large_overview.jpg
- small.pt / small_test.json / small_audit.json / small_overview.jpg
- cleanup_server.json：服务器删除清单和哈希。

服务器代码：`/DATA/DATA1/guest3/t12_git_audited_20260926`；任务 runs/simulation 下的 20260926_large_detail、20260926_small_detail 为本次完整训练记录，20260926_sealed_references 保留前一版可加载参考。

| 模型 | 预算参数/缓冲值 | MSE ↓ | PSNR ↑ | 边缘 L1 ↓ | FID变体 ↓ | KID ↓ |
|---|---:|---:|---:|---:|---:|---:|
| 大版 |149,755,866|0.00371867|30.32|0.0859617|35.37|0.005992|
| 小版 |9,958,098|0.00255774|31.94|0.0715071|86.24|0.016663|

统一 256×256、同一 2304 个测试指令对。MSE/PSNR衡量配对像素一致性，不等于纹理真实感；不能用小版更低MSE推出视觉更好。FID使用 torchvision ImageNet Inception-v3，**不是标准 TensorFlow FID**，仅作同协议内部比较。局部细节分数使用训练同类指标，不是独立 LPIPS。baseline 固定权重复评与计时见下节；它未按当前统一任务重新训练，不能用于结构优越性的公平结论。

大版上一轮 MSE 0.004986、PSNR29.04、FID变体48.68；小版上一轮 MSE0.002639、PSNR31.81、FID变体88.58。新大版改善明显，新小版改善有限；小版仍缺乏细纹理并存在局部重影，不能称已完全解决模糊。

## 执行和验证

Python：`/DATA/DATA1/guest3/t12_assets/venv/bin/python`。
入口：`audited_unified_run.py` 的 train/evaluate/audit/seal 子命令。具体完整训练/评估 argv、Git SHA、torch、设备和配置已写入每份结果的 execution 字段；以记录的 argv 复现，checkpoint 指向本目录 sealed 权重即可，不再需要旧 source 权重。

11 项结构测试通过，两套实际 GPU 审计均通过：language/vision真实调用、并行同输入、相位梯度、alpha下限、256输出、参数预算。使用的一张GPU已释放；未停止其他用户进程。

## 2026-09-26 baseline 固定权重速度与性能复评

源代码提交：190de2e494460db33c03c58e9ccbc0610ff5ce8b。run ID：20260926_baseline_comparison，位于任务 runs/simulation；本地同名目录含 timing.json、baseline_quality.json、execution.log。固定现有 baseline，不改变 Qwen+电子 decoder 架构、不重新训练。

GPU RTX 4090，batch=1，FP16 autocast（光学内部FP32），同一32-token指令与输入，256×256；预热10次、重复100次。CUDA事件起点为首个语言block输入，终点为RGB，不包括tokenizer、词嵌入、窄头输入投影/packing、加载和数据传输。是固定样例重复计时，不是全测试集平均延迟。

| 模型 | NN参数 | 固定条件缓冲值 | GPU全仿真均值/P95 ms | 全仿真加速 |
|---|---:|---:|---:|---:|
| Qwen28+电子decoder baseline |1,824,174,315|10,171,648|60.68 / 62.35|1×|
| 大版 |144,630,618|5,125,248|39.89 / 40.94|1.52×|
| 小版 |9,958,098|0|17.96 / 18.32|3.38×|

词嵌入311,164,928各自单列不计。baseline未使用Qwen vision tower/LM head，因此本合同内不是完整Qwen2.13B再加decoder；有效NN约1.824B。加入条件缓冲值，baseline预算1,834,345,963；大/小预算减少91.84%/99.46%。不计未参与推理的辅助分类router。

硬件代理计时使用缓存的router结果及expert/global CCD，保留电子幅度编码、装载、读出及融合；不能用这个代理输出评价质量。额外排除并行电子支路以遵循用户口径，仅为乐观下界，不是硬件端到端实测：

| 口径 | 大版ms / 加速 | 小版ms / 加速 |
|---|---:|---:|
| 去FFT与并行电支路，仅剩串行电子 |32.84|11.02|
| 上项 + 一次6.2682ms（此前约定） |39.10 / 1.55×|17.29 / 3.51×|
| 上项 + 两次6.2682ms（语言→视觉先后经过） |45.37 / 1.34×|23.56 / 2.58×|
| 保留电子并行支路串行GPU时间 + 两次光路 |48.80 / 1.24×|26.93 / 2.25×|

最后一行也不是物理并行实测；真实关键路径需要各阶段 max(T电子,T光)+外围电子，并计入设备传输。加速来自网络压缩及不同计算图，不能单独归因于光计算。

所有质量数值使用完整光学仿真/原baseline，未使用计时bypass、GT贴回或模板检索。同一2304对固定测试：

| 模型 | 整体MSE↓ | 整体PSNR↑ | 换背景PSNR↑（768对） | FID变体↓ | KID↓ |
|---|---:|---:|---:|---:|---:|
| 历史baseline迁移诊断 |0.123046|15.12|21.85|120.94|0.028966|
| 大版 |0.003719|30.32|29.04|35.37|0.005992|
| 小版 |0.002558|31.94|34.46|86.24|0.016663|

**baseline只训练过旧灯类换背景，未训练当前换目标/联合修改/扩展类别。**测试数据相同不代表训练协议相同；以上质量只能说明当前固定权重可用性，不能支持“光电模型公平胜过Qwen baseline”的论文结论。换背景子集也包含类别/数据迁移。公平质量比较仍需要保持Qwen+decoder架构、用同训练集/任务重新训练baseline（本次未做）。

复现命令（服务器代码根目录，AS=/DATA/DATA1/guest3/t12_assets，TASK=LightGenV2/tasks/t12_text_to_image，CUDA_VISIBLE_DEVICES=GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd）：

```bash
$AS/venv/bin/python -m LightGenV2.tasks.t12_text_to_image.benchmark_audited_editors --assets $AS --qwen /DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203 --large $TASK/runs/simulation/20260926_large_detail/adapted_model.pt --small $TASK/runs/simulation/20260926_small_detail/adapted_model.pt --output $TASK/runs/simulation/20260926_baseline_comparison --warmup 10 --repeats 100
```

权重SHA256见 timing.json。数据清单 test.jsonl：331874abbb8d8c7a4ac5ee3d6e030f20611cb5a9e96d3092917b7a5c4d7d5d2f；instruction-cache：66f457115cc7b1058f3ee42be0fec5815f101780c22b4805ad08940966dae97f；embedding-cache：e7a855849ef22e470aafdcf8ee583968ca0c8d5d17ff42603b103c49e97502ac。11项结构测试再次通过，完成后本GPU显存回到15MiB，无自己的计时进程。

## 清理记录

服务器仅删除明确清单内 35 份旧 .pt，4,808,586,899字节，包括旧错误光电模型及不采用的15M实验；保留数据、baseline、历史图/指标、sealed参考。服务器删除不可直接撤销，清单含SHA256。本地旧权重按明确目录送回收站；本目录两份正式权重不删除。

共享根工作树存在其他AI修改与分叉，未 reset、未强推、未全量暂存。独立整合分支是本任务唯一新的代码入口；不可将共享根的旧HEAD当作本轮训练代码。
