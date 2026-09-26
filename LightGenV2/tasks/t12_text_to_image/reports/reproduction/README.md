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

统一 256×256、同一 2304 个测试指令对。MSE/PSNR衡量配对像素一致性，不等于纹理真实感；不能用小版更低MSE推出视觉更好。FID使用 torchvision ImageNet Inception-v3，**不是标准 TensorFlow FID**，仅作同协议内部比较。局部细节分数使用训练同类指标，不是独立 LPIPS。baseline 未在本轮重测，不提供未经验证的速度或性能差。

大版上一轮 MSE 0.004986、PSNR29.04、FID变体48.68；小版上一轮 MSE0.002639、PSNR31.81、FID变体88.58。新大版改善明显，新小版改善有限；小版仍缺乏细纹理并存在局部重影，不能称已完全解决模糊。

## 执行和验证

Python：`/DATA/DATA1/guest3/t12_assets/venv/bin/python`。
入口：`audited_unified_run.py` 的 train/evaluate/audit/seal 子命令。具体完整训练/评估 argv、Git SHA、torch、设备和配置已写入每份结果的 execution 字段；以记录的 argv 复现，checkpoint 指向本目录 sealed 权重即可，不再需要旧 source 权重。

11 项结构测试通过，两套实际 GPU 审计均通过：language/vision真实调用、并行同输入、相位梯度、alpha下限、256输出、参数预算。使用的一张GPU已释放；未停止其他用户进程。

## 清理与 Git

服务器仅删除明确清单内 35 份旧 .pt，4,808,586,899字节，包括旧错误光电模型及不采用的15M实验；保留数据、baseline、历史图/指标、sealed参考。服务器删除不可直接撤销，清单含SHA256。本地旧权重按明确目录送回收站；本目录两份正式权重不删除。

共享根工作树存在其他AI修改与分叉，未 reset、未强推、未全量暂存。独立整合分支是本任务唯一新的代码入口；不可将共享根的旧HEAD当作本轮训练代码。
