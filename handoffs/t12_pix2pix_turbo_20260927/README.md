# pix2pix-Turbo 同任务外部 baseline

状态：正式微调、完整VAL选模及独立TEST2304均已完成。VAL选中step31104：20.863762 dB / SSIM 0.745786；固定权重TEST：PSNR 20.746391 dB / SSIM 0.740299。主指标为逐图PSNR算术平均。权重SHA256 `3a347c58affb53d8e7efc583bb5aecdaa2ac33bd316d792c12c805fa837b4c7c`。TEST从未参与训练或选模。

模型：官方GaParmar/img2img-turbo，锁定86f54146590ffb4543c8cf85b5a36657da670924。SD-Turbo预训练CLIP文本encoder、UNet、VAE；原生文本条件及单步scheduler，没有Qwen/PCA。新初始化任务LoRA和skip不是从零训练整个主干。

| 组件 | 部署参数 | 可训练参数 |
|---|---:|---:|
| CLIP文本encoder |340,387,840|0|
| UNet含LoRA |874,019,300|8,120,416|
| VAE含LoRA及skip |85,038,607|1,384,744|
| 全模型 |1,299,445,747|9,505,160|

词嵌入50,593,792另列，排除后部署预算1,248,851,955。LoRA未merge，后续如merge需重新统计；LPIPS、CLIP相似度网络和判别器仅训练使用，不计部署参数。权重冻结仍计总参数，不能以9.51M微调量作为模型总大小。

沿用TRAIN20736/VAL2304/TEST2304、三类编辑任务、原生256输出。3epochs、batch2累积4、lr5e-6、seed927；LoRA rank8/rank4，官方MSE/LPIPS/CLIP/GAN权重1/5/5/.5。完整VAL平均逐图PSNR选择best；固定后TEST只评估，不选模。wrapper把G各损失合为一次优化，区别于官方两个G更新，已明确记录，不宣称完全原样复现其trainer。

服务器工作树：/DATA/DATA1/guest3/2026OpticsMoE/.worktrees/t12_physical_robust_v2_20260927

run：LightGenV2/tasks/t12_text_to_image/runs/simulation/20260927_pix2pix_turbo_matched

训练启动源码：7c75a00e3；完整TEST导出源码：a3674c071；分支codex/t12-physical-robust-v2-20260927，已推送GitHub。训练PID527954在单卡GPU4完成，TEST评估PID1075779在单卡GPU6完成，二者已退出；其他AI进程未触碰，未操作实验光路。冒烟测试4微批前向/反向通过，5项CPU测试通过。

`best_checkpoint.pt`已从服务器下载并核对SHA256 `3a347c58affb53d8e7efc583bb5aecdaa2ac33bd316d792c12c805fa837b4c7c`；它只含任务LoRA/skip等适配权重，复现仍须加载锁定SHA的官方预训练SD-Turbo骨干。`test_export/report.json`、`sample_metrics.json/csv`及`images/reference,target,generated`是2304条固定TEST全量原生256图像；`overview_background.png`、`overview_object.png`、`overview_joint.png`用原四组相同的18个固定索引（各模式6条），无质量筛选。`pix2pix_test_export.zip`的SHA256为`d79e359abfb88531a4ddb0dbdd4df8fc6650ed6a157bd7c1a2d2c80491559268`；逐图PNG哈希、ID和prompt对齐审计在`../t12_five_group_summary_20260928/integrity_audit.json`。五组同样本拼图、逐图汇总及可视化表格在相邻汇总目录和`../../outputs/t12_five_group_summary_20260928`。旧四组结果保留。

这里的质量低于现有光电小版/大版及同任务Qwen28 baseline。样例明显有低对比度和细节弱化，不能隐去，也不能把这一特定微调实现概括为所有pix2pix-Turbo设置的上限。训练wrapper合并生成器损失，只部分沿用官方训练流程，不能称作完全复现官方trainer。PSNR量化配对保真，不衡量开放集生图。尚无按相同block→RGB边界完成的新跨架构延迟，不能挪用旧小版计时。architecture.json是实际模型统计；完整复现与协议见隔离工作树LightGenV2/tasks/t12_text_to_image/reports/reproduction/README.md。
