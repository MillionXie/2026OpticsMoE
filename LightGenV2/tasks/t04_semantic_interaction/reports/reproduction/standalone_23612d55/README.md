# OURS 原头独立交付包：实际复现验收

日期：2026-09-09。交付为 [openmoji_ours_standard_repro_23612d55.zip](../../../releases/openmoji_ours_standard_repro_23612d55.zip)。

- 大小：132,510,824 bytes（约126.4 MiB）。
- ZIP SHA256：`304b7c457c6f6c8e4e7f41f18deaef6f65270f86e09194f69c7a3e8f45b0294f`。
- 打包源码 commit：`23612d55106cb8c817cf4b4e7b38ea5bc138361a`，已同步 GitHub。
- 固定权重：`routerfill_shared_s73` 原 standard 读出头，best epoch40，不是 slim，也不是新训练的1epoch权重。
- 不含 baseline 权重、完整 Qwen、硬件驱动。包含 train5000/test1000、词嵌入缓存、最小冻结前端与相位PT。

## 完成的检查

1. 本地 T04 测试：24 passed，包括打包导入依赖、路径重定位、改文件拒绝校验、保护已有输出。
2. Linux 从 ZIP 新解压、清空 PYTHONPATH、指定空 HF_HOME、强制离线。校验18,217个交付文件，通过。
3. Linux CPU 单例 test_000008：源图自行车下方加入花，scene_exact=true。
4. Linux GPU（3090，torch2.6.0+cu124）完整1000test及同权重去光，核心4指标与reference误差均为0，详见 `reference_comparison.json`。
5. Linux 独立包重新训练接口：1epoch、5000train，产生best/last并完成评估，耗时约56秒（仅训练器统计）；Router和所有专家均有相位变化，见 `one_epoch_phase_audit.json`。这只是接口冒烟检查，其性能不作正式结果。
6. Windows 从本地ZIP独立解压：CPU、torch2.11.0+cu128，完整文件校验与同一单例通过，source/target/prediction PNG已人工查看。没有重新在Windows跑完整1000test或100epoch训练。

| 指标 | 原best参考 | 解压包GPU复评 |
| --- | ---: | ---: |
| changed_cell_accuracy | 0.8715 | 0.8715 |
| edit_grid_iou | 0.8327333333 | 0.8327333333 |
| object_f1 | 0.9339485237 | 0.9339485237 |
| scene_exact_match | 0.6900 | 0.6900 |
| 同权重去光 changed_cell_accuracy | 0.4055 | 0.4055 |

光路由Top2的语言专家选择占比为24.45% / 25.00% / 25.00% / 25.55%，视觉为25% / 25% / 25% / 25%，无未使用专家。去光降低46.6个百分点；是同权重消融，不是单独训练无光baseline。

## 交接方式

发送整个 ZIP 即可；师姐先看包内COMMAND.md，AI先看AI_README.md。安装依赖后按 verify → demo → evaluate 执行。随包data足以复现；新任意文本需要另行制备匹配词嵌入缓存，不能误称为支持任意新指令的一键接口。

Windows首次扫描18,217个小文件可能明显慢于模型推理；等待 `Verified ... files`，不要因此判断未使用GPU。建议短路径解压，避免旧Windows长路径限制。

可核对的执行证据均在本目录。原训练协议定期test选best，`selection_biased=true`，不以该验收掩盖选模口径，也不保证重新训练逐位一致。
