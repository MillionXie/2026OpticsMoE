# T09 原数据依赖核验（2026-10-05）

本次只读实际训练服务器资产，没有训练、推理、移动或删除文件。实际源码入口仍为
`/DATA/DATA1/guest3/demo_reproduction_20260915/LightGenV2/tasks/t09_multimodal_matching`；
核心源码已发布到 main，但原运行目录尚未切换。以下核验不等于完整任务迁移完成。

## 图文匹配

准备数据在 `/DATA/DATA1/guest3/demo_reproduction_data/clevr_attribute_s17_v1`。
manifest SHA256 为 `a5826f81d530e41e776eac528886936d64b5ad413e62268510fa62c0f09e5ade`。

- manifest 所列七项准备文件逐字节 SHA 全部一致。
- `raw_images` 中 1,250 张训练/验证原图逐一核对 `source_files.json`，缺失及 SHA 不符均为零。
- 图像身份划分为 TRAIN 1000 / VAL 250 / 保留 TEST 250，三组两两交叉均为零。
- 问答文件为 TRAIN 6000 / VAL 1500；独立图像数不能与问答数混用。
- 本次未核验保留 TEST 原图、场景原文件和测试特征缓存的完整重建闭包。

## 音文匹配

准备数据在 `/DATA/DATA1/guest3/demo_reproduction_data/mini_speech_matching_s17_v3`，
不是仅含下载原包的 v1，也不是不完整准备过程的 v2。
manifest SHA256 为 `e20d789937a44512ce4c91f222ba99145e36c63f3ee386295585d61cc018c0c1`。

- manifest 所列十项准备文件逐字节 SHA 全部一致。
- TRAIN 6263 / VAL 843 / TEST 867 的文件身份、说话人身份、记录音频 SHA 三种检查，
  三组两两交叉全部为零。
- TRAIN/VAL 问答数分别为 12526/1686。
- 原始 ZIP 仍在 `mini_speech_matching_s17_v1/mini_speech_commands.zip`，182082353 字节，
  SHA256 `49650f2341b26d886b46b3f4fb8fed59e30300b17550f1ee4a768b3106cf93a0`，与 v3 manifest 一致。
- 本次核验 ZIP 整包身份；尚未逐 WAV 解压核验，也未重建 log-mel 或执行测试。

## 限制与保护

原 manifest 中 `test_accessed=false` / `retained_test_decoded=false` 是准备时的状态，
不代表测试从未在后续实验访问；已发布任务 README 明确记载后续测试。
本次不重新评估指标，不改变探索/泛化口径。

原图、原 ZIP、所有准备文件、特征缓存、预测、历史 run 和测速继续原位保留。
私有详细收据为 `.codex_tmp/t09_asset_hashes_20261005.json`，只读核验程序在同目录；
不将源数据内容提交 Git。
