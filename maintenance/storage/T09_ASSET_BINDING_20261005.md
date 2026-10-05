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
- 后续只读补核已从原 ZIP 内存读取全部登记 WAV：TRAIN 6263 / VAL 843 /
  TEST 867，共7973条，每条SHA与登记一致，缺失和不符均为零。
  没有落盘解压、重建 log-mel、模型推理或指标重评。

## TEST 与其他任务资产的边界补核

实际准备目录 `clevr_attribute_s17_v1` 只有 TRAIN/VAL 数组和问题文件、1250张
原图及保留TEST身份清单；未发现该目录下的TEST图像数组／问题／场景文件。
源码 `prepare.py` 明确只下载所选TRAIN/VAL原图，官方场景通过远程ZIP读取。
因此这250个保留身份不能直接算作本目录已落地的TEST资产。
服务器另有 `t12_clevr_s17_v2` 与 `t13_four_modal_features/clevr` 的TEST文件，
属于另一个准备／实验链，尚未核对身份和预处理，不混用来补齐T09依赖。
此处只定位早期准备目录的边界，不推断数据被删除，也不自行下载或重建科学数据。

### 后续正式TEST缓存已定位并核验

真正的T09测试图像落在原任务运行目录的
`runs/simulation/clevr_frozen_test_s17_v1/images`，而非准备目录。
对应 `data_manifest.json` SHA256为
`9876b3c6573163bf10aa72b4ac244f0064b914250a24696ea10509c2443d2ebe`。
250张原图逐SHA全部匹配；1500条问题SHA匹配manifest，且父manifest匹配上述准备数据。
250个图像身份与保留TEST清单完全相同，和TRAIN/VAL图像身份交叉为零。
另五个后续CLEVR测试run的问题文件与该缓存逐内容一致，全部六份问题SHA为
`ba19956992118edbc5cb680a27647fbba409026f11ad16a60a552b7a386a3a6e`。
因此不是最终TEST原图缺失；已证明原运行缓存与准备划分的绑定。
尚未核验官方场景原文件的落地副本或重新生成问题，不能把本次身份核验称从零重建。
未加载模型、重评准确率或混入T13数据。

### 测试run与原权重绑定补核

五个实际评估run的metadata均绑定同一准备manifest，问题SHA全部匹配。
`clevr_frozen_test_s17_v1` 自身是输入缓存，不是带评估metadata的结果run。
五个run共12项锁定checkpoint全部在原任务运行目录存在，逐SHA匹配
`locked_selection.json`；其中六项历史记录使用仓库相对路径，需以原工程根解析，
不能以SSH命令当前目录解析后误报权重丢失。本次仅核对位置／SHA／记录epoch，
没有更改选择或原报告。

三个原源码身份分别为 `fe5f5d8b87c5933eee001b9674ae51655eecafc8`、
`1c7222a1119385475e1b464ca7efdb67cdc21a4c`、
`ceb694652183530af7405a2b0b5b2efc884f957c`，不能统一套当前main计算图。
测试manifest只有特征张量SHA记录，本轮没有重建该张量，仍是明确剩余例外。
详细只读收据：`.codex_tmp/t09_test_run_binding_20261005.json`。

## 限制与保护

原 manifest 中 `test_accessed=false` / `retained_test_decoded=false` 是准备时的状态，
不代表测试从未在后续实验访问；已发布任务 README 明确记载后续测试。
本次不重新评估指标，不改变探索/泛化口径。

原图、原 ZIP、所有准备文件、特征缓存、预测、历史 run 和测速继续原位保留。
私有详细收据为 `.codex_tmp/t09_asset_hashes_20261005.json`，只读核验程序在同目录；
不将源数据内容提交 Git。
逐WAV补核收据：`.codex_tmp/t09_wav_archive_audit_20261005.json`；
TEST路径盘点：`.codex_tmp/t09_test_asset_paths_20261005.json`、
`.codex_tmp/t09_all_test_paths_20261005.json`。路径盘点本身不证明资产语义相同。
正式TEST缓存补核收据：`.codex_tmp/t09_clevr_test_audit_20261005.json`，
记录六个run的问题身份及250张图像与父manifest／保留划分绑定。
