# T06 冻结 CLIP／YOLO baseline 入口恢复

2026-10-06后续：本地remote_patch四份旧源码/配置与当前main逐内容复核后，
已原样移入`archive/source_staging_20261006/remote_patch`；不是重新运行baseline。
两份原说明及含测速的结果报告仍在remote_patch原位、SHA不变。
恢复清单见`REMOTE_PATCH_SOURCE_ARCHIVE_20261006.json`，下文“不移动”的描述是之前审查时状态。

2026-10-05，发现当前 main 缺少旧已运行视觉 baseline 的运行模块、两份配置、原测试和
协议说明。本轮从原运行提交 `95904ecf97bcefe4a7ae76a6eab08a1d7cb217f0` 逐文件原样纳入：

- `LightGenV2/tasks/t06_video_quality_assessment/visual_backbone_quality.py`
- 同任务 `tests/test_visual_backbone_quality.py`
- 同任务 `configs/baselines/clip_vit_b32_lgvq_4f_r224.yaml`
- 同任务 `configs/baselines/yolo11s_lgvq_4f_r640.yaml`
- 同任务 `VISUAL_BACKBONE_BASELINES.md`

主运行模块 SHA256 为 `a67b38b0665c5e2ee76eec156cb7887a32117bcb66fde1f8849c7c672a77b09c`，
与本地 `remote_patch/visual_backbone_quality.py` 及原提交 Git blob 一致。
main 的 `project.py` 与 `quality_token_common.py` 已与该原提交逐blob比较，完全相同，
没有为了通过测试改写依赖或模型。其SHA分别为
`bc78c0f6acd24e2a6d738fc8baf8db1b85887ebb82a5ccc0c8243ccacbac0989` 和
`58a5fb721632863638d52e9b6ff2762f0125c997269bfea6834b9df4fef407d7`。

## 核验范围

服务器既有环境、CPU、2线程，从Git blob内存加载原模块及main依赖，原3项测试全部通过：
确定性Xavier初始化、10240参数的五质量行结构、CLIP／YOLO配置合同；两份原配置验证通过。
没有加载预训练backbone、读取视频、训练、重测指标或使用GPU。
本机默认Python误载Roaming目录Torch，DLL初始化失败；禁用用户包后没有Torch。
本轮未改装本机环境，不能把服务器CPU通过写成本机Torch环境已修复。

## 历史结果与资产不混淆

本地 `remote_patch/VISUAL_BACKBONE_LGVQ_20260922.md` 原报告仍原位保留，
包含原运行commit、manifest、预训练PT、特征缓存和输出PT SHA及旧耗时。
原实验为固定LGVQ 2250/558划分、四帧、冻结backbone、各目标10240参数读出；
原无VAL协议按最高TEST目标SRCC选PT，因此是开发指标，不是未触碰TEST估计。
它不是本次已从零重现的指标，也不能将其时间套给最新光学模型。
YOLO依赖版本／许可边界按原协议保留，官方预训练权重不进Git。

源码恢复并不解除原数据、预训练权重、特征缓存和历史结果目录的保护；
本轮不移动 `remote_patch`、不删除任何文件、不重启训练，也不改变正式光学合同。
私有核验收据 `.codex_tmp/t06_visual_baseline_contract_20261005.json`。
