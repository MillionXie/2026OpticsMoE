# 训练服务器失效分支配置

## 第二批完成

10个不同名称的映射已在实际服务器逐项检查精确archive ref/commit并退出配置，
合计第一批6项与第二批10项，共16个失效配置节。规划工具复查不再有未绑定候选。
实际3条分支设置全部保留，所有其他配置项及Git引用/索引/工作树/原状态不变。
最终配置SHA256：`63df797cf678d3afa7e6fe041a0a8310f6f4d87b7a149811604d962d379c5370`。
第二批私有备份和逐项收据：服务器主仓库
`.codex_tmp/stale_branch_alias_config_server_20261006/`。

以下是第一批记录，“10个待执行”已由第二批实际操作解决。

## 第一批记录

2026-10-06，服务器实际 `main` 根目录中执行已发布的只读规划和带备份清理工具。
第一批6个名称已经精确对应既有恢复引用且不被任何工作树作为分支使用，配置已退出：
`benchmark/a100-formal`、`benchmark/t07-abo-a100-baseline`、
`codex/abo-backbone-baselines`、`codex/lgvq-clip-yolo-baselines`、
`codex/lsp-salicon-openmoji-backbones`、`codex/t04-qwen-baseline-20260920`。

实际3条分支保留：main、现用OpenMoji、封存ABO；完整Git引用、工作树登记、
HEAD、索引及原9项历史记录状态前后不变。没有执行checkout、模型、GPU或设备操作，
全部内容文件原位，其他配置项逐项比较不变。

- 原配置SHA256：`b2e1491d603087d1d87da7c7c06d46537d8e0d6a19cd732e41720c308b4453f0`
- 第一批后SHA256：`2b5a7fb059ec1782e7bb68a73ddbf26947a50405a2629b88e94798b804c94bfd`
- 私有备份及收据：服务器主仓库 `.codex_tmp/stale_branch_config_server_20261006/`。
  原配置可能包含连接信息，不进入Git。
- 另10个名称已准备不同名称恢复映射
  [SERVER_STALE_BRANCH_ALIASES](SERVER_STALE_BRANCH_ALIASES_20261006.json)，
  此文件本身只表示待执行的精确核验方案，不代表清理完成。

源码通过Git发布；备份私有配置不向其他机器传输。该整理不是新的空间释放，
不是全部资产/旧目录已退出，也没有覆盖实验室现用工程。
