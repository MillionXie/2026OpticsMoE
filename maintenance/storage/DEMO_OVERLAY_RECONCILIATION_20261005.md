# 旧 demo 本机与实际服务器 overlay：已收拢和必须保留的差异

2026-10-05只读核对本机 `.codex_tmp/demo_reproduction_worktree`、实际服务器
`/DATA/DATA1/guest3/demo_reproduction_20260915` 与已发布 main
`4d2d15ca98737e99a8660d199b40f6a097c6123c`。按LF归一化的Git blob身份比较；
不加载模型、不读取数据集、不改用户索引或任何工作文件。

| 相对路径（LightGenV2下） | 三方核验结果 | 处理边界 |
| --- | --- | --- |
| tasks/t09_multimodal_matching/vision.py | 三方相同，blob e5f6cd467695adf44bcb3dd3c8f02aeccd8a7796 | 支持声明CNN通道宽度的overlay已归主线，无需重复迁入 |
| tasks/t09_multimodal_matching/test_selected.py | 三方相同，blob 578e14f9a37d034239df8640d3800b30dcc372b6 | 显式视觉前端选择已归主线，无需重复迁入 |
| demo_check/pure_optical/config.json | 服务器与main相同，本机增加oeo_activation=none | 本机独有声明保留，不能整页覆盖或按dirty删除 |
| demo_check/pure_optical/models.py | 服务器与main相同；本机blob e79cdccfc3999a1703acadd26f4766ed7ec72da0不同 | 本机独有OEO扩展保留，不替换服务器已验证后端 |
| demo_check/shared_frontend/run.py | main无此入口，本机与服务器也不同 | 共享冻结前端对照入口尚待完整依赖/运行证据审计，两份均保留 |
| demo_check/pure_optical/config_oeo_intensity_softsign.json | 本机与服务器相同，main无此配置 | 必要对照候选，需与对应runner/结果绑定后归入，不单文件冒充复现闭环 |

服务器逐文件SHA与blob记录：私有 `.codex_tmp/demo_overlay_identity_20261005.json`。
本机共享前端增加 `--optical-config` 并将选中配置写入来源清单；这不是服务器
run.py相同版本，不能只凭功能相似将二者合并。额外图例/assets未在本次六文件范围内，
不把它们归为垃圾。旧目录仍含独有代码、报告及资产，尚不满足移除条件。
全部baseline、测速、模型/数据与原工作树保留；本次没有净空间释放。
