# T11 原运行到实际数据的只读核验

2026-10-05。从服务器main的 `T11_ASSET_IDENTITY_20261004.json` 选取全部33份
metadata，实际文件SHA均与原登记一致。解析原命令中的数据路径及metadata数据摘要，
不以主目录HEAD替代实际训练版本，不启动模型、训练或TEST评估。

15个唯一文件均存在，共4,948,205,257字节；逐字节流式SHA完成。包含10份NPZ及5份
JSON清单，未解码数组，未验证像素级划分交叉或重新生成缓存。

## 与原运行数据摘要一致的8份NPZ

以下路径均相对 `/DATA/DATA1/guest3/demo_reproduction_data/`：

| 原位置 | SHA256 |
| --- | --- |
| kather_lc25000_c3729eeb/kather2016_binary.npz | ba8b6a30f99798c244377d2585c1fc3fe8bc234c2847afcd373a0932d40882b5 |
| kather_lc25000_c3729eeb/lc25000_lung_binary.npz | c923e22dd6bb85de24393a5de020f9beedb13040f5d4472b3ac2df53d071126a |
| kather2018_val7k_feef2710/kather2018_val7k_binary.npz | 2008579aa6c9d4a74c04d5555dff70dd887f7e9ddd351754ca6b675a14c28782 |
| hepatobench_b0ce0836/hepatobench_binary.npz | d53e4c99b4a4802775ee7230f49c0e5878d8fed822c22e3d9540cca316bda2ff |
| kather2018_crc9_domains_v1/A_original.npz | 287441e12c9f05c8257bf44b12b0157e77e934b88e5596e2dd9f85068484b51b |
| kather2018_crc9_domains_v1/B_stain_h.npz | 67a2337919e06ce5d3a59101f7f9097478a1f48bca9c2b03c65b7de0a6f4901c |
| kather2018_crc9_domains_v1/C_stain_e.npz | 3607b6e9fc0ad3dc6437d6ed83eccfc5d02c8a99b8c3d7330216fec6103c5407 |
| kather2018_crc9_domains_v1/D_scanner.npz | 914525fa5d62ebceae39991ef6575864e6263f6bc95bf0d9b6c0c9de72a6689a |

## 后续结构核对：9份NPZ及5份清单闭合

`check_t11_data_receipts.py` 对已取得的两份收据再做结构核对：Kather2016原metadata的
`data_sha256`与实际NPZ一致；5份JSON清单与原metadata嵌入内容比较43次全部一致。
因此9份NPZ已有历史数据摘要绑定、5份清单已有历史内容绑定。检查器四项合成测试通过，
包括改动数据/清单必须报错、没有期望摘要不能当通过；不会向原run写入核验文件。

## 尚未证明的边界

EuroSAT phase_only数据已取得当前SHA，但两份T11跨域metadata只记录路径，未保存
当时的数据SHA。不能借用另一个EuroSAT训练报告的摘要就称这两次T11运行历史绑定通过。
这不是发现不匹配，也不是原数据已完成全量备份；后续需核对清单嵌入摘要和来源归档。
患者独立性、镜像与官方原图等价性及数据重建依赖仍按原任务协议保留限制。

私有完整收据 `.codex_tmp/t11_data_content_20261005.json` 保存每份文件大小、当前SHA、
期望SHA、引用它的原run及清单文本。原始文件、源码、环境、权重和全部测速未变。
