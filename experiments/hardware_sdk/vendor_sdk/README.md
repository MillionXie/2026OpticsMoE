# Vendor SDK inventory

该目录随仓库同步，保证实验室电脑克隆仓库后能获得相同的厂商示例、
Python binding、动态库和说明文档。不要把运行数据写入这里。

| 目录 | 设备 | 公共 driver 名称 |
|---|---|---|
| `amplitude_holoeye/` | HOLOEYE 振幅 SLM | `holoeye` |
| `phase_meadowlark/` | Meadowlark 相位 SLM | `meadowlark` |
| `camera_dvp_legacy/` | 旧 DVP CCD | `dvp_subprocess` / `dvp` |
| `camera_tucam_mosaic/` | 新 Mosaic/TUCam CCD | `tucam` |

厂商 SDK 的许可仍由原厂文件约束。本仓库中的适配代码只调用公开 SDK，
不修改厂商 DLL、PYD 或示例代码。
