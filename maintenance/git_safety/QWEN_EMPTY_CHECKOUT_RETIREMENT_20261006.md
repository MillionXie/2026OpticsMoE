# 旧Qwen测速报告临时工作目录：退出开发入口

2026-10-06实际核验、执行并复核。本机分支数从3降到2：main和受保护OpenMoji。
没有删历史提交、测速、数据或任何内容文件，没有新增分支／工作树。

## 原用途和观察到的现状

原路径 `C:/Users/Xml12/AppData/Local/Temp/optics_qwen5090_report_20260906`
是旧Qwen5090D测速／报告工作目录，原分支 `report/qwen5090d-baselines`，
HEAD `119243dc68800b5eb2e6304474a4abde0a14e021`。不是当前ABO或OpenMoji入口。
此前与此次核验均只有Git指针及空目录，没有现存内容文件；3130项已跟踪路径
原本就处于未暂存缺失状态。本轮没有恢复、提交或覆盖这种状态，也不声称过去
未提交的数据可以从Git恢复。

本轮核验：没有命令行引用该路径的其他进程、没有暂存修改、没有软链接内容、
仅这一工作目录使用旧分支。原HEAD、全部3130项缺失状态、索引和主目录状态均受守卫。
命令行扫描不等于操作系统强锁；未停止任何进程。

## 实际处理和恢复身份

完整历史包verify通过并包含原HEAD：
`.codex_tmp/storage_git_backup_20261002/qwen5090_report_history_20261005.bundle`，
SHA256 `e0eb6f2ffed9b0a0f634d66c32c5550cb0039e13e9725a978639032fddd4ebc9`。
包记录完整历史，不依赖当前main作为前提。
恢复引用 `refs/archive/protected-empty-checkout-20261005/qwen5090d-baselines`
仍指向原HEAD。

先仅将HEAD表示改为原提交的detached状态，按旧SHA条件取消旧分支名；
不checkout文件、不清索引。再用Git原生worktree move把同一个工作目录收进
`archive/frozen_worktrees_20261006/qwen5090_report`，不是复制或创建第二套工程。
Git登记自动更新，仍是14个工作目录；旧Temp路径退出日常入口。

移动后HEAD仍为原提交，内容文件仍为零（仅.git元数据指针），3130项缺失状态保留；
索引SHA仍是 `b50cb7fde248f8a887b7b94620e2831f731ee8ed0e0452599ea58e77ef9f123b`。
主目录HEAD／tracked状态不变。未删除文件，未计新增空间释放。

这取消了一条误导性的历史开发线，并收拢一个外部工作目录；不代表3130个源码
已全部归并main，也不把缺失状态“清零”当验收。本轮没有移动任何现存测速文件。
需要恢复原位置时，先确认无同名目录／活动占用，再用Git原生worktree move返回；
需要恢复旧分支名时从上述恢复引用建立，不能覆盖main或现用OpenMoji。
