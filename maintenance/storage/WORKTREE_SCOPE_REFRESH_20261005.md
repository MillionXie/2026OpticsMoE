# 服务器运行目录只读复核

2026-10-05实查83个注册工作树全部存在，19条分支名；当前main引用
`aa78b93cbafbcd9c8a64c1ff10247468aa8429b9`，服务器根运行HEAD仍为
`3f85510285e5ffdfca28def93eef2eb082b1655c`。没有切换checkout或删除目录。

## 为什么不能按数量批量删除

10个工作树有已跟踪修改，62个有未跟踪内容，73个有忽略内容；类别重叠，不能相加。
忽略内容可能是正式数据、run或缓存，不等于垃圾。按本用户同UID的`/proc/*/cwd`
只读检查，有2个注册目录被进程作为工作目录使用；这不等于只有2个任务运行，
脚本可以从别处启动或引用其他目录，尚需命令、打开文件及下游引用审计。
未发现缺失注册目录，所以不能仅靠prune减少数量。

服务器根提交树与main的1265个路径差异分别是main独有908、根历史版本独有306、
同路径内容不同51；未提交overlay不在该比较内。不能称为1265个丢失提交，
也不能以清零该数字作为完成标准。

## 无额外文件的七个目录仍有用途待审

下面七项均为detached、无tracked修改、无untracked或ignored文件，当前没有同UID
进程cwd占用。用途说明来自其HEAD提交标题，只是定位线索，**不是完整依赖审计**。
本轮全部保留；后续必须验证必要对照、全部测速、源代码恢复包及下游引用才作取舍。

| 旧目录 | HEAD | 用途线索／保留原因 |
| --- | --- | --- |
| fa_vtab_20260914 | a2e6096fbe6ce43e6bbbc55b88f8ceb95aa4fd8b | 固定专家实验GPU UUID绑定，可能属于必要baseline |
| t03_handoff_20260915 | 3b931ccdcbd398f1c73f57d46621eccbddeff4f0 | SALICON校准Meadowlark/TUCam三步交接，需确认设备依赖 |
| t07_gallery_fp32_20260912 | c8ec81ced7f1c4d54b848ec99f747ac38c77bae5 | FP32图库训练损失对照，不自动视为无效 |
| t07_goal81_verify_20260912 | 5f3c704ad93ee8baade05ea2fc13823119c21d84 | 显式SHA权重独立复评，可能是结果证据 |
| t07_joint_best_restart_20260912 | 35055dc78d9ddd585e3d39357312f2f508cd287d | 固定已验证joint best用于成对重启 |
| t07_retail_audit_20260913 | 5a7bf527a1b4a479180097f493c5ac648057ab31 | ABO正则化、SHAPE pilot与交付审计 |
| t07_vision13_20260913 | 6c1320c2ff798321404a5e6060852033130b52b2 | 校准79.5833 best核验与有界vision13对照 |

完整私有只读收据位于 `.codex_tmp/git_runtime_scope_refresh_20261005.json`，
用途提交标题收据 `.codex_tmp/clean_tree_purpose_probe_20261005.json`；未公开数据、
进程命令或连接信息。该复核不批准删除，也不证明所有旧源码已归主线。
