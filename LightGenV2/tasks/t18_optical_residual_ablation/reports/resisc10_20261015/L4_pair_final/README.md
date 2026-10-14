# 原十类纯光配对完成：保存科学和四图实际QA通过，80目标未达

run `resisc10_s17_mainL4_input100_rho_pair120_micro8_gpu25_20261015`，训练160来源 `e380a206b5b032d5d6bf1a855846a73de556564f`。父585499和两子586248/586314实际退出rc0，失败0、test0，原项目GPU2/5释放，A100禁止。该名单按原20验证表现选前十，属于validation-informed开发子任务，不是原20验收或未见验证泛化。

| 组 | 实际轮数 | firstminNLL选中轮 | train | val/macro | balancedNLL | gap(pp) | 停止原因 |
|---|---:|---:|---:|---:|---:|---:|---|
| rho.3 | 120 | 113 | 83.50% | 75.8571% | .859055409 | 7.6429 | maximum_epochs |
| rho0 | 119 | 113 | 81.4286% | 76.1429% | .921296444 | 5.2857 | time_budget |

按首次最低验证EMA balancedNLL选E113，不能改选最高accuracy轮。两组满足val65/macro60/每类20/无zero/末20zero/all4/inputvariation/gap15的开发准入；sciencepassed与80达成是两回事。rho.3仍低于80，未测试，未自动启动正式六组或2/3层。两组选中同E113、共同完整119轮，准确率差-.2857pp仅single-seed描述，不证明残差无效或唯一原因。

rho.3最后history拟合4005.16255秒、含所选端点总4015.20644秒；rho0分别7235.67598/7260.06616秒。7200在每个epoch开始前检查，完整已开始轮和必要端点后退出，不称硬秒杀。Adam350/E，实际42000/41650更新，尾16，全部order/affine/MixUp/LR/early/time独立核验；同phase初始化与共同119轮order和增强保持。

独占真正saved CPU科学审核SHA `c7c7af78dc0fc84d79dd0552dda6fc42c8f82c1e53e1bdbba9c243745b4ab5af`；2CPU模型核shape/seed，模型forward/test数组/测试推理均0。audit/source/current-original/PT/ID-label-argmax/CM/support/NLL/firstmin与独占completed收据实际通过，机器JSON仅本地服务器。所有旧run/PT/data/源码/claims/failures保护，不重审。

rho.3弱类bridge train68.21/val57.14，wetland64.11/64.29，thermal_power_station81.07/64.29，intersection79.82/68.57；bridge主要错到tennis_court(15/70)、thermal(8/70)，thermal主要错到harbor(12/70)。整体train83.5/val75.86，存在拟合不足与泛化差距混合表现，不能宣布唯一原因或充分收敛。按最新人工授权，下一次有界fresh纯光配对只将MixUp.6改.2，作为改善拟合的假设，固定名单/网络/input100/划分/其他训练因素，每组max120或7200秒、只GPU2/5各一；不延长旧run或无限搜索。

四PDF、PNG、SVG和聚合CSV已一次真实作者、SHA传输、Poppler120dpi渲染及全部四页实际view_image通过。[实际视检](visual_review.md)。曲线、逐类召回、两完整10类验证CM的底层support/route/last20zero见同目录CSV；完整机器审核/manifest/transfer/visual JSON保持私有。08:50统一汇报，不提前重复推送。
