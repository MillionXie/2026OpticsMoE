# 当前四层原20类配对接续200轮：真实启动

最新人工要求展示曲线、继续优化80/保残差优势并允许三张非A100。真实100轮[曲线](../../caltech20_current4_20261015/curve_review/README.md)已生成PNG/SVG/CSV并实际view_image QA，双方末段NLL仍改善且残差泛化gap扩大。本次双方保持原配置完整恢复E100 raw/EMA/Adam8600/scheduler100/LR.002，接续E101-200；不early/不重播/不重置，原20同数据/窗口/配置，仅rho不同。

run caltech20_current4_s17_pair_continue200_gpu25_20261015，实际160来源c266b7973b105be6caa0dcdb95cfe2fe7cba2abc（不是旧RESISC160），current/original已核。CPU campaign2199729/queue2200431，GPU2残差2201158/GPU5无残差2201161实际运行，own恢复态gate passed，maxgradrel4.9564443e-7/4.9316043e-7。最近E168/E165验证65.72/61.13仅不同轮暂时观察，不当最终gap。无failure/test0，PIDs须实时核。

新CPU state proof runs/smoke/caltech20_continue200_state_20261015/checks.json与邻claim真实一次passed，SHAdb095e55762ee85359ad43b39195de62ee16168152aea7663c54bbd54a6e9235；真父PT/100历史/Adam/scheduler/rawEMA digest和fit编译核验，0models/forward/images/test，不重复准入。父100独占SCI588b5a7046f4f7e4ca6349459d9515874c153a6c31e2a943e48203bed54508a0已真通过且全退出，旧source763冻结；last PT与history SHA详见新profile。

每臂最多100新轮或3600新增fit秒，完整200各17200Adam（86/E/尾12）；安全限若阻止200保留未完成不延长。只2/5两子，与新十类候选一起共三卡2/5/4、一卡一子、2GiB+800MiB，无A100/外国kill。配置沿旧MixUp.2/LS.08/TV.05/capture.2/EMA.95/当前scheduler；不额外手动改LR。

自己的campaign等queue退出rc0/两子exit_code0实际退出且200完成，再独占一次audit_claim/log，新audit_caltech20_continue200.py独立核prefix100/restore receipt/rawEMA/PT/source/17200Adam/orderaffineMix/LR/预算/保存IDlabelargmax20CMsupportmacroNLL/firstminimum，0forward/test。新cpu_report/audit.json/audit_completed SHA真passed、rootstate caltech_continue200_complete_science_passed/campaign退出才完成。失败保留，不手动重审/retry/扩预算。完整200内首次最低验证EMA NLL选，不挑最大accuracy/gap。

这次一对及另外唯一十类开发之外不增加候选/测试/深度/电子/native/formal6/lab。旧155/各RESISC源与数据/PT/历史/失败/claims/markers全保留。Git只源码和聚合，rawJSON/逐sampleID/PT/完整日志私有；用户具体GitHub目的地授权持续，foreignT06保护。t18/09 ACTIVE保留，正常健康静默、真完成/新失败通知。
