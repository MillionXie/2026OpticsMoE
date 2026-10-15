# 筛选Caltech20当前四层：同100轮配对已实际启动

用户2026-10-15明确“差距优先，换回这个，快点训练当前四层”。本轮新授权仅固定筛选20类Caltech和当前四专家/四主层的一次fresh rho.3/rho0完整100轮配对；不early，不自动test/换类/追加候选/电子分支/正式6/native/lab/A100。旧173等完成阶段的禁止新训练不阻止此新人工范围，旧全部产物仍保护。

新run `caltech20_screened_current4_s17_pair100_input100_mixup02_gpu25_20261015`，登记训练来源`763e6a242a3fe27a513044dc6a6282a0ef574b7d`。155登记来源已核current/original；报告HEAD不同训练身份。CPU campaign父2104530/queue2105395，GPU2残差子2106051、GPU5无残差子2106196已实际运行，各一，A1000/test0。PID是历史快照，须实时proc command/UUID核。启动邻launch_claim/started/campaign_claim/log、root metadata与两生产门槛均真实存在，不重入。

最近只读残差E10 val28.6219%/macro26.7080%/NLL2.58718/9.163sec，无残差E9 val14.1343%/macro13.0885%/NLL2.78552/9.021sec，active2/pending0/fail0；仅早期健康，不能不同E拼最终gap。按初期实测100轮预计约15–20分钟，受共享占用变化；不是已完成或保证时长。

## 数据和光学合同

原冻结Caltech筛选20类与完整划分复用，不重新prepare/选类：Faces/Leopards/Motorbikes/airplanes/bonsai/crab/crayfish/grand_piano/helicopter/kangaroo/ketch/laptop/lotus/minaret/revolver/schooner/trilobite/umbrella/watch/yin_yang，train1372/val283/test2064metadata。原cacheSHA a81e78bbff06ea65e7ceba3771b688178276464e33fdcdedf3796fa2930fa6e1，manifestSHA91f7aa36a97713e300699367904c8866551996de199b9630c0545bb24dc5f60e。没有读本轮test图；旧Caltech test已观察，不称完全独立未见泛化。

当前908552相位、四224专家、E/G/E/G四主层+光router、478场、RGB100、联合8bit/OEO/固定20检测窗口，无电子可训练分类支路。相同seed17/init/order/affineMix，仅光学rho.3/0不同，采用筛选Caltech公共MixUp.2/LS.08/TV.05/capture.2/EMA.95/lr.002和当前scheduler；macro16/micro8，每E86Adam/尾12，共8600Adam。两组完整100、minimum_epochs1000禁止early；每组10800fitsec安全限、最多合计6fitGPU小时，GPU2/5各1、Torch2GiB+800MiB，不抢杀外国进程。

这是当前四专家光路的新适配实验，旧九专家Caltech测试+7.95等只是历史参考，不能保证复制；也不能将其与旧luminance/top2部署方案视为单因素比较。测试支持极不均衡，最终要同时报告accuracy/macro和逐类。

## 已真实一次通过的准入

CPU新proof `runs/smoke/caltech20_current4_cpu_native_20261015/checks.json` SHA8742176515c2cb69fcf0449a039ba7b2dd89130b5aeacd5fcb27ddc4cfe82fcc：真实train/val标签/ID/support、20softtarget/86steps/tail12、黑20路均匀.05与sum1、两个static及random/dim/black nativeBMP逆向/CCD五capture回放，14forward、test0。只是离线准入，不是已训练模型实拍。

GPU2新proof `runs/smoke/caltech20_current4_gpu2_numeric_20261015/checks.json` SHA0e30039214c583c71e072ff1ed9eeef5762e2d26e25815feef7ec4ccefd1e1ea：两rho/all20/macro16/tail12/micro1vs8，maxgradrel5.1187459e-7、all11finite非零、optimizer0/test0。两组实际producer own gate又各自真实passed，残差最大5.1838668e-7/无残差4.5483281e-7；不套独立收据替代生产门槛。共同initSHAf7de57b62eb33f6531128061deaa9ec2fb1bacbeea7ba02ecc50528189e60f37。

Linux导入/独立审核适配编译已过，不等于完成科学。独立审核引用的 `audit_resisc20_input_pair.py` 助手不在155运行登记来源内，已另外逐字节核与训练Git763一致，SHA9bee138135b1cfb5f616208629f541aebf4021a96325e32e561aa7d4e5a4013b；不篡改运行登记或旧助手，完成前后额外核该SHA/Git身份，不能冒称155包含此文件。

## 自动一次保存科学及结束条件

两组实际100/maximum_epochs、GPUproducer实际退出且原始exit_code0、queue实际退出rc0后，campaign自动独占新audit_claim/log调用新audit_caltech20_current4.py一次。核源/config/rawEMA/PT/init/11梯度/8600Adam/orderaffineMix/LR/预算/savedIDlabelargmax20CMsupportmacroNLL/firstminimum；CPU2model仅shape/seed、0forward/test。新cpu_report/audit.json/audit_completed SHA真正通过且campaign父退出、rootstate caltech_current4_complete_science_passed才报告最终结果。

任何失败/安全限未100保留，不自动retry/放宽/扩预算/手动重审。按首次最低验证EMA balancedNLL选模，不最大accuracy或gap。不自动test/PDF作者/markers/额外优化。源码只Git，数据/PT/逐样本ID/rawJSON/完整日志私有。用户已直接授权GitHub仓库及服务器同步；foreignT06修改/暂存/提交全保护，仅精确T18paths发布。t18和每日09晨报ACTIVE，新普通健康静默，新失败一次通知、真完整科学完成主动报告。
