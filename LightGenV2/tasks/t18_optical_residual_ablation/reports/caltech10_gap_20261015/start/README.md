# 验证知情Caltech十类：一次fresh150配对已实际启动

最新人工允许必要时挑类别，目标提高自身准确率并保持残差优势，允许三非A100。原完整档案仍有，比20更多类别；历史合格池47已用固定缩略图代理筛到20，代理不能证明光学收益。为有限时间与已实测当前光路证据，本次仅选唯一十类名单：旧当前四层E100验证残差recall>=60%、与对照差>=0，按residualRecall+.5*gain排序、同分name，取10后alphabetic。选择只使用验证，不用test，不反复换名单。

冻结Faces/Leopards/Motorbikes/airplanes/bonsai/grand_piano/revolver/schooner/trilobite/yin_yang，原indices[0,1,2,3,4,7,14,15,16,19]。新任务10类，不能冒称原20或完整101达到80；挑旧行平均不是新训练分数，未知是否提高。原test已见，不称完全独立泛化。原cache SHAa81e78bbff06ea65e7ceba3771b688178276464e33fdcdedf3796fa2930fa6e1/parentmanifest91f7aa36a97713e300699367904c8866551996de199b9630c0545bb24dc5f60e及角色/IDs保持。

一次metadata准备已实际完成：/DATA/DATA1/guest3/t18_caltech_gap10_20261015/data_manifest.json，SHA3227d5c1fbd86a2454b1df51d8b58e2ca772bfbe32c26733650d15a06b463f71。train761/val157/test1833仅metadata，没有读test图。当前全908552phase/四专家/四主层E-G-E-G+router/478global/RGB100/8bit，仅保留十原坐标窗口并10路归一，不增加电子分类头。

run caltech10_gap_s17_current4_pair150_gpu425_20261015，新169实际来源259cdcfdec6edc9b8ca29f9c7ddf18fb3668cffa（不是旧RESISC169），current/original已核。CPU campaign2220246/queue2220984，GPU4残差2221711实际在训、own gate passed maxgradrel6.9281612e-7，首E2 val11.46只是早期，不能当最终。无残差L4_gap10_rho0 pending1，允许的4/2/5空槽后自动启动。与原20接续2/5同时总3子，各卡1，A1000/test0；不同启动时间不改变双方完整150预算。

新CPU proof runs/smoke/caltech10_gap_cpu_native_20261015/checks.json及邻claim真实一次passed，SHA75af8c42cda5497163b255a8d13e17bab41f8aea2ca9fa29c293fc46f41acec3：真train/val IDs/support/10softtarget/拒绝testrole/黑10概率.1 sum1/nativeBMP实际逆/CCD5capture回放、14forward/test0。新GPU4proof runs/smoke/caltech10_gap_gpu4_numeric_20261015/checks.json及邻claim一次passed，SHAff5cd2de2b26c6cd283569814d5cb85f951db97f926b2bac90cf371622397747；两rho/all10/macro16/tail9/B1vs8/maxgradrel9.2564352e-7/all11finite非零、optimizer0/test0，不重任何prepare/cpu/numeric/helper。每个实际producer仍须自己的真gate。

双方fresh seed17同init/order/affineMix，共同MixUp.2/LS.08/TV.05/capture.2/EMA.95/lr.002和原scheduler，只rho不同。完整150、每E48Adam尾9、共7200步，不early；每臂3600fitsec安全限，不延长/retry。firstminimum验证EMA balancedNLL选，不最大accuracy或gap。自己的campaign在queue及两GPU实际退出原始rc0/双方150后独占新audit_claim/log/audit_caltech10_gap.py，独立保存科学核source/config/PT/rawEMA/init/11grad/7200Adam/orderaffineMix/LR/预算/每E10CMsupportmacro/最终保存IDlabelargmaxNLL/firstmin，0auditforward/test；自己的audit/completedSHA真passed及父退出才完成。失败保留，不手动重审或反换有利权重。

此为唯一一次类别开发配对，与原20接续分栏报告；80和正差均不能保证。不新增第三方案/test/电子/额外深度/A100/native/formal6/lab/PDF作者。原source/data/PT/history/失败/claims/markers保护，rawJSON/IDs/PT/完整日志私有，源码只Git。t18与09 ACTIVE，普通健康静默，真完成/新失败主动报告一次。
