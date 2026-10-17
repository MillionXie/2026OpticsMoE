# 四层平方根权重：原基线收据字段修复

原01d69e8a6预准入在创建CPU claim/out、构造模型或forward之前因KeyError best_sha256中止。原基线TEST result确实只保存passed/metrics/selected_epoch/test_evaluations/test_used_for_selection/test_informed_development；权重SHA和模型digest位于started.json，预测SHA由原独立saved_test_audit核验。原失败和248源码永久保留，不修改/重跑原入口。

本修复是实际身份合同的适配，不放宽任何数学门槛或只改输出名字绕过失败：复用原已验证viewreg.parent_info(BASE,4)，额外显式核started.selected_sha256等于父fit/best.pt/profile固定SHA、epoch86和原独立测试预测SHA。仍强制原完整六臂100/19700/所有原始生产退出与科学/测试SHA匹配；原对照不训练/重测。

原平方根权重profile一字不改，目标/指数.5/anchor.1/LS.05/20CPU LBFGS/同5140头/全非头冻结/VAL选步/一次新digest测试全部保持。CPU/科学用本新来源251=原失败248+3叶文件，新唯一baseline_schema输出，实际独立CPU证明后才能训练；新独占科学和保存测试数学均从新来源执行，不能冒称原入口passed。没有任何新光学forward准入或旧审核重做；拟合0光学更新/forward，真实TEST只在本科学通过后一次。
