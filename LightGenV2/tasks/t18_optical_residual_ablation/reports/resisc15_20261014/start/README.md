# RESISC十五类完整网络配对已经启动

用户明确要求从当前类别中选15类继续尝试。已按固定随机seed20261014从原20类选择15类，未使用test或按现有recall删去最难类别，名单为：机场、篮球场、森林、高尔夫球场、田径场、港口、交叉路口、岛屿、湖泊、草地、停车场、铁路线、火车站、环岛、火电站。
原20类cache及划分不覆盖；只保留这些类的原train/val样本，训练8398、验证1049，test1053仅原metadata保留，未读取测试数组或推理。15类为新开发任务，成绩不能直接作为原20类目标验收。

两组fresh完整L4 rho.3/rho0已实际启动，只占物理GPU2/5各一个子，A100/GPU6继续禁止。父146495、残差子147208、无残差子147277为启动身份；后续以实际proc/UUID和保存日志核。159源码闭包冻结4574b3e84b4426c42724b2bc0a144da04718d93e，原151/144/141/136/135/126/130不改，报告HEAD与运行身份不同。
同908552相位、expert/router224/global478、四主干E/G/E/G及额外router、5captures、原光图/OEO/dense4/8bit/M合同和input100。只用15个类别对应的原64x64固定检测窗口及15路能量归一化，不移动窗口、不引入学习head或缩网格。
同seed17初phase/order/affine/MixUp.6/LS.08/capture.05/TV.005/EMA.95/lr规则、macro16/micro8，完整每轮525次Adam更新、尾14。每组最多120轮或7200拟合秒，既有early-stop先到停止、完成已经开始一轮和必要终点评估；两组最多4新增拟合GPU小时，不启动第三候选、自动retry/扩预算或test，不承诺两组轮数或速度相同。

## 真实启动前检查

CPU独立标签/支持/ID、test load提前拒绝、旧错run提前拒绝、15路label smoothing、保留窗口、参数数和初始化核对通过。两组fresh静态native出口和6项random/dim/black实际BMP保存反变换回放通过，12次CPUforward，概率误差0，原光学入口/CCD/路由和blackzero一致。是fresh离线合同，不是获选trainednative或实拍验收。
实际GPU2新数值检查all15class、macro16、tail14、micro1对8通过，两rho概率/capture误差0、gradrelative<=6.58e-7、全部11phase finite/nonzero，同初始化SHAf7de57b62eb33f6531128061deaa9ec2fb1bacbeea7ba02ecc50528189e60f37。两实际生产子各自也通过门槛，最大gradrelative<=6.65e-7。
CPU父CUDA不可见，source/current与originalGitblob真实一致，项目GPU2/5各1、A100项目0。原始收据按SHA传输本地与服务器，Git只本聚合说明，不发布机器JSON或逐样本ID。

## 首完整轮仅是启动证据

首次完整观察rho.3 E1 val20.4004%、E2 val25.5481%/42.62秒；rho0 E1 val11.7255%/119.44秒（含全train评估）。均健康、completed0/failure0/test0。不同epoch不比较收益，首轮低精度不代表最终结果。完成后需对应新15类独立保存科学审核，工具已准备，当前未执行完整科学审查或selectednative。
专属私有t18_resisc15_audit_dispatch_20261014.py --action inspect真实readyfalse、新cpu_report/failure/claim均无；audit-ready只在两fitcomplete/实际父子rc0/159当前原始一致/无旧out时exclusive durable claim一次、CPUCUDA不可见审核，失败保持不retry。旧20的科学审核与图/PDFmarker均已完成保护，不重跑。
新的15类图工具尚未准备，不能套20类audit/CM/绘图或test入口；真正科学通过后才独立freshout制作、PDF skill一次正确count marker和实际render/view QA。

原strictval65/macro60/各类20/trainvalnozero/末20zero<=half/gap15/all4/inputvariation/实际selectedtrainedBMP保持15类口径；通过也需15合同的真正trainednative、独立depth2/3、正式六组相同profile/init/orderaug及所有真实savedscience/durableonce-test前置。75为目标不保证，当前未验收。仅使用两卡，不连接实验室或改变共享环境。
