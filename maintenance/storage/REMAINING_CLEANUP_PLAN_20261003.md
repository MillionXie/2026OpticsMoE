# 工程整理：当前状态、待办与验收边界

更新日期：2026-10-07。本页只维护当前待办，不再叠加“最新覆盖”历史段落。
**整体整理尚未完成。**过去完整256行记录保存在Git提交
`3adab6945ad68546255d8f4c9dd9486a54940232`的同一路径，不删除原报告或收据。

## 1. 当前范围与入口

- 日常只从仓库 [START_HERE](../../START_HERE.md) → LightGenV2任务页进入。
- 本轮整理本地主工程、Linux训练服务器与GitHub。按用户最新要求，实验室电脑
  排除：不再连接、同步、清理、切换或安装配置；原设备工程与数据保持原状。
  此前实验室建立本地Git入口是历史事实，不证明GitHub登录；认证未核验，不撤销旧入口。
- TASK_REGISTRY亦明确当前范围，旧bench路径和未切换标记仅保留历史身份，
  不再将实验室切换排入本轮待办；科学资产和现用目录依赖的例外仍未自动关闭。
- 长期主线为main，不新增开发分支、工作树或独立工程。现用OpenMoji和封存ABO
  科学运行目录保持原commit/overlay，不为主目录统一而切换或中断。
- 源码走Git；原数据/划分、有效CCD/收据、best/last、逐样本结果、全部测速和功耗
  原件保留，以manifest+SHA管理。外部上传仍暂停。

## 2. 实际盘点快照（不是实时余额或完成百分比）

以下是本轮在发布源码`3adab6945ad68546255d8f4c9dd9486a54940232`时对实际工作目录的
只读盘点；后续其他窗口可能产生新文件。不能把归档引用数、Git修改数和算法数混为一谈。

| 主目录 | 当前分支 | 已跟踪未提交项 | 未跟踪文件 | 开发分支数 |
| --- | --- | ---: | ---: | ---: |
| 本地2026OpticsMoE | main | 0 | 2600 | 2 |
| Linux guest3/2026OpticsMoE | main | 0 | 445 | 2 |

本地未跟踪分组：LightGenV2 1214、handoffs 833、experiments 170、
TransferFromElectricity 112、LightGenPublic 90、ABO_Lab_SHS_8um 88、ABO_Lab_8um 52、
MNIST旧台架17、FixedFeedbackSFT 16、outputs 4、tmp 4。
Linux对应分组：ABO_Lab_8um 156、TransferFromElectricity 85、LightGenV2 83、
LightGenPublic 80、experiments 41。差异不等于丢失源码，也不表示可以整目录ignore。
Linux数据盘当时可用251337773056字节；这是磁盘快照，不是本轮释放量。
后续T06八份原低秩历史配置已发布本地/GitHub `e8817babd`，17项相关CPU检查通过。
Linux同步仍待完成：两次发布及后续只读核验均在SSH握手超时，未执行远端切换；
最后成功发布核验为`c91304f0`，不能据此断言服务器此刻HEAD或进程状态。

后续在发布19531b80时再次实查Linux注册工作目录：**仍有83个，9个有已跟踪修改、
52个有未跟踪文件**。这是与开发分支数独立的整理余额，不能宣称仅剩两个工程。
详见[83个工作目录的当前身份](SERVER_WORKTREE_BALANCE_20261007.json)。观察到现用
OpenMoji与多模态复现目录有同用户进程cwd，原样保护；主目录cwd含检查/其他进程，
不能据数量宣称健康训练。其余目录零cwd也不证明可删，需继续核对导入、打开文件、
未跟踪/被忽略资产、必要对照和恢复点后退役。此次只读检查未清理任何目录。

六个`/tmp/t16_*`旧协议/增强/适配试验已做更细的同用户cwd/exe/cmdline/fd和
四套现用源码绝对路径扫描，未发现引用；被忽略文件仅132个Python/测试缓存，
无非缓存被忽略资产或符号链接。六个原commit均由本地既有T16归档引用覆盖。
这些副本仍带历史测速文件，且与主工程跨文件系统，不能直接重命名搬走。
已请求完整归档方式的用户选择；**尚未复制、移动、删除或退出注册**，83余额不变。
见[六套T16旧副本核验](T16_TEMPORARY_WORKTREE_AUDIT_20261007.json)。

| 保护的Linux工作目录 | HEAD | 当前已跟踪修改/未跟踪项 | 本轮处理边界 |
| --- | --- | --- | --- |
| t07_latestfresh35_20260929 | d979b53ed509907a3630bd2a36c317dfb3aac965 | 0 / 9 | detached封存入口；九个历史启动脚本保留，不执行 |
| t04_openmoji_robust_20260928 | 3b956503a4ecae9b6c20c789ead28621e42e6eeb | 2 / 21 | 现用分支保留，不覆盖、不据主目录clean宣称其clean |

OpenMoji旧32份源码/overlay恢复记录见
[保护收据](T04_CURRENT_SOURCE_OVERLAY_20261005.json)。它是10月5日快照，
本次另对当前23项逐文件比较原始SHA及本地/Linux实际Git恢复对象，全部一致：
20项由原0512414a归档恢复，三个根启动工具由2e64b6c归档恢复，后者恢复路径需映射。
见[当前23项复核](T04_CURRENT_OVERLAY_RECHECK_20261007.json)。没有新建归档或修改实验；
这只证明本次源码可恢复，不证明科学依赖闭包或允许退出现用目录。

## 3. 已关闭的主要入口缺项（不等于全任务复现）

- main任务导航、版本/PT/指标索引已建立；[TASK_REGISTRY](../../LightGenV2/TASK_REGISTRY.json)
  仍明确`migration_complete=false`，受保护旧运行目录不冒称已切换。
- OpenMoji应用线与robust线已分别登记。应用指定layered epoch45 PT
  SHA03cb861c…09eb21、仿真.8765/去光.4845，本地和Linux严格CPU加载139项state通过；
  原分层renderer、8份配置、expansion/DC/CCD合同已归主线。
  指定PT实拍/专属测速未核实，旧epoch40独立打包器不能冒充新版交付。
- 六套9月15日历史baseline的37份包装入口/方法/依赖/数据与PT身份已入main。
  原六ZIP逐SHA/CRC及738份原源码逐SHA通过，包装器默认拒绝嵌套Git、支持Windows长路径。
  LSP原方法说明与后来主线说明分别保留，不互相覆盖。
- T06正式Spatial/Temporal核心与离线适配、SHS显式打包分派已归主线。
  可选Temporal压缩工具及三份配置是本地历史来源，不升格为正式新模型；本地/实际Linux
  9项CPU检查通过，仅增防覆盖，未改变.8044封存PT。
- ABO rank72模型、离线回放、逐层runner与原8图库/4查询划分工具已归主线。
  划分工具与原d979b53e及实际保护服务器源码SHA一致，7项合成测试通过；
  原protocol/2400图/14400CCD与800查询结果没有重新生成或评估。
- 已具名处理的源码副本/结果产物只按证据退出日常入口或待提交列表。
  [157份报告身份](HISTORICAL_REPORT_PAYLOAD_VISIBILITY_20261007.json)、
  [测速导出源码](DEMO_EMBEDDED_SOURCE_VISIBILITY_20261007.json)、
  [三份失败形态试验退役](T12_MORPHOLOGY_TRIAL_RETIREMENT_20261007.json)均保留恢复边界。
  忽略不等于删除、释放磁盘或完成独有源码退役。

## 4. 真正还要处理的四类工作

| 优先级 | 工作 | 完成依据 | 不能采用的捷径 |
| --- | --- | --- | --- |
| 1 | 剩余本地/Linux未跟踪源码、配置与交付材料分类 | 正式依赖入main；原资产保留；重复源码有SHA、用途、占用和可恢复退出证据 | 按扩展名整批删除、整目录ignore、把2600项都说成垃圾 |
| 2 | 当前OpenMoji overlay与旧交付副本保全、依赖退出 | 最新实际commit+逐文件SHA；独有实现保留；无占用且无依赖才退役 | 仅看main HEAD、旧32份快照或目录名字判定最新版本 |
| 3 | 任务原资产/环境/旧入口绑定与历史缺项 | 下表逐任务证据；能补证的补齐、无法补证的具名保留 | 改SHA使预检变绿、重跑科学实验来掩盖旧证据缺失 |
| 4 | 最终本地/Linux/GitHub验收 | 实际工作文件、发布commit、导航、守卫和恢复检查，保留例外清单 | 只比较远端引用就宣布所有目录、模型或数据可复现 |

本地ABO旧子目录还有历史整目录ignore，Linux无同规则，需继续核对是否隐藏应保留的源码。
10月7日另用只读`check_visible_source_recovery.py`复核旧源码索引中仍可见的804项：
657项与索引指定Git恢复blob字节完全一致，53项仅CRLF/LF不同。旧索引未识别的94项
另按本地全部Git引用可达对象核对，全部存在原始字节完全一致的blob，无需新建备份。
这不覆盖索引外新增文件、不证明Linux同字节、占用或依赖，也不是删除许可。
94项包含旧测速脚本和历史现场诊断工具，继续保留；blob恢复身份不等于正式源码入口、
异地备份或无占用，不按“已备份”整目录隐藏。
另有12份9月28日A100历史原报告JSON逐SHA与本机原恢复ZIP一致，已精确退出待提交
列表；原文件不移动、不重算。identity、manifest、summary、配置及新报告仍可见。
28份9月27日已完成ABO鲁棒旧结果也按具体文件归类，原ZIP和当前字节一致；
原报告、模型和指标未改，[原结果保留身份](ABO_HANDOFF_RESULT_VISIBILITY_20261007.json)
包含逐文件SHA。合同、源码、未来报告和现用OpenMoji均保持可见；不是重新评估。
其中18份历史A100报告附带源码已按具体文件退出待提交列表，九种原始blob在本地和
Linux对象库逐字节一致；原18份文件、全部报告和测速数据仍原地保留。
见[历史测速源码身份](HISTORICAL_TIMING_SOURCE_VISIBILITY_20261007.json)。规则不覆盖
新文件或现用任务源码，不声称释放磁盘，也不将旧测速赋给新模型。
实验室checkout/SDK迁移与现场回归现在是范围外，不再放在下一批执行队列。
保护数据不等于已完成异地备份：
[1921份原资产恢复](UNTRACKED_ASSET_RECOVERY_20261006.json)只证明本机ZIP，
不冒称所有大PT、数据、机器SDK或三端完整灾备。

## 5. 各任务的明确例外

| 任务 | 已有正式入口/证据 | 尚未证明或需要保留的边界 |
| --- | --- | --- |
| T01 | DC20与必要baseline、原图/图库 | 历史前端revision与完整资产闭包未全核定 |
| T02 | 官方/个人候选分开；旧Deconv128头/官方TEST成员/200时序/83功率记录核验 | 历史训练启动版本、环境/资产未全闭合；旧测速不套新头 |
| T03 | 指定最终PT、SHS报告、纯CPU回放 | 本机资产与厂商环境未全闭合；旧Meadowlark不是正式SHS |
| T04 应用 | 分层大小/电子缩减指定epoch45，源码/PT严格加载 | resvg_py依赖本机缺失；原SVG/Qwen缓存闭包；新版独立包与该PT实拍/测速未核实 |
| T04 robust | rank64封存G2/G5与TEST开发口径；当前2/21项overlay逐SHA恢复核验通过 | 科学依赖与现用目录退出仍未证明；不替其他窗口选最终版；实验室机器迁移范围外 |
| T05 | 规划入口 | 未开展，不自动建模型或训练，也不伪称已完成实验 |
| T06 | Spatial/Temporal两份正式PT、历史配置、全部测速 | 旧Temporal-36默认资产/SHA预检仍失败；旧working-project release绑定未闭合；压缩工具不等于新成绩 |
| T07 | rank72封存、真实CCD、训练/回放/原划分源码 | 不启动现场回归或重评800；旧协议/目录与未知资产依赖分别保留 |
| T08 | 图搜文/文搜图分开，PT/原图/CCD/缓存身份 | 历史教师snapshot、CCD至读出缓存数值重建绑定未完全证明 |
| T09 | CLEVR/SpeechCommands源码及部分资产 | CPU缓存重建未达固定1e-5容差，原缓存不可当可丢临时文件；官方生成依赖保留 |
| T10 | 固定478缓存、原TEST与36份逐样本结果身份 | 九项历史TEST报告及两项训练完成报告缺失，不补跑或编造 |
| T11 | 病理最终版/必要对照/原划分 | 机器环境与旧目录退出条件；不与T16混名 |
| T12 | 大/小/正式17M及适配PT、必要baseline分开 | 教师/资产/源码副本退出条件；实验室迁移范围外；全部旧测速保护，不套新PT |
| T13/T16 | 实际源码、教师或必要对照 | 原资产与必要历史目录保护，不称新机器从零复现已通过 |

历史A100测速导出681项中672项原SHA通过，仍有七项文档/渲染图缺失及两项检查文件变化。
15项源码/配置已按原SHA恢复，不拿其他版本同名文档冒充原件。详见
[源码恢复](DEMO_TIMING_SOURCE_RESTORATION_20261007.json)、
[文档例外](DEMO_TIMING_DOCUMENT_EXCEPTIONS_20261007.json)。这些是具名历史限制，
不因维护测试通过而消失，也不反复同范围搜索或自动重测。

## 6. 核验工具及其范围

最近完整维护测试为354项，104.01秒全通过；包含T06、ABO、恢复对象核验及忽略边界。
通过的代码commit与331项等旧记录、两项回归修复过程见
[测试收据](MAINTENANCE_FULL_TESTS_20261007.json)。14项登记源码与本地权重SHA无错误，
24份入口文档206个内联链接无缺项；不据此宣称所有科学任务、私有资产或设备复现通过。

```text
python maintenance/git_safety/review_git.py
python maintenance/git_safety/check_task_registry.py --commit main
python maintenance/git_safety/check_task_registry.py --verify-local-weights
python maintenance/storage/check_historical_baseline_assets.py
python maintenance/storage/check_visible_source_recovery.py --index <本机历史源码索引> --summary
python maintenance/git_safety/check_source_archives.py --component-timing-payloads
python maintenance/git_safety/check_source_archives.py --server-formal-timing-payloads
```

带private/PT选项要求相应本机原资产。它们不加载正式模型、不训练或采集；
各工具只证明自身覆盖范围，检查通过不是删除许可。提交守卫已接入主目录但可绕过，
旧工作树不自动受保护；不能声称已锁住所有窗口或阻止全部凭据泄露。

## 7. 整体验收条件（尚未全部满足）

1. 一个主工程的任务页能找到最终源码/PT、数据身份、仿真/直接实拍/微调、必要baseline
   及对应真实测速；没有测过的明确写未测，不能混版本。
2. 本地/Linux/GitHub源码测试后发布main；保留的运行目录明确commit/overlay与用途，
   不假称已切换。实验室不纳入本轮验收要求。
3. 无必要用途的副本退出日常入口；清理前核验占用、导入、独有内容、下游依赖和恢复点。
4. 原数据/划分、有效CCD/收据、best/last、逐样本记录及全部测速/功耗不丢失。
5. 无法补证的历史缺项成为上述具名例外，不能隐藏在“已完成”标签下。
6. 用户能从START_HERE、任务README、TASK_REGISTRY和本页验收，不需翻相互覆盖的聊天。

当前不报缺乏统一分母的完成百分比、虚构腾盘量或没有证据的完成时间。
本页旧版本及所有逐目录收据仍由Git保留；本次收敛说明不删除实验数据、源码恢复对象或测速。
