# T12 图文条件商品编辑：main 统一入口

当前实拍主表使用17.03M小版：原权重 `5b4f9a37…` 与末端decoder适配权重 `eeec…`
分别保留。原权重仿真34.277751dB、未适配实拍27.551289dB；适配权重实拍31.552886dB、
其自身仿真28.890612dB，不能跨权重拼成sim-exp差距。精确指标口径和原产物见下面复现入口。

大版149.76M、小版9.96M及其他17M候选是分别保留的历史版本/必要对照，
不是几条新的长期开发分支。每套支持换背景、换目标、同时修改；高级材质及失败的
开放形态实验不作为最终版本。此页在main维护，旧分支名只记录原实验出处。

唯一入口：[复现与最终产物](reports/reproduction/README.md)。旧报告保留作历史记录，不代表当前部署入口。

**2026-10-28六小时自主推进：课程预热已完成，最终仍为复杂商品场景。**
`qwen_vl_sixhour_curriculum_workflow_20261028`已完整结束并释放自己的两个GPU进程。
8轮短指令课程`qwen_vl_simple_curriculum_warm_20261028`选step960，
SHA`52cf761c2df4781caa9ae5dec0ef09e8e63ecc41c76c6423308bd402fece3588`，
课程DEV PSNR20.989052、LPIPS0.293820、crop LPIPS0.376259；它不是最终复杂任务baseline。
原生指令mask＋固定0.15文字注入已通过真实反向，条纹／灯罩形状有响应，但未知靠垫
纹样保持仍不足，不把课程分数与旧复杂任务分数混用。后续从已训骨干继续，而非随机重训。

`qwen_vl_scene_backgrounds`只由冻结公开SD-Turbo准备TRAIN／DEV合成背景资产，不是
编辑模型的预测。首批108张广角房间图因缺少合适桌面、产品比例会失真，未批准作为GT。
6张`--layout tabletop --probe`近景探针桌面构图改善，房间语义／侧光仍需逐项复核；
新完整池要求书房书架、卧室床头、客厅沙发等上下文，不能仅凭teacher prompt认定实际属性正确。
复杂数据入口为`qwen_vl_scene_compose`／`qwen_vl_scene_render`。近景资产中的卧室／客厅
仍可能缺少空桌面，因此统一在数据制作时增加同split书房照片中提取的木质前景台面，
再按真实alpha摆放ABO商品并加接触影；不将台灯按房间尺度放成巨型落地灯。
白底JPEG先按真实alpha恢复前景颜色，避免二次white-matte白边；曝光／白平衡和方向
增益为记录在数据协议中的可复现像素场，不冒称物理逆渲染。色调／亮度的单属性对照
复用同一房间和桌面像素，仅改变已声明光场；其他场景用同split独立教师素材。
每个参考图的8个反事实指令共享同一参考RGB和同一生成seed，生成seed只依赖参考身份，
不从目标商品、目标场景或教师背景seed推导，避免随机状态偷偷携带目标标签。
TRAIN额外包含已声明的组件颜色增强：绿色金属灯罩成对变红／蓝，青色纹样底座可变蓝，
保留alpha／形状／亮度纹理和白高光；若目标组件缺少足够真实色彩像素则失败，不能只改文字。
这是合成编辑监督，不声称原ABO商品真实售卖这些配色；DEV／保留集不因此进入训练。
新数据预算1824／96须按行协议审核后缓存，原960／96兼容路径不改变。
复杂课程配置`qwen_vl_rich_scene_baseline_20261028.json`从已完成短课程best继续，
最多36轮、每2轮开发评估，连续5次不改善早停；固定完整Qwen及VAE，训练原有
适配器／空间读出及UNet指定参数。增加TRAIN专用场景／商品CLIP约束0.15／0.25，
不向推理加入CLIP。CPU制作时已修复DEV仅需2个场景时继续追加导致数量不符的问题，
并增加2条件回归验证；失败工作流保留，数据准备前失败不覆盖有效产物。
后续`qwen_vl_rich_postfree_20261028.json`将postfree结构写入checkpoint配置，
恢复／尾部／计时入口统一按此配置重建，防止旧入口把光电模块后的TF悄悄加回来。
新教师缓存明确增加vision23／language27端点；可选特征损失分别归一化学生与教师，
用弱log-RMS约束尺度，避免旧原始尺度MSE爆涨。尾部学生继承教师的指令mask和文字
读出增益，不能压缩UNet时丢掉它们。上述是待GPU验证的候选，不是质量或速度结论。
LG恢复可同时低学习率适配已有adapter／spatial读出，不增加部署网络；教师使用独立
冻结读出，学生只共享冻结UNet／VAE。学生读出完整保存在宏checkpoint的head_overlay，
尾部、raw场景审查与计时严格恢复，不能只存宏却丢失训练过的读出。CPU验证教师不被
学生更新污染、overlay重载和指令mask一致。`qwen_vl_scene_review`输出按三任务分组的
完整prompt／reference／GT／raw生成图，GT只进入离线评分和排版，不进入生成器。
扩展独立测试的`qwen_vl_scene_holdout`采用已审核10款／18原视图，每视图在10个
独立背景上下文中构建参考图，每个上下文各做三种编辑，共540对、每任务180对，
其中参考场景180张。背景由显式`--evaluation-assets`另起资产run，TEST seed不与
TRAIN／DEV交叉；此标志只准备背景，不运行编辑模型。测试商品两种角色及RGBA均与
完整warm-start父链TRAIN／开发排重，RGB＋prompt重复会失败。540不是540款独立商品，
统计应考虑商品／参考上下文聚类；不靠重复同一输入或随机种子试验凑样本。
测试资产可预先制作，但模型／协议必须冻结后才统一推理，绝不进入特征缓存、梯度或选模。

复杂baseline中期仍出现灰色替代品、属性遗漏，不能因背景更清晰就接受。代码审查确认
原图像token按因果位置先于指令，最终图像token本身不能读取后面的文字；原UNet仍有
文字cross-attention，但空间参考读出此前只接收0.15全局文字广播。
待验证候选`qwen_vl_rich_conditioned_baseline_20261028.json`在现有空间读出加入
零初始化instruction FiLM（524,544参数，非新增TF／RGB编码器）：使用同一Qwen指令
token均值调制128个空间通道；TRAIN参考重建aux保持未调制。CPU零调制与旧结果严格
相同，启用调制后文字有非零梯度。另加TRAIN同RGB／不同指令的LPIPS排序损失，
负例不能来自DEV／TEST，也不能偷偷改变参考像素。仅在单独run真实反向、视觉审查后
判断是否改善；不把架构分析当作已证实的唯一原因。正在执行的旧run和待审查队列仍
固定各自发布源码；追加loss候选暂不替换当前FiLM运行版本，待其训练和raw审查释放后再同步。
复杂数据已正式早停24轮，best为14轮step3192、SHA`4655a12a833e773772f0958fd6c9ddc3203f2ed5c3f4588e0dbae39654571bff`；
完整raw DEV96为PSNR16.805277／LPIPS0.419855／crop0.381493。六cell raw与cache的hidden、RGB及PNG完全一致，
红灯罩幽灵形／灰化、枕横条遗漏等未通过语义审查，不作为合格生成或LG质量结论。
已测基线firstTF到GPU RGB均值127.38ms，UNet53.43ms、VAE11.11ms；即便删掉全部TF且无替换开销，
仅UNet＋VAE地板也约限制在1.97倍，因此两倍目标必须同时压缩尾部。此为单进程诊断，不是最终DAG或实机加速。

条件化轮次后若小前景仍被场景损失掩盖，备用`qwen_vl_rich_detail_baseline_20261028.json`
在相同推理网络上强化面积归一的GT前景RGB、真实alpha边界梯度和物体crop LPIPS，并试弱PatchGAN。
判别器只看TRAIN完整图／目标crop，推理没有D或GT裁剪；先用零张量对SN做权重缓冲初始化，再执行
G／D独立更新，D及CLIP参数单列训练专用、D状态随checkpoint保存。DEV按全图LPIPS＋0.5crop选模。
所有新增loss默认关闭，不改变旧run；CPU区域归一／边界梯度／G-D隔离验证通过，GPU效果尚待独立smoke与审查。
TRAIN原图进一步抽查发现一处新引入的数据增强缺陷：旧绝对RGB色差阈值会漏掉暗部，
再用亮度比例缩放时可能红通道裁切，导致部分“红灯罩”GT仍带绿块／过饱和色斑。
原始ABO图未改动，问题仅在声明的150张TRAIN合成配色目标，DEV／TEST不含该增强。
`qwen_vl_scene_repair`以逐行显式`hsv_component_hull_v2`修正组件色相，保留HSV明度／饱和度、
中性高光和暖色黄铜；旧recipe仍保持原像素便于复现。修复写入独立数据run，原数据不覆盖。
只有逐张验证参考RGB、prompt、caption、noise完全不变，且原PNG与原cache匹配后，才复用
冻结Qwen特征并重绑定150个TRAIN目标tensor；1920输入没有重算或混入DEV监督。修正后仍需
先看原色／旧增强／修正增强QA板，再进行真实GPU反向，不把CPU修复代码当作已改善模型。
所有教师资产保留出处、seed和SHA；不为独立TEST生成背景或缓存。ABO CC BY 4.0
仅陈述ABO来源，不把教师背景自动标成ABO授权资产。

新增可选`qwen_vl_semantic_loss`为冻结官方CLIP的TRAIN图像—caption感知约束，
可分别约束完整场景和物体crop，梯度只回传生成图／学生，CLIP不加入推理。
仅在配置显式开启时加载本地已固定上游与权重，记录训练专用冻结参数和权重SHA；
缓存仅TRAIN文本，不用GT图像作为该loss编码器输入。旧配置关闭，旧推理图不变。
该新loss已通过CPU梯度合同，尚未做GPU训练；不得先称已提升。多随机种子新协议可保留
每条记录noise_seed，只有显式非空shared_noise_seed才覆盖，避免默默固定所有生成状态。

**2026-10-28用户要求重做简短指令／简单背景数据。**
用户随后明确论文最终仍需多属性复杂商品场景生成，并授权增加TRAIN、将独立评估扩至
500余对。故简单数据仅作为8轮`qwen_vl_simple_curriculum_warm_20261028.json`课程预热，
不能当作最终复杂任务baseline，也不以五种纯配色替代论文场景。后续完整任务须增加
真实商品属性覆盖，设计同参考图的文字反事实对照，另建约576对商品隔离保留集，
清楚标记商品数与重复视图／场景数；正式模型冻结前不做保留集推理或选模。
原始视角增量筛选已记录在`configs/qwen_vl_rich_scene_review_20261028.json`：
保留原32款／64个已审TRAIN视图，新增4款清楚的单头台灯＋8款纯色／大条纹靠垫，
每款仅1个已目视确认的完整正面／轻侧视角，合计44款／76视图；完整任务预算1824对。
DEV仍8款／16视图／96对。独立保留集计划沿用原8款／16视图，另增1台灯＋1靠垫各1视图，
共10款／18视图×10条件×3任务=540对，各类别270对；不是540个独立商品。
未采用自动mask填充率排名作为最终视角：它常偏向台灯俯视／靠垫背面。候选全图及
新增台灯所有视角已检查，排除多灯头、落地／三脚架灯和坐垫；测试坐垫样商品
`B07YJ4HM7S`未纳入。新选产品尚未生成正式多属性场景数据，更没有测试模型分数。
新增`qwen_vl_simple_edits`，profile `configs/qwen_vl_simple_edits_960_96_20261028.json`。
沿用已筛完整的台灯／靠垫商品及TRAIN／DEV身份，960／96规模不变。换背景只指定
一种影棚配色；换物体只指定一种款式（圆柱／锥形灯罩、条纹／几何图案靠垫）；
joint只组合这两项，不再同时要求底座、材质、色调、亮度、房间和窗光方向。
英文实际模型指令最多14词，中文解释随清单保存。背景为5种配色的平滑影棚光照场，
无窗框、房间道具、随机矩形、半透明大块；真实ABO alpha不腐蚀。
同款式在TRAIN对应多个商品，不加入SKU／品牌；同一参考商品＋目标款式固定代表商品，
背景颜色不选择目标SKU。检查相同RGB／prompt／seed无冲突目标、角色与像素跨split隔离。
目标是该款式的一个监督例子，不宣称唯一合法生成；生成质量仍须检查文字遵循与保留内容。
准备阶段只CPU，旧数据、旧权重及所有失败证据保留；240旧保留集不用于这次开发。
如后续正式比较采用新协议，baseline／LG须在同一新数据上训练、评估，旧任务均值不可混用。
`render_pair`和特征cache按行协议显式分发，新短指令不会被旧完整商品注释重新扩写。
本轮优先重做数据及展示，不继续启动旧复杂任务的先验修复／LG训练。
CPU run `abo_simple_edits_960_96_20261028`已完成并通过两端13项合同测试；
TRAIN为32款／64视图／960对，DEV为另外8款／16视图／96对，每任务分别320／32对，
原生256×256。实际指令7–12词。TRAIN实际替换目标中，锥形灯罩7款、圆柱灯罩8款，
条纹靠垫2款、几何靠垫7款；没有用单个商品代表所有训练指令。
1056对共2112张输入／目标PNG及逐文件SHA保存在服务器同名run，
清单、12张任务示例及报告已SHA下载本地同名run；本地`new_task_overview.png`为
六个任务单元的原生tile汇总。它们是**重制数据示例，不是新模型生成效果**。
该新协议尚无训练权重／模型分数；旧复杂场景权重没有被覆盖。

用户授权随后6小时自主推进电子效果与LG同口径>=2×。电子候选配置为
`qwen_vl_simple_baseline_20261028.json`：先从新参考RGB＋7–12词指令重新缓存完整冻结
Qwen视觉24／语言28输出，再从已训电子空间头继续最多60轮，DEV每2轮选LPIPS best，
12轮无改善早停；不排除同款式多商品样本，保持960／96预算。可选原生instruction
读出由processor逐行验证token边界，只让现有语义resampler关注指令token，并以固定
0.15 RMS增益将其上下文混入已有空间读出；不增网络、参考VAE分支或额外TF。
冻结Qwen／VAE，训练原adapter、空间读出、UNet cross-attention／LoRA／空间注入；
以新TRAIN latent／区域RGB／感知及参考辅助损失学习简单编辑，不再对齐旧复杂caption。
保存best／last、四步真实反向smoke及学习曲线，完成须核查短指令实际执行，不能只看清晰度。
缓存入口`qwen_vl_simple_edits --stage cache`按准备清单SHA核对，不复用旧复杂场景特征。
本候选初始化使用已训生成骨干，后续所有LG对比必须在同一新协议上重做，不能借用旧均值。

**2026-10-28公开生成先验轻量修复pilot已完成，未通过质量验收。**
run `qwen_vl_baseline_nativeprior_20261028`在600步TRAIN条件暖启动后完成10轮，
选择step1200，SHA`22c62dfb252fb55ec6e5c26923f3c20dec478425b390df8b5f647fd990e3cbcf`。
96开发图PSNR17.031515dB、LPIPS0.486359、crop LPIPS0.462532，明显弱于原电子对照。
600步条件对齐末步TRAIN loss为0.146073；4步真实生图反向及6cell原RGB推理一致性通过，
不代表语义或质量通过。生成物体纹理较清晰，但仍把红色灯罩等指令生成为浅色外观，
背景与参考物体保持也不足。TRAIN诊断LPIPS从第2轮0.5692降至第10轮0.4658，
DEV从0.5716降至0.4864；本pilot不能简单归因于后期过拟合，也不盲目追加轮数。
完整Qwen、VAE和原生UNet底座冻结，实际训练4,983,537参数；所有旧模型保持独立。
workflow `qwen_vl_native_prior_workflow_20261028`已完成，自己的两个GPU子进程退出，
释放收据为真。240保留测试未运行。后续对这套新权重的4条TRAIN条件替换诊断已完成，
原生caption下的输出与公开SD-Turbo近似，实际Qwen条件仍生成通用浅色物体，支持优先
检查条件接口；不把清晰度提升等同于编辑正确性，也不据此开始LG主表或速度结论。
新增可选训练工具支持：复用128条TRAIN文字条件，在独立run由公开冻结SD-Turbo
产生教师latent／预览图，使用关闭空间注入的辅助latent一致性损失约束adapter，
后续配对训练可间隔保留此约束。教师不输入参考RGB，不从DEV／保留测试生成监督；
如启用，这128条是明确标记的合成教师样本，不能再称“只有文字、无教师图像”。
该新增路径目前仅代码及CPU合同测试，尚未做GPU生图反向或训练，不宣称已改善。
用户提出进一步简化指令与场景；新数据协议尚未执行，不混入现有960／96结果。

**2026-10-28原生条件诊断已完成；转入保留公开生成先验的轻量对齐。**
run `qwen_vl_native_condition_probe_20261028`只对4条TRAIN做前向，无更新，GPU已释放。
这4条上已训电子模型实际条件LPIPS均值0.1164，换成特权原生CLIP目标caption后0.4139，
再关空间条件0.4138；LG对应0.3332／0.4280／0.4458。关闭空间条件没有恢复语义，
而公开SD-Turbo纯文字oracle仍可生成更丰富外观。该小样本结果指向UNet与学习条件的
共同适配／先验偏移，不能据此宣称找到全部原因，oracle不计正式主表或可部署能力。

下一候选`configs/qwen_vl_baseline_nativeprior_20261028.json`使用完整冻结Qwen打头，
恢复公开SD-Turbo UNet原生底座，空间注入卷积重新置零；保留并继续已有adapter／
空间读出。冻结UNet底座和VAE，只训练rank8 LoRA、空间注入／读出及adapter。
先以TRAIN原生CLIP特征对齐600个廉价缓存步骤，再做10轮低学习率配对适配，
降低像素／latent回归权重并在TRAIN以0.2概率关闭空间条件。CLIP仍仅为训练教师，
推理没有新CLIP或参考VAE支路。该路径不是从随机生成骨干重新训练，也不复制旧UNet
已经共同适配的cross-attention权重；旧模型及结果保留。
`qwen_vl_native_prior_data`从8款既有TRAIN参考图构造128条颜色／形态文字对齐条件，
只增加文字教师监督，不生成128张目标图、不改变960配对图／96开发图的身份，
不使用240保留测试。原长／短完整Qwen特征按SHA复用；训练对齐目标来自TRAIN标签。
默认旧配置的初始化及运算保持原样，公开先验路径必须显式开启配置开关。

**2026-10-28原生条件／空间条件诊断入口。**`qwen_vl_native_condition_probe`仅使用固定的
4条TRAIN物体替换样本（每类2款不同目标），分别检查已训Qwen条件、原生CLIP训练目标
caption条件、关闭空间条件的原生caption，以及公开SD-Turbo纯文字生成。caption特征
是明确标记的特权训练标签，只用于定位瓶颈，不能当作可部署模型或同任务baseline；
公开纯文字生成不接收参考图，与编辑任务不同。GT图像仅用于显示和诊断指标，
不输入生成器，不执行优化，不对96开发／240保留测试做前向或选模；共享缓存可能含有
未使用的开发行，仅TRAIN索引被送入诊断。训练后曾提升到FP32的
可训练参数恢复时保持原dtype，覆盖条件的临时forward在结束／异常时均恢复。

**2026-10-27删后续TF／指令读出10轮pilot结果：已完成，尚未验收。**
顺序workflow `qwen_vl_postfree_instruction_workflow_20261027`已完整结束，两个子进程均退出GPU，
释放收据`own_gpu_released=true`。LG run `qwen_vl_postfree_focused_20261027`选择第8轮／step960，
SHA`727b302f37dd8cd5c213f773f9b66b546b39a2b57a0401541174defe157c9181`；
完整960 TRAIN／96开发的PSNR19.162805dB、LPIPS0.343111、crop LPIPS0.377389。
其整体含词嵌入1,132,149,024参数，按约定排除311,164,928后820,984,096；本轮训练34,269,681。
对应电子续训run `qwen_vl_baseline_focused_20261027`选择第10轮／step1200，
SHA`cc2d7af88b62ada0740bd2ce8a311e3e17e21c49f2abf851504a24f28cee7da3`；
PSNR21.987018、LPIPS0.254200、crop LPIPS0.330950。原100轮baseline仍单独保留，
这次续训没有实质改善整体感知均值；新LG全图指标及换物体仍明显落后。
各run完整类别×任务／历史、原始PNG与全部提示词保留；不得只选改善的crop指标宣称成功。
两组各6个原RGB＋文字核验与缓存推理的PNG逐像素一致，推理没有GT／掩码／参考VAE旁路。
从execution开始到最终report，LG约545.66秒、电子约842.22秒，包含缓存／评估／原图核验；
这是本次冻结前端的10轮适配时长，不能当作动态光学恢复训练的等价加速倍数。
新增240对仍未进行模型推理，未进入选模；当前postfree＋FocusedHead尚未正式测推理时间，
不套用旧候选1.31倍估算。当前图像仍有明显白色灯罩／靠垫纹理不遵循指令的问题，
仅简短指令与单标量读出注入没有解决；后续须先定位原生文字条件、空间条件及生成尾部的瓶颈。

**2026-10-27用户追加：删去每模态光电模块后全部TF，并优化指令遵循。**
新增候选由`qwen_vl_postfree.py`和`qwen_vl_postfree_run.py`实现，保留视觉前6／语言前4
原生TF，视觉18:24及语言24:28也实际移除，合计移除42/52个TF；保留原merger、
DeepStack位置接口、末端norm及每模态单router＋E/G，alpha仍>=0.6。
原有20个TF的候选、best/last、数据与指标原样保留；这不是静默改变原候选。
新候选关闭RMS融合里逐forward的GPU→CPU标量日志，融合输出及梯度逐元素一致测试通过，
alpha和路由在评估时显式记录，不能因此省略训练／部署中的编码、读出或融合耗时。

同RGB／seed的2条TRAIN诊断发现，单独将red改blue，原baseline的适配条件相对RMS变化
约24.2%（灯）／13.7%（靠垫），LG仅1.54%／1.82%；输出颜色仍不正确。
短句本身没有解决问题，少量TRAIN诊断不推广为全部数据上的因果结论。
候选因此让已有QueryResampler只关注经原processor验证的指令token；将这组Qwen上下文
按RMS匹配后广播注入现有视觉空间读出，只增加一个可学习缩放标量，不新增TF／宽MLP。
Qwen及光电核心在首轮快速pilot冻结，缓存实际删层后原／简短两套特征；训练时50%短句，
保留商品区分属性、场景条件与原RGB/GT，不盲目截断描述。仅TRAIN目标VAE编码监督
latent，结合区域RGB、全图和物体crop LPIPS；不以错误teacher图像作为主要KD目标。
训练已有adapter、空间读出、UNet cross-attention／空间输入卷积及最后decoder块，
VAE冻结。缓存是训练加速，部署仍必须从原RGB＋文字运行实际光电Qwen；结束核查
6个类别×任务单元的原输入推理PNG与缓存推理的一致性。新240评估集不参与此pilot。

`configs/qwen_vl_postfree_focused_20261027.json`为10轮LG快速pilot，使用已训379.56M
UNet尾部；随后`configs/qwen_vl_baseline_focused_20261027.json`作为完整Qwen＋原867.62M
UNet的同类读出／训练方法电子对照，各自独立run、不覆盖旧baseline。
10轮只用于快速检查是否恢复语义，完成不代表达到5%质量或2×速度门槛。
旧候选最终诊断计时仅约1.31×串行CUDA区间替换估算，且质量未过门槛；不是实机加速，
不能作为已经实现2×的证据。删层与读出候选还需按同边界重测及DAG核验。

**2026-10-27指令敏感性小诊断。**用户要求加快迭代并检查长prompt的影响。
`qwen_vl_instruction_probe`只取960 TRAIN中每类第一条固定参考RGB，比较长红色指令、
短红／蓝／黑指令、保留原图及简单背景指令；同参考、同seed、完整Qwen与LG使用
同一个未压缩生成尾部，不做梯度，不读取240保留测试集或把GT传入模型。
记录完整prompt、实际token数、图像token／适配条件变化、原生PNG及像素敏感度；
敏感度大不等于遵循指令，需直接检查颜色和形态。原生因果Qwen里指令前的图像token
不能看未来文字，现有空间读出依赖这组token；LG语言光学宏块为全条件联合混合，
须分别测量，不能把原生因果结论直接套给LG。此诊断尚不改变训练提示或当前计算图。
诊断只在当前顺序workflow及其GPU子进程退出后使用GPU6，不和尾部训练并发。

**2026-10-27 新增240对保留评估集（与960 TRAIN／96开发集分开）。**
用户已批准增加测试样本。CPU入口`qwen_vl_holdout`先沿完整电子baseline的
`--parent`和早期`--resume`追溯8级运行，核对每一级TRAIN及开发集的source/target商品、
已有family标识及RGBA哈希；随后打开原始ABO TEST池，只看原始视角筛选，不运行模型。
审核配置`configs/qwen_vl_holdout_240_reviewed_20261027.json`绑定原清单与两张候选联系图SHA。
4款台灯＋4款靠垫，各2个完整视角、5组场景／指令、3任务，共240对；
每类别120对、每任务80对、类别×任务各40对。240对来自8款／16视图，不冒充240个独立商品。
已记录父模型及本轮开发商品／像素与新商品池不交叉；family缺失时仍回退SKU，
不声称已穷尽近重复家族，更不声称公开Qwen/SD预训练从未见过这些商品。

筛选run `qwen_vl_holdout_240_screen_20261027`，数据run `qwen_vl_holdout_240_20261027`，
均位于本任务`runs/simulation`。`prepare`导出原生256PNG、`evaluation_pairs.json`、
仅含参考图／文字的`inference_inputs.json`、逐文件SHA及数据ZIP；不生成训练／特征PT。
本次不改变正在运行的960 TRAIN／96开发集或模型，不追加任何GPU任务。
新增240对须在冻结比较权重／选模协议后统一推理，不进入早停／梯度／针对性调参；
若之后据其结果继续优化，必须改标为开发结果。baseline只需对新增样本重跑推理，
不因评估样本数增加而重新训练。尚无该240对的模型分数或FID，数据准备完成不代表评价完成。

**2026-10-27 路由诊断与保留权重续训。**
原接续run `qwen_vl_native_hybrid_tail_20261027` 的特征蒸馏开发loss从step250的20.81
升至step2000的3685.85，TRAIN也同步增大，属于优化不稳定而非仅凭train/dev差距判过拟合。
原流程正确选择step250；图像恢复随后已改善至第6轮PSNR19.6396dB、LPIPS0.412820、
crop LPIPS0.431661，但尚未达到电子baseline的质量门槛。96条开发样本的视觉top2始终
为[1,3]、语言始终[0,1]，两模态各只有一种选择，不能据soft概率熵接近均匀声称均衡通过。
因此有记录地停止本任务自己的恢复PID2881299，保留step2880 best/last及全部结果，best SHA
`547e0e96af2a10c0277482ba9ae45018e5ca9aaf0ace8dbd5c39dbdf221192ba`；
`controlled_stop.json`校验权重未丢失、原GPU子进程已退出，未停止其他任务。
原工作流的failure表示受控切换导致的子进程退出，不冒称完成或悄悄删除失败证据。

后续run为`runs/simulation/qwen_vl_native_hybrid_phasebalanced_20261027`，具体启动／进度
以其execution/progress为准；通过工作流`--reuse-features`与`--initial-hybrid`复用原教师缓存
和上述已训练best，不重新训练电子baseline、不从随机光电模型开始。配置
`configs/qwen_vl_hybrid_phasebalanced_20261027.json`：电子LR1e-5，相位LR3e-4、
router相位LR5e-4，240步暖启动再余弦衰减；这些是待验证的训练调整，不声称已证明原因。
跨TRAIN样本EMA加入真实硬负载项，同时每20步用18条TRAIN的固定同版本光学router
做批量边际／硬负载及弱互信息正则，避免只平衡跨训练时间变化的soft EMA。
补充训练批次只用于优化，每条推理仍每模态一个router；不强迫每条指令均匀、不增加
电子分类router、不在开发测试上更新统计或梯度。新增4维load EMA仅为训练状态buffer，
不增加部署参数或光学层数。旧checkpoint只允许缺失该训练buffer，其余权重严格校验。
alpha仍[0.60,0.90]，原数据、光学几何、幅度编码、生成尾部与电子对照均不变。
后续特征蒸馏增加持续恶化早停；当前续训直接复用图像best，跳过已完成的特征蒸馏。

**2026-10-27 新电子baseline已完成100轮，接续光电工作流已启动。**
电子run `runs/simulation/qwen_vl_lamp_pillow_960_96_100epoch_20261027`，按预先指定的
完整开发96条平均LPIPS选择第50轮／step6000，权重SHA256
`64e3c1b0ae3b52e8c29a90a488acf49bed7bb64cc2fb6f896c24c9d38cf41fa6`。
全图PSNR22.323428dB、LPIPS0.251404、物体crop LPIPS0.332593；这些是开发选模指标。
完整推理3,047,916,491参数，含词嵌入311,164,928；仅TRAIN的VAE encoder34,163,664
及参考辅助头132不计入推理。其全流程／原图核验已完成，原GPU子进程释放收据为真。

| 开发任务（各32条） | PSNR dB | LPIPS↓ | 物体crop LPIPS↓ |
|---|---:|---:|---:|
| 换背景 | 22.288031 | 0.271183 | 0.381730 |
| 换物体 | 22.310653 | 0.242724 | 0.305678 |
| 同时修改 | 22.371600 | 0.240304 | 0.310370 |

第50轮后TRAIN诊断继续改善而开发感知指标回退，未因100轮末步训练分数更高而覆盖best。
定性检查仍发现指令遵循问题，例如红色灯罩指令生成白色灯罩、绿色条纹枕头生成浅色纹理；
故该版本是**已训练、固定的电子对照，不是已通过最终质量验收的成品**。后续既检查
相对指标，也检查文字语义与局部细节，不能以接近这个均值直接宣称成功。
带完整prompt的示例在该run `overview/prompt_review/{background,object,joint}_prompt_examples.png`，
全部96条浏览页为`all_two_category_with_prompts.html`。历史17M的34dB结果属于另一数据／模型
协议，不与本轮96条数值混作升降比较。

接续run为`runs/simulation/qwen_vl_native_hybrid_tail_20261027`，顺序workflow PID2840723，
实际实现源码`0cc8cc563cf996dbdd6a93a91c50a565f28f4073`；先通过本地和服务器9项CPU合同测试，
等待电子全流程结束后已进入教师特征准备，尚不代表光电质量达标。原始1056条实际语言
序列99–131 tokens，未超过224行容量，无静默截断。当前仍使用GPU6做接续任务，不能
把电子baseline的释放收据说成整个后续任务已释放。
新选中电子权重计时在该接续run `electronic_latency`：第一视觉TF→GPU RGB的附加分段诊断
平均156.183ms；无分段钩子的GPU输入就绪→GPU RGB中位156.441ms，CPU现成RGB／文字
→CPU RGB中位160.412ms、平均187.509ms（36次包含首次1105.367ms）。与旧父权重预评估
137ms不是同一次测量，最终speedup使用匹配的新计时，不能挑旧的快慢数拼倍数。

**2026-10-27 用户已批准顺序实施：电子baseline完成后，光电替换和尾部压缩。**
本轮新增实现入口为`qwen_vl_hybrid_workflow.py`，配置
`configs/qwen_vl_hybrid_eg1_20261027.json`和`configs/qwen_vl_tail_quality_20261027.json`。
工作流先等待电子baseline的完整工作流报告和GPU子进程退出，不能拿当前中间best抢跑。
随后依次做最终电子计时、原始RGB/文字的完整Qwen教师特征缓存、真实反向smoke、
2000步TRAIN特征蒸馏、最多30轮图像恢复、尾部真实反向smoke及最多30轮UNet蒸馏。
GPU6单卡串行，任一步失败停止后续步骤，保存failure及子进程释放收据；不操作实验台。
baseline现有100轮计算图／loss／数据不变。此处是实施协议，不冒称候选已恢复质量或已2×。

原TF权重从被替换位置实际移除，电子残差是轻量MLP；每个模态只注册一个光学router。
视觉256×196到224×224的reshape保持所有token的布局对应，降维投影仍是有损的可训练
接口。原始视觉DeepStack位置5/11/17和语言早期三次注入均保留。保留的Qwen注意力仍用
原生因果／位置编码；新语言光电宏块对整段已提供的图文条件联合混合，不宣称它仍是
严格因果的自回归文字生成器。当前限定B1、无padding完整提示、最多224个语言token，
不通过截断规避容量上限。后续batch优化须重新审计，不能直接喂多图拼接的视觉序列。

图像恢复只更新光电宏块，冻结生成尾部；loss包括教师最终token归一化MSE/cosine、
教师latent/RGB蒸馏、真实TRAIN平衡区域RGB和全图／物体crop LPIPS。尾部随后保留
同一VAE／scheduler、语义和空间读出，先以[256,512,1024,1024]通道、每尺度一层resnet
构造UNet，折叠teacher LoRA初始化，分别抽取skip拼接通道及GEGLU两半，保留原始
timestep embedding。用TRAIN教师输出latent/RGB/多尺度空间注意图及原配对监督蒸馏。
若恢复不足则保留本轮证据后增加E/G或调整恢复策略；若质量过关而速度不足再比较更窄尾部。
单个模态的路由边际用跨TRAIN样本EMA弱均衡，不强迫每条指令的top-2分布一致。
恢复／压缩都保存train诊断、开发测试全图／crop／类别×任务历史，持续回退时提前停止，
只保留best/last模型权重；开发测试仍不产生梯度，不称独立泛化。

**光电方案及计时预评估的审议记录（下文方案已获上述用户批准）。**
完整Qwen电子基线继续训练，不改变已运行Python计算图。用户随后补充：每个模态只允许
一个光学router，后接一对或两对Expert→Global；电子残差保留并按指定主口径不计时，
光学融合alpha下限从旧候选的0.35上调为**0.60**。后续允许开发测试定向分析和微调，
但该96条仍是开发指标，不升级为独立测试。目标是质量comparable且同口径至少2×。

已咨询窗口“端到端时间测量”（`01a172c5-819c-7202-a52c-2413af2921b0`）：
一次SLM刷新＋CMOS曝光／读出周期为`1000/1400+0.300+0.0307=1.044985714 ms`。
该值是硬件参数模型，不包含所有未测传输／OEO开销，不是实验台整机实测；不能把
T06按多视频摊薄的0.348329ms搬来作本任务每层延迟。视觉和语言有先后依赖。
每模态1 Router＋K对E/G，若每个E/G分别对应一次完整采集，则该模态`1+2K`次周期；
两模态K=1共6次=6.269914ms，均K=2共10次=10.449857ms，一边2对另一边1对共8次
=8.359886ms。专家top-2同场并行不是两次采集，额外分场／曝光／重载若实际需要必须
增记。不能把增加的传播层数都视作免费，也不能将无中间采集的多相位面与完整OEO周期
混称一层。每阶段的编码、读出、融合和共享电子计算仍计；只按用户明确范围排除并行
电子残差分支，另保留含残差的物理关键路径估算`pre+max(T_e,T_o)+fusion`。

预评估入口`qwen_vl_latency.py`：真实原RGB＋prompt运行视觉24／语言28＋生成头，
Qwen与生成器同时驻留，不复用语义缓存，不加载目标、掩码或参考VAE。GPU6 A100，
原生256，6个类别×任务单元各固定2例，B1×3轮；无额外预热、保留首请求。
源码`ae701916083cf2507ece89f975ba54ea2328ea23`，run
`runs/simulation/qwen_vl_latency_parent_20261027`。固定父权重SHA
`25b994d9959d2f5d4d46415fcc83c268eb85e9d43d81016b83ce6814266cef2d`，
与当前100轮候选同架构，但**不是最终选模权重的正式计时**。端到端CPU现成RGB／文字
→CPU RGB平均161.287ms、中位133.484ms（36次含首次1093.800ms）；GPU输入就绪→
GPU RGB平均156.373ms、中位129.910ms。无阶段钩子的主测与附加诊断分开。
第二遍不插入中间同步的阶段钩子诊断：第一视觉TF→GPU RGB平均137.085ms，全部
Qwen TF共47.079ms（视觉17.703、语言29.376），UNet58.279ms，VAE decoder11.969ms。
钩子诊断有额外Python成本，不能把137.085直接冒称无钩子的正式稳态值。
UNet＋VAE约70.248ms已经接近／超过稳态总时延一半，所以仅替换Qwen TF不足以支撑
2×的承诺。最终选中权重仍需同硬件重测，并分开报告实测电子时间及光学硬件模型。
此次仅暂停本任务PID2729222约75.924s；已核验同PID/starttime恢复，计时PID2771423退出
GPU。其他任务未动，训练继续占用GPU6，不能因此声称训练显卡已释放。

方案以不改变Qwen-first接口为前提：
1. 先保留Qwen原始patch/位置编码、视觉merger、视觉DeepStack接口、词嵌入、语言
   位置／padding／因果掩码和末端norm。初始质量优先候选保留视觉0–5及18–23，
   以光电宏块替换6–17；保留语言0–3及24–27，替换4–23，共替换32／52个原TF。
   视觉宏块中间／末端分别监督原第11／17层DeepStack输出，保留第5层原输出。
   不用旧17M的Qwen-style权重冒充本次完整Qwen层替换。
2. 每模态单router输出top-2选择，在同模态全部E层复用；E1→G1优先，恢复质量不足时
   再比较E1→G1→E2→G2。每个E/G有独立相位／读出参数，每阶段保留同输入的并行
   可训练电子残差与光学残差更新、RMS同尺度融合；alpha建议初始化0.65并约束在
   [0.60,0.90]，明确这是归一化融合系数，不等于实测光贡献占比。残差作用于特征，
   不把原始商品RGB贴回输出。需要额外干涉光路的相干旁路不在此方案中凭空假定。
3. 保持原物理ROI478／传播518、四个224专家tile、top-2和原有光学router。视觉原生
   256个token超过旧224行输入上限，不允许偷偷截断或改相位网格：拟逐token将1024维
   投影到196维，把256×196可逆重排成224×224编码，读出后反向重排／投影回256×1024。
   “可逆”仅指reshape，不声称降维投影可逆；该接口须先做形状、往返布局、教师特征恢复
   smoke审计。语言保持原token／padding顺序，使用固定224行容量；超长提示必须显式
   拒绝或另计分场，不能无声截断。硬件编码采用保零tanh(abs(field)/0.5)并保留复相位，
   BMP只round(255*a)，不按批次峰值重归一化。
4. 冻结完成训练的电子teacher，仅TRAIN做宏块入口／出口、DeepStack及最终语义／空间
   token归一化MSE＋cosine蒸馏，先固定生成尾部恢复接口，再联合小学习率微调adapter和
   必要保留层LoRA；沿用训练图像latent、平衡区域RGB和LPIPS监督，不从随机生成器重训。
   每模态／每物理槽分别记录alpha、top-2频次、熵与样本变化，均衡只作弱正则。
   恢复干净质量后逐步加入泄漏功率比例、错位、读噪、k空间和phase dropout课程，不同时
   重度开启全部扰动；干净／扰动指标分开。允许开发测试定向调优但保留协议标记。
5. UNet亦须压缩：先按同样推理精度与执行优化公平重测baseline／LG，再用电子teacher的
   中间多尺度／输出latent和RGB感知蒸馏缩减UNet通道及冗余块。VAE暂保留以保住细节。
   尾部总计争取25–35ms仅是测量驱动预算，非已达到结果；保留TF数量、光层数与尾部
   预算联合搜索，若实测不满足2×或质量门槛就不宣称成功。不能靠漏计UNet/VAE或将
   baseline与LG用不同缓存、batch、执行优化来获得倍数。
6. 质量门槛建议三任务分别检验：全图及物体crop LPIPS相对各自电子baseline最多恶化5%，
   CLIP语义相似度至少保留95%；PSNR直接报dB及MSE，不能把dB的5%当5%误差。
   若要求像素误差也不超过5%，采用MSE≤1.05×，对应PSNR下降≤0.212dB，另由用户确认。
   固定同样本、prompt和seed看轮廓／纹理，不能仅凭背景占优的总PSNR放行。

训练诊断入口`reports/qwen_vl_data_review_20261027/learning_curves.py`只读现有history及
最新评估JSON，不改变训练。保存全程global history和从启用时开始的三任务快照；旧快照
缺失不伪造。TRAIN PSNR是原先固定12条诊断而非960条均值，开发测试为完整96条，
差距扩大是拟合风险信号，不能仅凭这一项确诊过拟合。100轮不是收敛保证；需要同时看
最近多次感知／物体crop／分任务曲线、best轮次及图像。本轮截至30轮，TRAIN诊断PSNR
26.813、开发PSNR22.327，开发LPIPS从0.32314降至0.25437；物体crop从0.38516降至
0.32929，但第10轮以后约0.328–0.333波动，已提示局部细节平台，不能只看全图改善。

**2026-10-27用户已批准继续960 TRAIN＋96开发测试、100轮训练。**
当前视角审核已通过并绑定profile SHA
`64c5dd7b8d5e0f9ce4bad96cc1f250425a87e46fe780607699851b581ffbdbf3`。
自动文字初稿有一处把灯座装饰误称“三个灯泡”，已在任何梯度训练前停止本任务自己的
初始缓存PID2716726，保留原缓存／失败日志；未停止其他进程。
逐款视觉校正的灯描述见`configs/qwen_vl_960_96_caption_review_20261027.json`，
原自动描述和审核后profile分别保留，审核后缓存为
`qwen_vl_lamp_pillow_960_96_reviewed_cache_20261027`。从
`caption_reviewed_launch.json`记录的工作流PID2722419恢复，源码
`feeba4f2e0c0c21cf9f346f02a2b46360f86e1ba`；恢复前断言未有任何训练execution／best权重。
这段记录说明标注纠正和运行身份，不表示100轮已完成；进度以工作流报告为准。
审核后缓存与4步真实smoke现已完成，正式训练PID2729222已进入paired更新，首次确认
进度90／12000步，损失有限；尚无首次完整96条开发测试评估，不预报性能提升。
本地已镜像该run的`config.json`、`execution.json`及`active_data_source_receipt.json`，
GPU仍由本任务训练使用，最终释放以工作流的`gpu_release_receipt.json`为准。
本轮实施入口为`qwen_vl_expanded_workflow.py`，只用GPU6，按标注→prepare→cache→
4步真实smoke→100轮训练→原生PNG／完整prompt／语义指标／原图Qwen核验的顺序执行。
任何阶段失败即停并保存failure，GPU子进程退出后保存释放收据；不终止其他任务。
训练配置`configs/qwen_vl_lamp_pillow_960_96_100epoch_20261027.json`，batch8共12000步，
每5轮完整评估96条开发测试，按LPIPS最低选best，保留best/last。
完整Qwen视觉24＋语言28保留；输入只有参考RGB＋文字＋采样状态，额外参考VAE推理
支路明确不加入。训练VAE编码监督及参考辅助重建不构成部署时的图像旁路。

初始按alpha完整度评分的候选曾偏好俯视灯罩；调整长宽比后又混入落地灯，均未用于
训练。最终改为视觉逐款核对所有可用灯视角，人工指定配置
`configs/qwen_vl_960_96_reviewed_views_20261027.json`：全部TRAIN台灯改成底座与灯罩
可辨认的正面／轻侧面，开发测试去掉三脚灯B073P39KRV并换为完整红色台灯B076V9V7DM；
靠垫两视图用同一图案面，排除纯色背面／过薄侧面。由助手进行视觉复核，只看原始图，不按生成成绩筛图。
接续run中的`visual_review.json`标明实际审阅方式，绑定profile与联系图SHA。
`revise`保存自动候选原件及具名修改收据，不覆盖原数据、旧run或正式光电模型。
图文描述由冻结完整Qwen根据单个商品正面生成，禁止品牌与SKU；它是数据标注步骤，
推理条件只有原参考RGB＋编辑文字，不读取目标商品图或描述库。

运行目录：筛选／审核`qwen_vl_lamp_pillow_960_96_tablescreen_20261027`，新缓存
`qwen_vl_lamp_pillow_960_96_cache_20261027`，电子训练／汇总
`qwen_vl_lamp_pillow_960_96_100epoch_20261027`（均在任务`runs/simulation`）；
smoke使用同名`runs/smoke`目录。上述为启动与复现协议，是否完成以各run报告为准。
全部96条评价都明确为参与选模的开发测试；内部兼容字段`val`不表示另有验证集。
导出包含三任务各4个固定商品例子的完整prompt＋输入／GT／输出，以及全部96条浏览页、
逐图全图／物体crop LPIPS、编辑／保留区PSNR，另加离线CLIP目标描述／方向相似度。
GPU空闲及模型质量不是由完成标志保证，需分别检查收据和全部图像，之后才替换光电block。

以下保留启动前的数据规模与架构审计记录；“未启动／建议”只描述当时状态，当前运行看上文。

**历史规模调整：用户随后认为约1000对足够，2304对方案在启动前已取消。**
当前建议并据此做清晰视角预筛：TRAIN每类16款×2视图×5场景×3模式＝960对；
开发测试每类另4款×2视图×2场景×3模式＝96对，总1056对，不另设VAL。
TRAIN32款／64个真实渲染视图，开发测试8款／16视图，商品及两角色像素不交叉；
配对数不冒充独立原始图数。三任务TRAIN各320对、测试各32对，各类再均分。
`qwen_vl_expanded_data.py --stage screen`先检查alpha边界、前景面积、薄侧面及纹理，
再输出完整候选contact sheet人工视觉复核；不能只靠分数声称视角已审核完成。
其`annotate`阶段必须有绑定screen profile SHA的通过记录；本轮未启动新规模训练。
从原TRAIN／VAL角色池各自选商品，保留原角色；原VAL子集明确转作参与选模的开发测试，
不称独立泛化，不打开或挪用历史TEST。Qwen-first、无参考VAE推理支路及100轮目标不变。
后续光路可先从TRAIN固定抽48–96对做校准／电子微调，测试96对仍不参加梯度；
这是后续验证的起始预算，不预先保证实测适配足够，不要求把960对全部采到光路。

**最新数据规模纠正：下述144对／24对的100轮正式训练没有启动，只有4步smoke完成。**
用户认为独立内容太少，并明确允许不单列VAL、直接按测试结果选模。执行时必须把
该测试集标为**开发测试／用于选模的测试集**，不能冒称独立TEST泛化。内部旧`val`
路径／缓存角色若被复用，须保留原始身份，并显式写明其新用途；不与旧封闭目录的
2304条TEST混同，不擅自把历史TEST商品转成TRAIN。
本次服务器原始清单盘点（仅TRAIN／VAL元数据，未打开历史TEST）：两类各48款TRAIN
商品、各576个渲染视图，共96款／1152视图；原VAL各8款、各64视图，共16款／128视图。
此前144对仅来自12款／12视图，24对仅来自另4款／4视图，是筛选后调试子集，
不代表原数据只有这么多。一个商品图＋多个背景／指令变体不能计成多个独立商品。
扩大版本建议：TRAIN每类48款、每款2个清晰视图、每视图4个场景×三模式，共2304对；
开发测试每类4款、每款2视图、每视图2场景×三模式，共96对。两者商品角色须互斥，
训练侧192个真实渲染视图，测试侧16个，配对数分别为2304／96，并非2304／96张独立
原始商品照片。该规模是视角清单审核前的**拟议配额**，不是已生成的新数据或已开跑
配置；清晰视角／商品不足时先记录实际数量，禁止补入残缺视角凑数。旧run均保留。
下文100轮小数据配置现在只是已通过smoke的待替换方案，不能继续照它启动完整训练。

**2026-10-27最新授权：只保留Qwen作为参考RGB＋文字的图像编码入口，明确拒绝额外
参考VAE推理支路；计划台灯＋靠垫的完整100轮电子baseline实验，数据规模以上方最新
纠正为准。**
推理链保持完整冻结Qwen视觉24＋语言28→语义／空间读出→预训练单步SD-Turbo UNet
→VAE decoder。TRAIN的VAE encoder只编码监督latent，原有参考重建辅助监督也只在
TRAIN使用；没有参考VAE latent作为推理条件，没有像素贴回、目录检索或GT输入。
这次是现有单步配对监督生成器的微调，不能冒称DDPM随机时间步去噪训练或未经改动的
原生公开模型。Qwen缓存来自完整原图＋文字前向，推理参数仍计入全Qwen。

配置`configs/qwen_vl_lamp_pillow_100epoch_20261027.json`：从
`qwen_vl_ottoman_continue_20261027/best_checkpoint.pt`继续，保留其训练历史。
按类别过滤已验证252条缓存，TRAIN144／VAL24，各类6／2款商品，每类三任务均衡，
各模式TRAIN48／VAL8。坐墩、桌子不参与本轮梯度或模型评估，不重新选商品、prompt
或GT。全部身份保存在run的`active_pairs.json`，原缓存及SHA不覆盖。
100轮×18步＝1800次更新，batch8；UNet5e-6／adapter1e-5／空间5e-5、注入1e-5，
36步warmup，cosine降至起始学习率10%。每5轮对完整VAL评估，按**全图LPIPS最低**
选best，报告其PSNR而非另挑PSNR最优权重，保留best/last；100轮不等于必然收敛。
启动前执行12项CPU合同测试及4步真实反向smoke；只用GPU6，正式完成后核对自有PID。
新run `runs/simulation/qwen_vl_lamp_pillow_100epoch_20261027`，smoke为
`runs/smoke/qwen_vl_lamp_pillow_100epoch_20261027`；下方旧run保留不覆盖。

评价主张：三任务分别报告全图LPIPS↓＋物体区域LPIPS↓＋编辑区／保留区PSNR↑，
避免平均背景掩盖主体。物体LPIPS统一使用source／target前景并集的固定box，外扩8px，
两图同box裁剪后bilinear到128；这是显式命名的裁剪协议，不能混同全图LPIPS。
换背景时物体box为保留物体，换物体／联合时为两者并集；GT mask只参与离线loss／指标，
不进入生成前向。全图LPIPS net=alex，原图256；已同步写入逐样本JSON。
指令语义建议补充CLIP目标描述相似度及CLIP方向相似度，后者必须有输入／目标两段
场景描述，不能把“replace...”操作指令直接当成目标场景caption；保留CLIP偏差说明，
与盲评“指令是否执行、主体是否完整、纹理是否自然”搭配。该语义指标尚未计算，
不得把推荐或待算指标写成已测结果。LPIPS也不是唯一真值，且当前训练已有LPIPS loss。
FID／KID作为分布指标不适合由只有4款商品的24条VAL做主要结论；KID虽有无偏估计，
小样本方差及重复商品依赖仍然存在。未来独立扩充评估池后再统一计算，不靠改变指标
掩盖形变。来源：[LPIPS](https://richzhang.github.io/PerceptualSimilarity/)、
[Diffusers编辑评估](https://huggingface.co/docs/diffusers/v0.32.0/conceptual/evaluation)、
[KID](https://arxiv.org/abs/1801.01401)。先做好电子质量，再替换Qwen中的TF block；
本轮未开始任何光学结构替换。

此前2026-10-27用户复核模糊结果后，**曾暂停新训练，后续只保留台灯＋靠垫两类，移除新加的
坐墩（ottoman），桌类也不恢复**。以下三类run保留为历史开发证据；未将删除坐墩后的
24条VAL子集冒称新训练或性能提升。沿用现有商品和编辑次数时，两类为TRAIN12款／
144对，VAL另4款／24对；这只是下一轮范围，尚未准备新缓存或启动两类训练。

本次先做源码／历史结果审计和CPU排图：

- 全Qwen冻结参数2,127,532,032；UNet含适配参数867,616,708；推理合计3,047,916,491，
  但实际更新55,335,988参数。Qwen提供理解特征，不能把冻结理解参数直接等同于已对齐
  的图像生成能力。当前并非完整预训练图像编辑模型的原生前向：融合语义token经
  2048→256→77×1024 resampler；64个视觉位置经小读出头变成16×32×32条件，
  与噪声一起送入单步UNet。没有参考VAE latent、RGB skip或预训练像素重建路径。
- 同36条VAL关闭空间注入：21.105553→21.073062dB（差0.032491），说明该支路当前
  的测得贡献很小；不能由此断言原Qwen视觉信息不可恢复。Qwen图像位置本身也不等于
  VAE的可逆latent。原生merger读出对照此前未胜出，不应无证据重复该轮训练。
- 首轮从step108到432，TRAIN固定18条诊断PSNR24.331939→26.320242，
  VAL21.105553→20.657515；不是同一TRAIN全集统计，但趋势显示继续拟合没有改善VAL。
  条件校准对照也未改善。这些是已测得现象，接口失配／先验损伤是待验证解释。
- 历史`qwen_vl_turbo_full_20261026/vae_diagnostic.json`36条目标图VAE往返重建
  平均37.531884dB、范围31.467617–42.036621；不是当前36条的上限测试，但不支持
  “VAE天然只能生成现在这么糊”的解释。SD-Turbo官方建议512，当前256需要原生
  256／512对照，不能直接认定放大分辨率就解决形变。
- 目前损失是固定seed单步输出对目标latent的MSE＋RGB L1＋LPIPS＋参考辅助读出＋
  TRAIN文字条件MSE，无对抗或扩散去噪训练目标。像素配对可能压平多解细节，但本轮
  输入配方＋完整prompt分组未发现对应不同目标商品／场景的冲突；不能把均值化解释
  当作已经证实的数据错配。新VAL目标商品确实未见，文字也不确定其精确像素／视角。
- 两类24条既有输出的换物体全图PSNR21.350440，而编辑区仅17.858650dB；
  背景会稀释主体错误。完整prompt能看出颜色、轮廓和材质不跟随，不能只归结为锐度。
  背景的半透明矩形光斑有一部分原本就在合成输入／GT中，不全是模型伪影。

下一步仅建议受控诊断，不自动训练：先在小组固定TRAIN样本上比较VAE重建，以及
未经T12修改的SD-Turbo原生CLIP条件256／512结果；再固定相同UNet比较原生条件和
Qwen adapter条件，分开诊断文本／生成先验。TRAIN目标描述仅是显式标注的诊断oracle，
不得当成部署输入或泛化成绩。若原生模型清楚而Qwen条件模糊，先修接口；若两者都糊，
先查生成器工作点／训练后的先验；若VAE本身糊，再查数据和编解码。空间保真可比较
同一Qwen视觉塔更多token／多层特征，不另开图像encoder；若仍不足，再由用户判断
是否接受InstructPix2Pix式参考VAE latent条件（不是像素贴回）。仅在TRAIN小样本
保真通过后开展两类VAL训练，GAN/细节损失放在形状与条件正确之后，不用于掩盖形变。
参考：[SD-Turbo官方模型卡](https://huggingface.co/stabilityai/sd-turbo)、
[InstructPix2Pix原论文](https://arxiv.org/abs/2211.09800)、
[pix2pix-Turbo官方实现](https://github.com/GaParmar/img2img-turbo)。

提示词展示入口：`reports/qwen_vl_data_review_20261027/prompt_review.py --run <首轮run>
--output <首轮run>/overview/prompt_review`，CPU读取并校验原始PNG的SHA，无训练／GPU。
输出`lamp_prompt_examples.png`和`pillow_prompt_examples.png`各固定首样本×三模式，
每行含完整英文prompt、输入、GT、生成图与指标；`all_two_category_with_prompts.html`
为自包含的全部24条浏览页，`all_prompts_metrics.csv`保留逐样本完整prompt与区域指标。
这些都是既有三类权重的两类子集；原PNG保持256、不缩放、不锐化、不贴回GT。

用户随后要求以正常训练方法建立电子baseline、最多100 epoch，并展示更多三任务例子。
三任务仍是背景替换／物体替换／联合替换，不缩成只有物体替换。
`prompt_review.py`现额外输出`background_prompt_examples.png`、
`object_prompt_examples.png`、`joint_prompt_examples.png`，各按清单顺序保留每类两个
VAL商品的首个场景，共四例；每例完整prompt＋输入＋GT＋已有best生成，不按质量选图。
这12例与完整24条浏览页均来自原三类权重，并非拟议新baseline的结果。

100 epoch是拟议训练上限，不代表一定收敛或必然提高；仍须TRAIN训练、VAL选best、
保留best/last，独立TEST不参与调参。拟议结构保留完整Qwen视觉＋语言，额外用参考图
VAE encoder提取空间latent，将其作为生成UNet的图像条件，而非直接重建输出或贴回像素；
Qwen条件控制语义，UNet修改生成latent，VAE decoder输出整图。训练GT只用于监督，
推理仅输入参考RGB＋文字＋采样状态。用户正在澄清该额外VAE编码路径，
**结构尚未确认，100轮训练尚未启动，也未分配GPU**；不能将旧单步回归训练默认为新方案。

用户2026-10-27进一步要求取消桌类，新一轮类别选为**台灯＋靠垫＋软包坐墩
（ottoman）**。坐墩先从现有ABO CC BY 4.0原TRAIN/VAL中视觉筛选6/2款，
选封闭块状软包、完整视角及低矮脚，排除开放木架、细长腿和组合家具。
明确候选见`configs/qwen_vl_ottoman_screen_20261027.json`；CPU复现入口
`screen_third_category.py --profile ... --data ... --output ...`，预览保存在
`runs/simulation/qwen_vl_third_category_screen_20261027/selected_ottoman_candidates.png`。
这是最初筛选阶段的原始渲染前景预览，不是模型生成；随后训练结果见下文。
不把结构更简单等同于已验证生成质量。旧桌类数据、权重和下面结果原样保留作历史对照。

随后用户批准继续训练。当前新轮廓数据配置为
`configs/qwen_vl_lamp_pillow_ottoman_20261027.json`：每类6款TRAIN／2款VAL，
每商品一个清晰完整视角；台灯只选实体底座款，不再使用三脚或弯杆款。
TRAIN每视角四种输入场景×三种编辑，共216对；VAL每视角两种输入场景×三种编辑，
共36对。各类别／模式完全均衡；这不是216／36款商品，而是18／6款商品的编辑配对。
坐墩从`abo_category_screen_256_v1`继承原角色，灯／靠垫来自原256数据；使用
`qwen_vl_curated.py --category-data ottoman=<目录> --parent-data <完整历史角色清单>`
准备，禁止把父模型TRAIN暴露过的商品或解码RGB＋alpha放入新VAL，TEST保持封存。
旧父模型的桌类TRAIN历史不被擦除；只是在本轮优化/验证清单中排除桌类。
缓存和权重分别进入`runs/simulation/qwen_vl_lamp_pillow_ottoman_cache_20261027`及
`runs/simulation/qwen_vl_ottoman_continue_20261027`；配置
`configs/qwen_vl_ottoman_continue_20261027.json`从审核数据续训best继续16遍TRAIN，
完整Qwen视觉24层＋语言28层、融合后空间读出、单步SD-Turbo UNet和VAE decoder不变。
训练方法沿用父模型，无新增GAN或网络结构；VAL选best，附类别×模式及编辑区指标。
以上是启动协议；以下记录完成结果，不预先声称坐墩生成质量已达标。
针对首轮预览仍有目标颜色／花纹跟随不足，另配
`configs/qwen_vl_ottoman_condition_refine_20261027.json`，仅使用原有TRAIN条件教师
缓存校准adapter800步，再以较低UNet学习率继续8遍TRAIN，条件对齐权重.15。
这不增加推理CLIP或新网络，不改变数据／VAL；从首轮best继续，保存独立best/last及
训练前同队列重评，不预先认定这一小对照一定优于首轮。

**本轮已完成，保留首轮续训best，不采用条件校准权重。**同一36条开发VAL结果：

| 版本 | 逐图平均PSNR/dB↑ | LPIPS↓ |
| --- | ---: | ---: |
| 父权重在新36条上重新评估 | 20.803183 | 0.331757 |
| 首轮续训，best step108 | 21.105553 | 0.314567 |
| TRAIN条件校准后继续，best step108 | 20.537715 | 0.321835 |

首轮完成432步；条件校准对照完成216步，均按VAL选模、保留best/last，没有读取TEST。
首轮三个模式背景／物体／联合的PSNR分别20.996873／20.987332／21.332456dB，
均为12条；未接入光学，仍为完整冻结Qwen视觉＋语言的电子可行性版本。
空间支路关闭的同权重诊断为21.073062dB，不能当作另行训练的架构消融。
新坐墩输出出现块状主体，但仍明显模糊、变形，不能可靠保持或替换目标外观；
颜色、纹理、轮廓和指令跟随仍有明显错误，
**尚未通过生成质量验收**；不替换原封存17.03M光电部署模型。

首轮选用权重为`runs/simulation/qwen_vl_ottoman_continue_20261027/best_checkpoint.pt`，
SHA256 `25b994d9959d2f5d4d46415fcc83c268eb85e9d43d81016b83ce6814266cef2d`；
训练源码`22213bacf899f277c95656382e5b988b022bbe47`。
原图核验固定18条VAL，仅RGB＋文字重新运行完整Qwen：与训练缓存路径的生成PNG
逐像素一致；正常／灰图／错配文字的PNG域PSNR分别20.819044／18.120168／14.932350。
这些18条干预诊断不是全36条浮点指标，也不证明模型已具备开放生成能力；
目标图仅由独立离线评分程序读取，推理未输入GT、mask或训练辅助latent。

首轮run的`overview/`集中保存`all_val_generated.png`全部36张生成图、
`lamp_three_modes.png`／`pillow_three_modes.png`／`ottoman_three_modes.png`
固定首样本原图／GT／父权重／本轮best四列对照、`per_image_metrics.csv`、
`image_manifest.json`和全108张原生256PNG的`val_images.tar`。
导出入口`reports/qwen_vl_data_review_20261027/export_val.py --run <首轮run>`验证
best权重SHA及逐文件像素身份；本机解包在该run的`images/val_best/`，不做锐化或GT贴回。
条件校准run `qwen_vl_ottoman_condition_refine_20261027`保留作失败开发对照。
本轮不与旧封闭商品库的34dB、旧72／144条VAL或独立TEST指标混比。

2026-10-27按用户要求增加**结构清楚的人工审核开发数据**：
`configs/qwen_vl_curated_easy_20261027.json`逐商品固定清晰视角和无品牌外观描述；
入口`qwen_vl_curated.py --stage prepare|cache`。从原TRAIN保留36款商品×2视角，
生成432对；原VAL另留6款×2视角，生成72对；两划分三种任务均衡。
排除悬浮桌、柜体、组合家具及难辨认视角，不因模型生成好坏筛选。
换物体／联合编辑为每商品固定一个审核通过的目标视角，两个清晰视角均可作参考；
每参考视角使用两种输入背景，避免相同输入与文字被监督成不同目标视角。
所有目标仍限于本划分；完整Qwen视觉／语言处理参考RGB＋指令，GT仅用于配对监督。
`qwen_vl_spatial_run.py --data-run <审核缓存目录>`从已验证空间权重继续训练，
配置`configs/qwen_vl_curated_spatial_train_20261027.json`；不从头训练，不改变前向图。
父模型见过更大TRAIN池，本次缩小不能解释成擦除历史暴露。新72条VAL重新测量父模型
和候选，不与旧144条均值直接比较，不声称独立TEST泛化；原数据／模型／结果均保留。
审核像素、配对预览、哈希和训练产物在对应任务run中，Git仅维护源码和审核profile。

审核数据还增加一个开发对照`qwen_vl_curated_native_spatial_20261027.json`：
空间条件取同一个Qwen视觉塔merger的原生64个token，语义条件继续使用完整视觉＋
语言融合状态。缓存以校验过的原特征缓存SHA和逐样本ID绑定，使用
`qwen_vl_visual_cache.py --cache-run <审核缓存>`产生；生成前向没有第二个图像encoder。
原图推理在一次完整Qwen前向中用hook截取merger输出，额外缓存不是部署时检索库。
该对照从未加空间头的full父权重初始化，用TRAIN条件预热1500步、TRAIN参考latent
预热600步，再训练8遍审核TRAIN，条件对齐权重.5。两种视觉token组织共享同样的
空间读出参数量；新增空间注入卷积学习率1e-4，UNet其他可训练参数仍为1e-5。
该试验同时调整特征读出与条件训练，不能将收益独立归因于某一项。
不使用VAL条件教师，不以TRAIN教师替代Qwen推理，不访问TEST或实物光路。

本轮审核数据及两组训练均已完成，**尚未达到生成质量验收要求**，不替换封存模型。
统一72条开发VAL（每模式24条）结果如下，不能与旧144条VAL均值比较：

| 同一审核VAL上的版本 | 逐图平均PSNR/dB↑ | LPIPS↓ |
| --- | ---: | ---: |
| 空间父权重重新评估 | 20.956173 | 0.373219 |
| 审核数据续训，best step270 | 20.898009 | 0.343701 |
| 原生视觉读出＋条件校准，best step108 | 20.793780 | 0.354539 |

原生读出对照未优于续训版。图像已更容易辨认，但生成仍有桌类形状/类别错误；
不能将输入清理、TRAIN下降或模型完成运行说成生成问题已解决。Qwen参考重建诊断中
当前浅层读出也未重建清楚桌腿/层板，尚不能据此断言Qwen特征本身不可重建。
仅关闭原生空间注入的均值20.784455dB，当前该新支路贡献有限；它不是独立训练的消融。
另取固定18条VAL，通过公开完整Qwen重新处理原图＋文字，完成正常、灰图、同类同模式
错配文字共54张输出；GT仅在独立离线评分脚本中读取，不进入推理模块。
18条PNG域PSNR分别20.895017、17.309781、17.359866dB，不能当成72条浮点均值。
缓存与原始完整Qwen路径输出并非比特一致：18条平均像素绝对差约1/255，局部最大45/255；
原生视觉批次及数值路径可能相关，但没有完成单因素归因，不能声称原图推理与缓存完全一致。

本轮只看以下任务run，均保留原始产物：

- `runs/simulation/qwen_vl_curated_clear_cache_20261027`：审核清单、源RGB/GT、完整Qwen缓存及SHA。
- `runs/simulation/qwen_vl_curated_spatial_20261027`：审核数据续训best/last、全部72条VAL及输入清单。
- `runs/simulation/qwen_vl_curated_native_spatial_20261027`：原生读出best/last及`raw_reference_check`原图推理。
- `runs/simulation/qwen_vl_curated_comparison_20261027`：分模式/区域指标、固定9组对照和原图重建诊断。

两个绘图/核验入口为`reports/qwen_vl_data_review_20261027/compare.py`与`raw_score.py`。
较早的`qwen_vl_curated_cache_20261027`只做过准备，在训练/特征缓存前已由clear版本取代，
不能当成另一组有效模型。正式独立TEST尚未建立，本轮不能宣称最终泛化或光学实测结果。

2026-10-27启动**Qwen视觉空间读出**开发试验，入口`qwen_vl_spatial_run.py`，
配置`configs/qwen_vl_spatial_20261027.json`。完整Qwen视觉24层和语言28层保留，
256参考图＋文字仍是唯一编码输入；tokenizer、预训练Qwen层及全部融合token不裁剪。
除现有77×1024语义条件外，按公开processor验证的image token位置取出8×8视觉网格，
经过2048→128投影、两级卷积上采样及16通道输出，得到16×32×32空间条件。
UNet原4通道噪声卷积保留，新增16通道零初始化卷积注入空间条件；仍使用原生一步Euler
和VAE decoder，不恢复旧纯文本Qwen，不添加独立图像encoder、参考VAE推理支路、
GT／mask／目录索引或像素贴回。空间头4通道辅助读出仅TRAIN预测参考VAE latent。
原Qwen3-VL rot_pos_emb按merged block行列遍历，merger合并连续2×2块；
本机安装的Qwen3-VL forward无窗口置换。每条缓存的token数／image位置均由原processor
核验，实际图片processor再检查首TRAIN、首VAL和末样本，拒绝静默猜测位置。

沿用10月26日完整TRAIN／VAL身份与像素、从21.0465dB的full best继续，旧run不覆盖。
该父权重完整训练3456对；本轮发现6条粗略目标描述各指向多个TRAIN商品，
在确定性latent监督下先排除对应58条优化样本，剩余3398 TRAIN／144 VAL。
这是减少本轮歧义监督，不声称父权重没有见过这58条TRAIN样本；VAL角色不变化。
训练商品允许复用，重点是不用品牌/SKU作指令、不把未见目标商品放入TRAIN，
也不把扩大随机种子解释成已验证多样性。背景仍为原程序场景，不宣称开放域背景泛化。

先做600步TRAIN参考latent读出预热，再以batch8训练约3遍TRAIN，VAL逐图PSNR选best。
主损失初值：目标latent MSE1（前景权重3）、编辑／保留区域均衡RGB L1 .2、LPIPS .5、
参考latent辅助 .1、TRAIN条件对齐 .01。Qwen/VAE冻结，训练空间读出、条件adapter、
UNet LoRA及原文字cross-attention；冻结原噪声卷积，不新增GAN。VAE encoder仅产生
TRAIN参考／目标监督，CLIP只复用原TRAIN条件缓存，不进入推理。
记录整图／编辑区／保留区PSNR、LPIPS、空间条件置零及重新编码灰图／错配文字对照；
“视觉塔调用过”不能替代对实际贡献的核验。所有结果仍是开发VAL，不评独立TEST。
本轮仅使用用户授权A100 GPU6 UUID`GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55`，
显存上限75%，不停止其他任务；代码先发布Git再运行，完成核验自有PID释放。

2026-10-26开始独立的**纯电子可行性试验**，不替换上述封存模型。入口
`qwen_vl_turbo_run.py`，配置`configs/qwen_vl_turbo_electronic_20261026.json`。
完整公开Qwen3-VL-2B（视觉24层＋语言28层，不裁token／层）处理参考RGB和编辑指令；
保留全部融合token，经77个可训练query的两层cross-attention适配器，驱动公开SD-Turbo
UNet及原生单步Euler调度器，再由配套冻结VAE decoder输出全部像素。
推理无参考图VAE encoder、CLIP、GT、mask、检索或像素贴回。不是原Qwen本身会生图；
生成先验来自SD-Turbo，Qwen负责图文条件，适配器及UNet rank8 LoRA学习编辑。

本轮仅TRAIN/VAL开发验证：从原TRAIN选96商品×6视角，从原VAL选12商品×2视角；
每输入两种背景、两种目标、两种联合编辑，完整候选3456/144对。目标池也限于各自划分，
按source及target双角色检查SKU、上游family（存在时）和解码RGB＋alpha哈希隔离。
缺失family元数据时只能宣称SKU／精确像素隔离，不能冒称已人工排除所有近似款式；
正式独立TEST尚未建立，不读取旧TEST，不将旧T12权重或教师用于初始化。
先选类别×模式各2对，得到18 TRAIN＋18 VAL的小样本试验，不将其结果称为正式泛化。

公开冻结Qwen对选中商品生成外观短描述作为数据标注（包含VAL指令标注，但不拟合VAL），
先仅用TRAIN的目标场景描述和冻结原生CLIP文本条件对齐适配器；CLIP只作训练教师。
之后冻结VAE及Qwen，训练adapter和UNet LoRA，监督为训练目标VAE latent MSE、
0.1 RGB L1、0.1 LPIPS及0.01条件对齐。VAL只选best，不反传、不按TEST调参。
VAE encoder仅用于目标监督／重建诊断。固定噪声逐样本可复现，暂不宣称seed多样性。
先检查VAE重建、小样本过拟合和图／文条件响应，再决定是否扩展完整训练集。
这是含数十亿公开骨干参数的电子baseline，缓存不免除部署Qwen参数；不是小光电模型。

运行记录进入`runs/smoke/qwen_vl_turbo_electronic_20261026`，保存配置、命令、Git身份、
源码和公开权重SHA、划分审计、完整token缓存身份、best/last、TRAIN/VAL预览及逐图PSNR。
使用用户指定GPU1 UUID`GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d`，允许同卡共享，
本进程显存上限70%；结束核对自身PID，不终止其他任务。

**该电子方案尚未通过可用性验证。**10月26日四轮小样本试验均已完成，短程阶段未扩到
3456对完整候选、未接光、未打开TEST。下面均为同一18 TRAIN／18 VAL开发队列，
表内TRAIN与VAL来自各轮按VAL选中的同一best权重；不与旧封存指标比较。

| smoke run后缀（均为20261026） | VAL选中的step | TRAIN PSNR/dB | VAL PSNR/dB |
| --- | ---: | ---: | ---: |
| `electronic` | 120 | 19.2584 | 16.1480 |
| `aligned` | 180 | 20.7363 | 14.8750 |
| `foreground` | 600 | 22.8725 | 15.4519 |
| `cross_attention` | 200 | 23.6289 | 15.0387 |

目录完整格式为`runs/smoke/qwen_vl_turbo_<后缀>_20261026`，配置文件同名。
首轮VAE对36个开发目标的平均重建PSNR为37.5302dB。
第二轮仅把TRAIN条件对齐由240增至2400步；第三轮从同数据第二轮best继续，
增加归一化前景latent/RGB损失及LPIPS权重；第四轮从第三轮best继续，
解冻UNet已有cross-attention，并将所有样本噪声统一为seed1042。
第四轮last的TRAIN为27.1884dB，但对应VAL仅14.3632dB，不能用这个TRAIN数字
描述VAL选中best。第四轮交换跨样本的整组图文条件后VAL为15.2423dB，
比正确条件15.0387dB还高；这是整组条件错配诊断，并非独立图／文消融，
不足以宣称参考图与指令的泛化控制成立。仅凭当前结果不能确定唯一失败原因。

推理总参数3,047,511,259（约3.05B，含311,164,928词嵌入；按旧约定排除后
2,736,346,331），包括Qwen 2,127,532,032、adapter 2,918,400、UNet含LoRA
867,570,628、VAE decoder 49,490,199。前三轮可训练4,578,304，第四轮54,930,624；
解冻不增加部署参数。训练用VAE encoder 34,163,664及CLIP教师不进入推理。
各轮仅保留best/last，不把微调参数量当成部署参数量。原光电模型／实拍数据未更改。

不依赖缓存的入口为`qwen_vl_turbo_infer.py --image ... --prompt ... --checkpoint ...
--qwen ... --sd ... --output ... --seed 1042`。它只有RGB、文字、噪声输入，
不读取数据集、目标图、mask或CLIP；完整Qwen实际前向后再加载生成头。
第四轮`uncached_probe/verification.json`核验该入口与缓存的Qwen特征SHA完全一致，
单样本PNG PSNR与缓存浮点计算仅差0.00224dB（量化差异），无GT推理支路。
四轮训练与该推理核验的五个自有PID均已退出，GPU1上其他任务保留。

汇总脚本为[小样本诊断整理](reports/qwen_vl_turbo_electronic_20261026/summarize_pilot.py)，
生成第四轮`overview/pilot_summary.json`和固定三模式对照图；完整18样本网格亦在各run。
TRAIN展示last与VAL展示best时已分区注明不同权重，不伪装为同一权重的泛化差距。
这批结果是接口／训练可行性诊断，**不是已达标baseline或论文独立TEST结果**。
代码已发布main；服务器主目录另有其他任务未提交修改，故仅按公开commit检出本轮
T12路径并核验字节，不整仓pull、不切换其他运行目录，实际server HEAD与发布commit
分别记录在各run的execution／launch身份中。

小样本仅含每类2款输入、合计6款，并不能代表批准的96款TRAIN覆盖。末步能拟合
商品结构，因此后续按`configs/qwen_vl_turbo_full_20261026.json`进入完整3456 TRAIN／
144 VAL开发训练，再判断泛化；原四轮失败现象保留，不据此宣称整个架构已失败。
完整run为`runs/simulation/qwen_vl_turbo_full_20261026`，仍不打开TEST／接光。
使用同一分割与当前小样本last初始化；扩充resume必须校验父TRAIN/VAL角色与输入
均为子集合、像素哈希相同，拒绝跨角色曝光。只重对齐TRAIN条件，不拟合VAL。
批量2×累积2，2592更新正好遍历3456对三轮；每轮洗牌，VAL144只选best。
TRAIN PSNR仅评估固定18对诊断子集，不冒充3456对全量均值。CLIP教师描述明确类别，
Qwen实际输入指令不因教师描述修订而改变。缓存RGB改存无损uint8、保留全部融合token；
流式SHA已做序列化往返检查，每次评估不再重复读取数GB缓存。模型推理图保持不变。

完整覆盖阶段已完成3遍TRAIN：best为step2160，固定144 VAL的平均PSNR
21.0465dB（背景21.2968、目标20.8280、联合21.0147）；错配整组条件后16.4111dB。
因此正确条件已有明确影响，但仍存在模糊和目标属性偏差，尚不称为最终可用版本。
清晰度微调`configs/qwen_vl_turbo_semantic_refine_20261026.json`使用相同TRAIN/VAL及缓存继续：
解冻UNet已有最后一级decoder，加入TRAIN前景CLIP图文／图像语义约束及渐进的
full-frame＋前景PatchGAN。正图文对按逐样本配对计算，不平均无关的B×B相似度。
这是借鉴pix2pix-Turbo损失家族的自定义训练，并非照搬其trainer；相关官方源码
保留在已钉住Git版本的`assets/vendor/img2img-turbo-ssh`。
GT框／mask仅生成TRAIN损失裁剪，Qwen和生成器forward不接收它们。
CLIP和discriminator仅用于训练，不进入推理图；解冻已有层不增加部署参数。
微调仍以VAL PSNR选best，附LPIPS和三模式结果，TEST保持封存。

这轮864更新（再遍历TRAIN一次）已完成，run为
`runs/simulation/qwen_vl_turbo_semantic_refine_20261026`。VAL选中step432，
PSNR20.730876dB、LPIPS0.346143；相同输入／权重的微调前复算为21.046506dB、
LPIPS0.358324。感知指标略改善、PSNR下降，图像仍有目标形态错误和纹理缺失，
**尚未达到可用基线的质量要求，不因新run完成而替换原best或进入光学压缩。**
推理结构／总参数仍为3,047,511,259；本轮可训练71,651,264。
692,449参数判别器和151,277,313参数语义CLIP均仅训练使用，不计部署。
同样本完整144条逐图指标、best/last和原生256 PNG保留在服务器run；
`overview/full_cohort_summary.json`与固定九类×模式对照由
`reports/qwen_vl_turbo_electronic_20261026/summarize_full.py`生成，不挑选高分样本。

`qwen_vl_turbo_diagnose.py`还在固定18 TRAIN诊断对上对比原生目标描述CLIP条件：
公开未微调UNet的PSNR12.2898/LPIPS0.7746，微调后UNet为20.2380/0.3573，
同一微调头接Qwen条件为23.0069/0.2746。公开生成先验可以给出清晰图像，
但其构图与配对GT不同；微调后接原生文字条件也存在形变，因此不能把失败唯一归因于
Qwen→CLIP接口，或把原生教师样图当作本模型推理结果。
这些额外目标描述只来自TRAIN、无优化，不作为VAL/TEST性能或严格因果消融。
结果位于该run的`conditioning_probe`，与正式图文输入结果分开保存。
本轮best权重SHA256为`455d3a9e8f1440586ed9c6a26328271191ff9f61db607e7f90eed9dcc903d7ea`。
其`uncached_probe`用固定VAL第50项的RGB／指令重新执行完整Qwen及生成头，
Qwen融合特征SHA与缓存完全一致，原生生成PNG逐像素一致（最大uint8通道差0）；
推理及该一致性核验均不读取GT，核验比较的是已保存的模型输出。

9月21—22日旧cleanrender GAN、parallel optical、lamp场景／半Qwen替换、商品修复、
chair style及decoder optical共11份原JSON汇总已补入main，逐LF字节与原7093ec4608
和实际服务器v2源码一致。它们是旧容量／划分的历史仿真、训练耗时及PT身份，
不是当前17.03M实拍新结果；原数值和全部测速未重算，数据和权重不进Git。

原数据合同与caption缓存测试 `tests/test_dataset_contract.py`、`tests/test_feature_cache.py`
已补入main，原字节与保留服务器v2及7093ec4608逐LF一致。四项合成CPU测试验证
旧UCO3D许可／对象划分拒绝和caption首次出现顺序；不读取正式数据或缓存，
不能将这些测试通过视为当前ABO编辑数据许可审计、正式划分无泄漏或完整性能复现。

2026-10-08补齐主线缺失的`configs/optical_channel_robust_v2.json`：原字节SHA
`7babe6bb8f707b8634f3b32d36147302b3c210f0e1fcfad3d881111b98686941`，
与7093ec4608及实际服务器v2一致；这是恢复原通道配置，不是更改光学合同。
80份原历史生成／编辑／baseline工具和测试亦逐LF字节核对两处原来源后纳入main，
包括cleanrender、chair、decoder-optical、电子baseline、product repair/scene等。
它们保留旧容量及合同，不升格为当前17.03M最终部署模型。
本机离线CPU完整任务测试99项通过（13.80秒）；未加载正式PT、重算精度、训练或采集。
另外三份本机独有设备辅助工具未混入此次服务器来源结论，原位继续保留。

2026-09-23 七模型 baseline 与测速从[历史身份说明](../../reports/20260923_t12_three_task_large_small/HISTORICAL_IDENTITY_20261004.md)进入。
原报告中的“current primary/final”只指当时版本；原报告、计时及图片不改不删，
不能将其旧模型测速套用到当前17.03M实拍版本。

历史测速源码 `benchmark_product_editors`、`benchmark_three_task_bundle`、
`benchmark_unified_256` 及[9月23日紧凑编辑器原对照](reports/20260923_compact_product_editors/README.md)
已原样纳入main，十份文件与保留服务器v2源码及原提交`7093ec4608`一致。
三个入口仅核验`--help`，未重新测速。原报告的35.0606ms是电子计时28.7924ms加
固定光学估算6.2682ms，不能当成实验台端到端实测；63.4864ms是当时完整电子baseline，
不套给当前17M权重。原报告、训练/测试摘要及计时分布全部保留。

复现说明原先引用但主线缺失的 `benchmark_audited_editors` 与 `build_lab_package`
也按保留服务器v2和原提交`7093ec4608`原字节恢复。前者区分仿真CUDA计时、FFT旁路和
乐观硬件估算，不能当作实拍延迟；后者是历史20M有界振幅版本的Git源码打包器，
不是任意模型的通用部署器。两个入口仅核验`--help`，未测速、生成包、操作设备或上传。

更早的[2026-09-21纯电子单步对照](reports/20260921_electronic_baseline/README.md)
也保留原训练入口 `electronic_turbo_run`、训练实现、原配置及训练／提示推理报告。
三份源码／配置与服务器保留的v2工作目录原字节一致；此次只核验帮助与三项合成CPU合同。
它是冻结Qwen+条件适配器+冻结SD-Turbo/VAE的历史文生图对照，不是本页17M编辑版，
也不是当前完整Qwen28配对编辑baseline；原报告否决的小GAN不升格为正式方案。
没有重新训练、生成图片、测性能或修改任何正式权重／数据。

2026-10-07早期 `t12_text_to_image_20260920` 源码副本已归档退出；正式v2和审计版权重目录不动。
4968份原文件含17份独有源码／补丁、三份早期运行记录及全部旧测速均逐SHA校验保留。
其旧编辑器源码与当前主线的差异已核对，原功能及后续封存模型支持留在主线；
旧prototype不能替换本页当前17M版本。恢复归档位置及SHA在
[目录收据](../../../maintenance/storage/SERVER_WORKTREE_BALANCE_20261007.json)的`latest_t12_early_retirement_20261007`。

正式 17M 实验室捕获与电子适配的 17 份源码已逐 SHA 收录；来源、旧入口恢复点、CPU 检查和尚未迁移的环境边界见 [硬件源码登记](lab_shs8um/SOURCE_STATUS_20261006.md)。这是源码身份收敛，不是新实验或硬件运行环境验收。

已保存PNG的逐图感知/区域指标工具现在由[任务内指标入口](perceptual/README.md)管理。
仅修复旧导出包的模块改名依赖，不改变已有图片、结果或模型；旧作图及汇总源码保留。

旧 `handoffs/t12_small_baseline_share_20260928` 是17M模型对外交付快照，不是第二个开发工程。
2026-10-08两个旧交付目录 `t12_small_baseline_handoff_20260928` 与
`t12_small_baseline_share_20260928` 已完整移动到仓库
`archive/legacy_lab_projects/` 下同名目录，退出日常handoffs入口。共11,040份文件、
836,705,405字节，包含原权重、数据、导出源码、结果和所有测速，长路径完整清单与
归档后文件集合／长度／SHA全部一致，未删除载荷或释放这些资产占用。
下方“原位”和旧 `--t12-share-payloads` 路径属于历史状态；需要旧包检查／命令时
先按归档收据完整恢复原目录，不自动重建包或覆盖新目录。
本页正式17M源码与资产身份不变，服务器及实验室运行目录未移动。
见`maintenance/storage/T12_OLD_DELIVERY_ARCHIVE_20261008.json`。
其中231份源码/报告/配置载荷已逐字节或显式CRLF核对主线/原历史提交，并保留三端Git恢复身份；
原文件、数据、权重及全部测速不删，仅退出Git待同步列表。两个交付辅助脚本及两份不同的
历史说明继续可见，未被整目录忽略。精确范围见
[旧交付快照身份](../../../maintenance/storage/T12_SHARE_SOURCE_PAYLOAD_VISIBILITY_20261006.json)。
保留本地交付文件的机器可用 `python maintenance/git_safety/check_source_archives.py --t12-share-payloads`
只读检查这231份原字节是否变化；源码新克隆未附私有载荷时会失败，不自动重建或覆盖。

配对数据导出工具已归任务：`python -m LightGenV2.tasks.t12_text_to_image.export_baseline_pairs
--dataset-root <原数据目录> --instruction-cache <原指令缓存> --split test --output <新目录>`。
它不再寻找交付包里的source/assets，保留原round量化、配对字段及20736/2304/2304数量守卫，
拒绝覆盖已有输出；GT只用于导出目标，不能交给baseline推理。三项合成CPU测试不读取正式数据。
该入口使用当前main的数据实现；未做历史渲染逐像素等价核验，不能宣称重建原交付PNG。
原交付export_pairs.py及其冻结source、数据、结果继续保留，不替换历史引用。
旧stage/materialize_pairs.py所用的单split `input/target/pairs.csv`布局，可在同一主线
导出入口加`--manifest-format csv`得到；保留prompt和原配对列，不再另依赖交付source路径。
CSV不支持all，避免旧目录布局混用。原物化脚本和已导出数据仍保留；这里只验证格式合同，
未重新导出正式数据或声称历史像素逐一复核。
原stage脚本已有三端Git恢复身份：`refs/archive/reviewed-t12-stage-export-20261006`，
提交`19c6cebcae3ca3326db7605f43f106f5573288e8`，原字节SHA256
`e3cca629d94a0b21abf0c1233d2b8c8e836c546fb83be57dba1388e7b17c6a7d`。
它是历史数据物化工具，不是额外模型或新的开发分支；不执行、不删除原工具及已有PNG。

原17M单样本干净仿真另有明确入口：`python -m LightGenV2.tasks.t12_text_to_image.infer_formal_sample --help`。
显式提供原5b4f PT、数据、指令缓存和embedding缓存，固定CPU、不调用设备，拒绝其他权重及覆盖输出；
保留旧交付工具的1042+index随机种子和floor PNG量化。推理仅接收reference、prompt，不传GT。
合成测试验证种子与量化，不代表正式图像性能重放或全部资产闭包。旧infer_ours.py仍原样保留。

## 2026-09-27/28 同任务 baseline 补训与五组汇总

原`audit_checkpoint_components`、`audit_unified_optics`、`audit_unified_ablation`、
`audit_unified_seed`四份审计入口也已原样归主线，逐LF字节与保留服务器v2及`7093ec4608`
一致。四个`python -m LightGenV2.tasks.t12_text_to_image.<模块> --help`通过；原路由统计、
置零光学输出、MSE汇总和seed差异函数仅做了无数据／无输出的合成CPU检查。
参数审计依赖旧small/model、large/unet等payload结构并有原固定VAE计数，不是所有新PT的
通用预算工具；正式17M仍用本页的verify_formal_checkpoint。其余审计重放FFT仿真固定
子集，不是实际CCD采集或完整2304条TEST；alpha不是光功率占比，去光只置零expert/global
输出并保留电子／identity路径，不能另改消融定义。原脚本会写指定JSON且没有新增防覆盖，
如要复查必须使用新输出路径并绑定原资产。本轮未读取正式PT、图片、缓存或重评原指标。

### 历史紧凑／剪枝电子对照入口已补齐（2026-10-07）

9月21日紧凑BK-SDM-v2-Tiny和结构剪枝UNet的六份训练／推理源码、两份原配置、
两份原测试、四份原说明／汇总已归主线。14文件与保留服务器v2及原提交`7093ec4608`
逐LF字节一致；不修改算法、配置、原成绩或测速，原资产仍在`t12_assets`。
其原报告分别从[紧凑对照](reports/20260921_compact_electronic_baseline/README.md)及
[结构剪枝对照](reports/20260921_pruned_unet_v1/README.md)进入，原两份results.json和
紧凑版50.01ms历史4090计时保留；剪枝版未测推理延迟，不能套用紧凑版时间。

接口帮助：`python -m LightGenV2.tasks.t12_text_to_image.compact_turbo_run --help`、
`compact_turbo_infer`、`pruned_turbo_run`、`pruned_turbo_infer`（后三者使用同一模块前缀）。
四帮助及五项原合成CPU测试通过，未加载正式PT／Qwen／图片、未训练或重新测速。
这些是旧800/100/100单物体文生图及蒸馏条件缓存对照，不是当前20736配对编辑、
完整Qwen28补训baseline或17M光学部署。核验原历史需绑定原adapter、UNet、VAE和缓存，
不能直接把当前编辑数据或权重塞进这些入口；后续新实验仍从顶部正式复现页进入。

新增外部baseline：官方pix2pix-Turbo，upstream锁定 `86f54146590ffb4543c8cf85b5a36657da670924`。沿用当前20736/2304/2304配对数据及256输出；SD-Turbo预训练骨干，不从零训练主干。CLIP文本编码器冻结，微调UNet/VAE LoRA、输入卷积及官方VAE skip卷积。完整推理参数1,299,445,747，按约定排除词嵌入50,593,792后1,248,851,955；实际微调9,505,160。没有Qwen或PCA条件接口，不把它标作Qwen baseline。3轮训练在完整VAL选中step31104，固定权重TEST2304取得PSNR20.746391dB/SSIM0.740299；权重SHA256 `3a347c58affb53d8e7efc583bb5aecdaa2ac33bd316d792c12c805fa837b4c7c`。这一外部baseline明显弱于当前模型，不能宣称凭参数规模质量必然更好。来源、实现差异及命令见复现入口。

完整 Qwen28 + 原电子 UNet/adapter + 冻结 VAE baseline 已完成当前 TRAIN20736 的3轮补训，VAL2304 按逐图平均PSNR选中step15000，固定后评估TEST2304。没有缩减Qwen层数，没有用窄头缓存代替Qwen28。训练源码 `b813628ea`，导出源码 `c657f959d`，checkpoint SHA256 `599805ad3062bebf67acf3b515f0a812fb843506643e18251004f180131b4b52`。

统一256×256、浮点RGB[0,1]、逐图PSNR后平均：原17M小版仿真34.277751dB，小版电子decoder微调后EXP31.552886dB，大版仿真31.428599dB，同任务补训baseline27.260453dB。参数预算依次17.026642M、17.026642M、149.755866M、1834.345963M；含固定条件缓冲值，按约定另列冻结词嵌入311.164928M。

**小版主表仿真与EXP不是同一权重**：原5b4f权重EXP27.551289dB；微调eeec权重仿真28.890612dB。不可用34.28与31.55之差代表同权重仿真—实测差距。当前表比较已完成的模型，不等于排除不同训练历史后光学模块的因果优势。旧跨任务baseline保留为历史诊断，不纳入当前主表。

本地交付 `handoffs/t12_four_group_summary_20260927`，四模型每个包含2304张原生生成PNG、对应输入/GT、逐图CSV/JSON，代表图按类别/模式固定索引选取而非质量筛选。表格位于 `outputs/t12_four_group_summary_20260927/T12_four_group_performance.xlsx`。完整复现命令与局限见复现入口。本任务不采集光路，不使用GT贴回、超分或锐化。

新增pix2pix-Turbo原生TEST全量图片、报告、同样本拼图及五组逐图汇总位于 `handoffs/t12_pix2pix_turbo_20260927` 和 `handoffs/t12_five_group_summary_20260928`；五组表格 `outputs/t12_five_group_summary_20260928/T12_five_group_performance.xlsx`。保留旧四组文件原样，避免覆盖已引用的结果。未新增跨架构统一延迟实测，不把既有小版旧计时挪用于17M或pix2pix。

## 历史记录：2026-09-27 小版物理鲁棒性候选

### 更早的单物品条件VAE比较入口（历史baseline，不是17M实拍版）

原 `__main__/run/evaluation/training/losses` 五个模块及七份依赖YAML已原样归入main，
不再仅存在于本地未跟踪文件中。12份原字节与实际服务器旧审计目录及原Git提交
`a3f5e6fe7027e92fe29be2934750f427cdfd91e7`完全一致；
[来源与核验范围](historical_cvae_source_import_20261007.json)保留精确SHA。

只看旧接口可用 `python -m LightGenV2.tasks.t12_text_to_image --help`。
它必须显式选profile，比较早期LightGen latent生成与Qwen缓存特征+VAE方案，
**不是本页正式17M图文编辑入口，也不是当前完整Qwen28补训baseline**。
正式版本仍从本页顶部复现说明进入，不把旧CLI的lightgen名称解释为当前最终权重。
旧CLI的训练／缓存／评价会写输出，且保留原`--force`行为；复查旧实验必须绑定原资产，
只使用新的输出目录，不能直接覆盖已有run。此次只运行帮助、六配置加载及不写run的
合成CPU前向／反向合同检查，没有读取正式数据/PT、重训、评价或生成新指标。
发布后在训练服务器main `4e14b2ce` 同样通过帮助、六配置加载及两种合成CPU合同；
CUDA设备隐藏，不占GPU、不写原run。此检查不等于旧正式数据／权重复现。

下文“最新候选”仅指2026-09-27历史续训轮次，不是当前正式权重；
5496204d及更早13cf候选均不替换本页顶部的5b4f原权重和eeec适配权重。

本分支追加语言均衡/极强扰动试验，**均衡尚未解决**：所有试验实际语言top2仍为专家1/2。仅作为图像鲁棒性改善候选交付 `5b4f9a37…`（17,026,642参数，与前版预算相同），完整VAL2304：clean34.4315/.928509、stress29.3008/.892036、severe28.4256/.885481、extreme27.7628/.879783。候选保留原standardized_region_energy读出，不启用实验性log读出。来源为 `20260927_language_balance_input/last_checkpoint.pt`，不是该run未通过均衡守卫的best_checkpoint.pt；在完整VAL上仅按图像质量/干净保护人工选为备选，不能称为均衡选模成功。新增极强扰动已进入TRAIN，不再称为留出未见扰动；TEST仅固定权重后评估。交付目录 `handoffs/t12_language_balance_20260927`。

最新候选为 `5496204d…`、17,026,642参数、VAL选择step800。训练源码 `4065cbfa2`，新增文字条件空间先验，不改变光路或幅度/BMP编码。完整2304条VAL clean=34.1925/.924963、stress=27.7766/.877212、severe=27.0017/.870978、extreme=26.3708/.864833（PSNR/SSIM）；全部未实测。alpha下限.35，实际四融合值.4730/.4618/.4615/.4668；语言专家仍偏斜，残影/纹理不足仍存在。交付入口 `handoffs/t12_channel_severe_20260927`，完整配置与局限见复现报告；下面13cf版为历史候选，不是最新权重。

分支 `codex/t12-physical-robust-v2-20260927`。从 bounded-tanh 权重 SHA256 `3901e4fb…` 继续训练，不改变网络、tokenizer、Qwen-style 两层文字头或 478×478 相位区域；预算仍为 9,958,098。新权重 `13cf9a201bce42f58c19a0ff85fd11fe9940e18be036a86adf163320303edc3e`。

TRAIN 微调、VAL 选模，保留干净教师约束和干净/扰动双前向。训练通道明确将 30% 定义为干涉前的名义泄漏支路**功率比例**，使用 sqrt(0.7)/sqrt(0.3) 的复场叠加；不直接给图像加常数。CCD、错位、k-space、phase bypass dropout 仅训练/扰动评估时开启。部署幅度严格保持保零 `tanh(abs(E)/0.5)`，保留复场相位；BMP 仅 `round(255*a)`。

完整 2304 条 VAL：干净 PSNR 33.6471→33.5685，SSIM 0.920208→0.919405；组合扰动 PSNR 27.3636→27.7775，SSIM 0.867222→0.871593。选模清晰度保护阈值为 PSNR 下降≤0.2 dB、SSIM 下降≤0.002。新权重**尚未做光路实测**，不能将合成扰动改善等同于实测改善；用户另行安排实测。

## 架构与边界

输入 RGB 图像及文字，输出单次前向 256×256 RGB。原 Qwen tokenizer 和冻结词嵌入保留；文字主干是训练过的两层缩窄 Qwen-style Transformer，不是原预训练 Qwen 层。语言和视觉均调用审计版 DC20 光电模块：电子/光学同输入并行、router/top-2 experts、global、RMS 融合，历史alpha下限.4，本轮候选.35。478×478 有效光学区域，四个 224×224 专家分区；两路纯相位参数合计 958,728。

大版为 D640 文字头、条件适配器、128/256/512 窄 UNet、冻结 VAE encoder/decoder；小版为 D512 文字头与 48/96/160/224 CNN encoder/decoder，无额外 decoder refinement。冻结词嵌入 311,164,928 参数按约定单独列出，不计预算。大版神经网络 144,630,618 参数，另固定条件缓冲值 5,125,248，合计 149,755,866；小版 9,958,098。

推理不做 GT 掩码贴回或商品检索；模型使用可学习软融合。训练 GT 背景是程序合成的有限场景，目标商品来自封闭目录，因此不可宣称开放域生成或未见商品设计。种子多样性弱、专家负载偏斜仍是局限。

数据为 ABO 灯/桌/靠垫，12 款指定目标。基础 train/val/test 为 1728/192/192 图像，商品身份 144/24/24；扩展指令对 20736/2304/2304。训练包含像素、边缘、感知与局部细节损失和弱 PatchGAN；小版从大版蒸馏。判别器和感知网络仅训练使用，不计部署参数。

原实验整合分支为 `codex/t12-audited-editors-20260926`，这是历史出处，不是当前开发入口。
当前源码归入main；已有运行目录仍保护，不因此切换或覆盖。权重自带结构配置，严格加载，
历史兼容源码及必要checkpoint保留用于baseline/旧结果复查。

原17M实拍权重可从主线只读核验：
`python -m LightGenV2.tasks.t12_text_to_image.verify_formal_checkpoint --checkpoint <原5b4f权重路径>`。
该入口默认仅接受原5b4f；核验适配eeec时必须显式加`--variant decoder-adapted`。
2026-10-07实验室原eeec文件已用发布提交`3e414950bcabcaf7edfefa65f831645170201b12`
的构建器在CPU严格加载，参数17,026,642、文件SHA前后不变；不读取数据或调用SDK。
这不是重新测精度，也不是实验室入口切换。配置与光学实现仍复用T01及历史后端，
相关依赖未解除前不能删除。
