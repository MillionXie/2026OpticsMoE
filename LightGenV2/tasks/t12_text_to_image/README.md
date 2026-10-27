# T12 图文条件商品编辑：main 统一入口

当前实拍主表使用17.03M小版：原权重 `5b4f9a37…` 与末端decoder适配权重 `eeec…`
分别保留。原权重仿真34.277751dB、未适配实拍27.551289dB；适配权重实拍31.552886dB、
其自身仿真28.890612dB，不能跨权重拼成sim-exp差距。精确指标口径和原产物见下面复现入口。

大版149.76M、小版9.96M及其他17M候选是分别保留的历史版本/必要对照，
不是几条新的长期开发分支。每套支持换背景、换目标、同时修改；高级材质及失败的
开放形态实验不作为最终版本。此页在main维护，旧分支名只记录原实验出处。

唯一入口：[复现与最终产物](reports/reproduction/README.md)。旧报告保留作历史记录，不代表当前部署入口。

**最新规模调整：用户随后认为约1000对足够，2304对方案在启动前已取消。**
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
