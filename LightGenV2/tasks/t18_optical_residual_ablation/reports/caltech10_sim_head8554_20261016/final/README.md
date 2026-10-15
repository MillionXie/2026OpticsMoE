# 同一8554电子头：仿真训练与实拍交接

本次保持二/三/四层既定光学权重、数据划分与无残差100轮光学模型不变。有/无残差统一使用同一 `CCDHead`。先双方电子3轮；随后无残差电子冻结，仅残差电子继续。所有实际训练、审核与进程退出已核。电子头改变不改变CCD采集时的光学权重，同光学身份、同采集合同的数据继续保留。

## 最终本轮候选

|深度/组|电子实际/选中轮|训练%|验证%|测试%|测试macro%|测试差pp|原准确率与差距双门槛|
|---|---:|---:|---:|---:|---:|---:|---|
|2/rho03|12/12|80.5519|75.7962|83.9607|76.7039|4.2553|True|
|2/rho0|3/3|68.7254|63.6943|79.7054|71.7107|||
|3/rho03|27/27|93.4297|82.8025|88.5434|85.1801|7.2013|True|
|3/rho0|3/3|80.0263|75.1592|81.3421|75.6731|||
|4/rho03|3/3|99.6058|87.2611|93.2897|90.0809|9.3290|False|
|4/rho0|3/3|84.3627|78.3439|83.9607|79.7343|||

表中轮数是该选中候选所在训练分支的电子轮数，不是重训光学轮数。光学父版本保持：L2 E1143、L3 E123、L4最早达到既定成绩的E488；无残差光学均完整100轮，L2选98、L3/L4选100。所有表均是同一选中模型的train/val/test。全逐类见class_recall_test.csv，原始数据/PT/特征/逐样本预测未上传。

## 同一架构与实拍合同

`CCD478 → 除空间均值(clamp1e-6) → log1p → adaptive pool16×16 → 256维 → 仿真TRAIN固定mean/std(std下限.001) → Linear256×32 → ReLU → Linear32×10 → 加log原光学10窗概率(clamp1e-8) → softmax`。共8554可训练电子参数，另512个固定归一化缓冲。FP32权重33.414KiB，权重+归一化35.414KiB；PT容器大小不等于参数大小。

共享实现为caltech10_sim_head_model.py；lab读出已由其所有者改为导入同一CCDHead/ccd_features。实拍接收的是实际训练的head_state/net.*/mean/std，不随机重建、不重新估mean/std，也不另加网络。原光学固定读出以前没有可训练电子神经头；不能把旧纯光仿真成绩伪称已含此电子头。

二/三层已授权交接给“保住光路性能-实验室性能测试”；L3真实同权重logits误差0、五项CPU检查通过，L2也已真实接收核验。采集L3之后继续既定L2/L4各组，由唯一SDK所有者执行。四层未保住双门槛，交付时明确标为待接受候选；使用与E最终验证所选完全相同参数的最早A首3轮PT，不用后续容器副本混淆身份。head_handoff.csv给出每层每组精确服务器PT/SHA、光学父SHA和归一化SHA。

## 训练和复盘记录

A：双方电子3轮同初始化/架构/优化器。电子CE初始head来自零输出层，但E0只用于等价校验，从未参与选模。common3_comparison.csv保留六组实际3轮成绩。无残差电子自3轮起冻结；后续性能差使用这个真正已训练的电子对照，不能套旧纯光对照。残差L2至12轮即达双门槛。三层普通CE尝试至123轮，四层至66轮；虽有验证NLL改善，差距未保住。三层超过“少量epoch优先”预期，因此在全部收据已核的空闲边界停止旧controller，单独review_stop收据保留，未篡改旧status/权重/历史。

B：从训练过的共同E3头开始，原光学TRAIN正确样本使用.05CE+KL，错误样本CE；新9轮后选E12仍未完全保住差距。C：从B选E12继续，正确样本只有KL，错误样本CE；L3新增15、分支总27轮达到双门槛；L4新增9、分支总21轮仍选12，未达到。B/C在新测试前固定改为验证accuracy优先、balancedNLL解平局；没有用最高测试epoch选模型。新Adam重建明确记录，电子权重及固定归一化从相应父版本加载。

D/E：L4干净TRAIN只有3/761个光学错误，新增原TRAIN仿射views1/2以补充错例。D入口错误地把只有data/cacheSHA的元数据传给epoch_plan，缺少seed；在任何增强forward/电子优化前失败。原源3a4e93及两failure永久保留。E独立配置seed17，与原光学PT seed一致；CPU独立逐项重放761个order/angle/scale/translation，拒绝缺seed，核原失败SHA不变后新唯一输出执行。原D未冒称通过。E只额外提取TRAIN两视图，光学始终eval/no-grad/零优化，仿真TRAIN均值方差不变；2283行、761唯一TRAIN ID，36步/轮尾43。

E实际增强原光学正确数（clean/affine1/affine2）：[758, 752, 749] /761。

trial_review.csv完整保存A/B/C/E各最终候选与身份，不删除未改善的尝试。分支总轮数不等于整个开发实际花费：L3电子总更新轮预算=A123+B9+C15=147；最终E27路径为A首3+B9+C15。L4开发电子轮预算=A66+B9+C9+E新增轮数，最终E分支为A首3+E新增轮数；不把丢弃尝试从成本中消去。

## 独立核验与局限

新CPU解析梯度/初始化/归一化和真实特征micro1vs8检查通过。B/C/E每段保存原始loss/logits/orders，独立CPU重算CE/KL；A保存并独立核head预测、epoch顺序/Adam步数。各自核pooling、选模、head_state、ID/标签/argmax/CM/support/macro/NLL，并核光学PT SHA前后不变。A生产退出后审核；B/C/E生产进程在段内同步等待保存审核，最终实际退出/原始rc0另行核，不能把等待审核称退出。所有已完成段audit_completed/auditSHA/auditrc0与测试result/predictionSHA真一致。最终训练CPU/GPU均已退出，T18 GPU项目0，不占A100。

本轮独特电子参数digest测试次数（合并A/B/C/E）：{'L2_rho0': 1, 'L2_rho03': 4, 'L3_rho0': 1, 'L3_rho03': 49, 'L4_rho0': 1, 'L4_rho03': 22}。TEST特征仅从固定光学模型缓存一次，不等于每个深度整个开发只测试一次；同一head digest复用原结果，测试新权重多次且用来停止开发。

本任务类别已按验证知情筛选，旧测试曾观察，多轮测试反馈用于开发；单种子、残差额外训练，不能称独立未见泛化或同预算纯残差因果优势。测试1833张，Faces/Motorbikes/airplanes占91.27%，所以同时报告macro和全部逐类。L2总体准确率提升但macro76.7039低于旧纯光76.7726，未隐瞒。模型/归一化/采集版本SHA固定；旧数据、PT、claims、失败、图与历史全部保留。所有定时任务保持暂停，未创建新定时任务。

## 运行来源（与报告HEAD区分）

- A: 4b21e2fe81825c3fb571a5c5e537ed64bbe7086d，来源数226，CPU实际退出。
- B: 37a7767b3c22f5bda3c152a167697ae175e53715，来源数230，CPU实际退出。
- C: c4206bcf0ae0ee641bd7fdb6083cc0cbf20dae96，来源数234，CPU实际退出。
- E: 2f6fda51590683134c0489d24e000890bcc62fc7，来源数242，CPU实际退出。
- D失败源3a4e93a6be56fbd5ddbfea04756bcc09f9a20139，旧失败保存；E修复独立来源。

以下为选中候选精确收据：

|深度/组|头PT SHA|科学SHA|测试result SHA|
|---|---|---|---|
|2/rho03|5c8b60a7f45cf2800c5fd657ca7f4496b32c54e7b543db4927034a265a906ad3|083b8af42d4759848e1782ef877a85a7a2259daaf1ffa00f4b12d54f469ca209|579bdab2e73a50dffab6cfa0d8ce3c1b8dd6252599f27b4f825924f01dc5c5f1|
|2/rho0|1ff43adebf21699bb0688890cfe994c18f574e7e16c18fdf2367435846a1e803|1a3ecd6fad4a4bd25c0aa3a759e13ceac2a71b353793f1c8033beba0935659eb|b588602fbcb13e66d7abde31e8aa76443d704c1e0e4365bf161d06ecae57ce55|
|3/rho03|36978f6a187a190bd205b00e6351ca2fb0e31eb480e75967126bcb9930320785|af600b8bc1746e4cb3c73e66980efd0c789bb4f59d29465cc3b4c444ced8bf5f|879cd356f2ee3efa2324c1f96da2d5f731b7ce3eaf266e8606158e2728877f98|
|3/rho0|826f6c7b2375561dcfa987e44c72bb80dae3931055a14ff4ab1ff7eefecbc1b0|b40dde89df1096c9ee8a76c401e9a1af5e4d49fb4b8da2fa13e08194fb817ebd|e78d485a50b96cab25d381bde238d07a6d9147a1c017a0d7a1b7366fdf51f599|
|4/rho03|b98cc1df1964b442ae58dd7ddf4c206ef879cd333ace0c989760294adfc088f9|6e9f9138bba0fd64e142feba4dddd6c73e82d85ce5c81667fab756723dad38f4|6e52a32890342fdafd17f5402fb3e73de6fd8a241f8677bf6cf4e51e7907db82|
|4/rho0|75b328a843f0742f5469fde82da93b37b16794c6c3465a5064db4128d3e3bacc|ee8351c4ff1a0718ee42943a7332b82fd82421912ee1fd4739643e4b629d8b16|cdbeb78818d3877c57fa97f3bebd4daf293ce5c760cc51207af7fefb0ac51545|
