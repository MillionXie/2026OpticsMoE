# 人工确认替换十类：同100轮纯光配对实际启动

用户2026-10-15明确选择保持十类，换出bridge/tennis_court、加入railway_station/terrace；并直接授权发布至https://github.com/MillionXie/2026OpticsMoE，通过Git同步服务器启动。本轮只此一对，旧实验全部保留。

固定名单：desert, harbor, intersection, island, lake, meadow, railway_station, terrace, thermal_power_station, wetland。原20坐标indices[3,7,8,9,10,11,14,17,18,19]，train5600/val700，test700仅metadata。原cache与split不变，新metadata SHA d8bebc86403f4deb13b0c32ecbf79e233a16f7e33ca87b36463c50156269c884。此次新prepare已一次实际完成，不重做。

Run：resisc10_reselect_station_terrace_s17_pair100_mixup02_gpu25_20261015。
实际训练来源f405eaf2214f9da0ada739fb5cb661747f1b8933；新164源闭包由原160加runner/profile/protocol/audit四文件组成，各current/original实际一致。初始代码提交0fe1c44e0，不把后续报告HEAD当训练来源；其他T06提交和修改保留。CPU campaign父1500151，CPU queue子1500964，GPU2残差子1501738、GPU5无残差子1501804已实际运行，PID是历史快照，后续须proc/UUID核身份。两组都fresh seed17同初始化SHA f7de57b62eb33f6531128061deaa9ec2fb1bacbeea7ba02ecc50528189e60f37。

## 实际准入与首轮

- CPU新名单真实检查passed：train/val标签/原始ID身份与supports、黑图十路概率sum1、14forward、原尺寸478完整BMP逆向与CCD离线路由/五capture回放；不是训练后实拍。检查SHA b7803d7212ea0f896c65899ea71d601f7d178fcf1a0d8dd827b6e6cf4b6783c8。新claim/out全部存在不重跑。
- GPU2新两rho真实numeric passed：全部10类、macro16、tail16、micro1对8概率/capture最大误差0，最大梯度relativeL2 7.2430413e-7、全部11相位梯度finite非零；0 optimizer/test。检查SHA b560caebb1d4a4f149aa5283c119f5120ef2f556c2efde4d7548b9e9b12f296a。新claim/out保留。
- 两实际producer自己的production gate均passed，最大梯度relativeL2约6.6143e-7/7.2388e-7。不是用独立numeric收据冒充生产门槛。
- 首轮真实E1：残差val24.8571%、NLL2.065995、43.35秒；无残差val15.5714%、NLL2.281069、36.46秒。两组active2/pending0/fail0/test0，GPU2/5各一、A1000。只是首轮普通训练，不能宣布最终提升或将此差替代100轮比较。

## 冻结合同与后续

同原908552相位/四主层+光router/输入100/固定窗口联合8bit光图，无电子分类支路。MixUp.2、LS.08、TV.005、capture.05、EMA.95、AdamW lr.002/原scheduler、effective16/micro8、350Adam每轮保持。两组100轮，minimum_epochs1000明确禁旧early；每组10800新增fit秒安全上限，不到100不称完成、不自动续预算。选模仍首次最低验证EMA balancedNLL，不改用最高acc来扩大残差差距。

CPU campaign在queue实际退出且两GPU producer真实rc0、各100轮后才一次独占新保存科学。自己的audit_claim/log/cpu_report/audit.json与audit_completed SHA正确才称科学通过；失败保留不retry。不要另手动audit或重启campaign。新准入/prepare/start均已做，后续只读health及等待campaign自身审查。无新test、无电子后备重试、无A100/正式6/lab/旧图作者或新自动PDF。

本次是验证知情开发新任务，不保证换类提高准确率或差距。旧20实际114/58均time_budget，不能说其已完成同100轮；旧10最终冻结测试77.8571%属于旧名单，本轮不得套用。
