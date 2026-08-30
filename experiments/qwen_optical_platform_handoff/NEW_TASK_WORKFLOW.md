# 新任务迁移清单

## A. 仿真负责人交付给硬件负责人

- [ ] task contract 已验证。
- [ ] 数据 split ID 和类别/回归分布已保存。
- [ ] 电子 baseline 与光电模型使用同一 split。
- [ ] 光学几何与真实平台一致。
- [ ] checkpoint 由 development 选择，test 未参与。
- [ ] 每层 phase BMP、compact amplitude、manifest 和 SHA 齐全。
- [ ] 训练时的 CCD normalization 有独立单元测试。
- [ ] 至少包含一组人工形状的仿真 detector 输出。

## B. 硬件负责人开始正式采集前

- [ ] 新电脑重新完成 LUT、曝光、时序、四点 homography。
- [ ] 0/中间/255 灰度无饱和且响应方向正确。
- [ ] 棋盘格/不规则形状确认双 SLM 对齐。
- [ ] 菲涅尔只用于距离、方向和 ROI 诊断，不替代形状一致性。
- [ ] 人工图案 PCC/SSIM/能量比报告已生成。
- [ ] phase SHA 与本层 payload manifest 相符。
- [ ] 采集目录为空或已明确归档，避免混入上次 CCD。

## C. 逐层微调

- [ ] 当前层 CCD 数量与 manifest 完全相等。
- [ ] 当前层及更早实测层使用 measured CCD。
- [ ] 已实测上游冻结；下游仍可训练。
- [ ] 不用 test 选 epoch。
- [ ] 下一层 amplitude 由“本层最佳 checkpoint + 本层实测 CCD”重新导出。
- [ ] 下一层 manifest 记录上一层 checkpoint SHA。

## D. 新任务完成判据

- [ ] 仿真指标、每层实测后指标和最终指标都有独立记录。
- [ ] 仿真—实测差异不只报告 PCC，至少同时报告 SSIM、NMAE、能量比和饱和率。
- [ ] 失败结果也保存配置和 SHA，不能只保留最好的一次。
