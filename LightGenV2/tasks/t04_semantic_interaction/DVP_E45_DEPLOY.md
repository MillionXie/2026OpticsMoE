# OpenMoji layered e45 · DVP 小相机部署

唯一模型：`runs/simulation/layered_dc30_ccdsmall_selected_e45_s73_20260925/selected_checkpoint.pt`，SHA256 `03cb861c3ac344556601eb3eb6d7d1a22b77a54d2e7e68e85d77ee30fb09eb21`。配置 `layered_scene_exp05_dc30_ccdsmall`、layered v3、epoch45。原服务器同权重1000 test Changed-cell Accuracy 0.8765；跨机RTX4060复评0.8775（数值差0.001，不能声称逐位一致）。非旧版固定网格0.8715，也非另一版0.9315。

独立工程在师弟电脑 `E:\code\guest\2026OpticsMoE\OpenMoji_Lab_DVP_8um_e45`。`runtime_exact` 来自训练commit `7121e2e0758ef3aae8c2aa29e8002a6e81e7d204`；`data`仅含1000 test、manifest与token缓存；`weights/selected_checkpoint.pt`为唯一部署权重。复用旧OpenMoji工程的冻结Qwen前端safetensors，不加载语言Transformer；硬件调用已验证的ABO DVP/Holoeye/Blink驱动。绝不覆盖旧SHS工程或旧CCD。

物理口径：振幅/相位分别1920×1080/1920×1200、8 μm，模型478×478@17 μm按宽度映射到1016×1016@8 μm，传播10 cm、532 nm。DVP相机完整5480×3648，20260924四角TL(995,172)、TR(4310,172)、BR(4303,3440)、BL(980,3435)，一次透视到478×478。正式采集固定20 ms、gain1、SLM等待240 ms、每次取第6帧，保存线性8bit PNG；不逐张拉伸、不做log或背景扣除。每层选定的相位反灰度/空间翻转与CCD朝向记录在四条试采报告里，不从旧设备配置盲抄。

1. `verify.py --project 工程 --limit 1000` 仅离线复评，不占光路。
2. `pilot.py --project 工程 --output 工程\runs\pilot4_... --exposure-us 20000 --wait-ms 240`：4条×6层、8种相位候选与CCD方向选择；这4条是校准样本，不能作为独立测试精度。
3. `full.py --project 工程 --pilot 工程\runs\pilot4_... --output 工程\runs\full1000_... --exposure-us 20000 --wait-ms 240 --batch-size 4`：1000条×6层，阶段顺序language router/expert/global → vision router/expert/global，每层的下一输入来自前层真实CCD。先校验权重/manifest/pilot，按阶段开头及每50条重拍固定参考图检查亮度与重复PCC。异常自动报错停止；原CCD不覆盖。中断后只在审计现有文件通过时对同一命令加`--resume`。
4. 查看 `runs\full1000_...\progress.json`、`health_checks.json`、`report.json`；最后一个仅在6000张CCD及电子推理全部完成后生成。`ccd\阶段\*.png`是透视校正后的实测8bit，`amplitude\阶段\*.bmp`是实际下发图，`phase\*.bmp`是本次相位。

注意：4条试采的阶段PCC约0.06–0.14，说明实测光场结构仍与仿真不同；选方向仅用于诊断与部署，不能保证全1000性能。训练中的30%未调制分量是随机扰动合同，推理时不在软件上人为再加30%。正式运行时保持三种GUI关闭，避免远程桌面改变SLM扩展屏状态。
