# T08 冻结前端快照：当前资产身份，不倒推历史缓存来源

2026-10-05只读服务器文件核验，没有加载模型、使用GPU、提取特征或修改缓存。
逻辑模型 `Qwen/Qwen3-VL-Embedding-2B` 的当前HF main引用为
`9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`，实际目录：

`/DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`

该快照16个文件均已完整读取计算SHA，原符号链接及blob未改动。
`model.safetensors`为4,255,140,312字节，SHA256
`c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1`，
与HF blob名称一致。config SHA256
`9172f55b0b9cce70b7f67b10c58a408ccf3ec15c587e6efd4d5f41631237fded`；
tokenizer SHA256 `def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a`。
完整逐文件私有收据：`.codex_tmp/t08_snapshot_identity_20261005.json`。

## 剩余边界

反向教师缓存确实存在且方向/提示词/CSV/顺序/张量检查已通过，见
[缓存身份](T08_TEACHER_CACHE_IDENTITY_20261005.json)。但其identity只记录逻辑模型名，
没有固定revision或权重SHA；**当前HF引用和当前模型内容不证明它就是历史生成缓存的快照**。
新入口将model_id改为绝对目录后会与历史缓存的字面身份不符，保护检查仍应拒绝。
不得为了让入口跑通改写缓存identity、静默放宽匹配或冒称绑定闭环完成。
需继续寻找历史run模型revision证据，或明确批准独立缓存重建；本次未采取后者。
原数据、权重、有效缓存及全部测速原位保留，既有运行工程不切换。
