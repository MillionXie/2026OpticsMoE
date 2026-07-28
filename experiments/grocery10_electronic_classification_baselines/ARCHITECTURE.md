# Architecture

| Model | Feature dimension | Direct head |
|---|---:|---|
| ResNet-18 | 512 | LayerNorm(512) → Linear(512,10) |
| EfficientNet-B0 | 1280 | LayerNorm(1280) → Linear(1280,10) |
| MobileNetV3-Small | 576 | LayerNorm(576) → Linear(576,10) |

All backbones and heads are trainable. Formal configurations use differential
learning rates:

```text
backbone learning rate = 1e-5
classification head learning rate = 3e-4
```

The loss is standard ten-class cross-entropy. Prediction is
`argmax(raw_logits)`. No gallery image is needed at training or inference.
