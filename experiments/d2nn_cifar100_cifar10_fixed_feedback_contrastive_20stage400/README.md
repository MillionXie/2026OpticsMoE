# CIFAR-100 → CIFAR-10 contrastive optical fixed feedback

This independent experiment replaces the class-index-bound 100-way classifier
with a transferable 128-dimensional L2-normalized embedding. It does not modify
or overwrite the completed CIFAR-100-C classification experiment.

The optical backbone remains 20 sequential OEO stages on a 400 × 400 canvas.
Every stage performs phase modulation, 5 cm propagation, square-law CCD
detection, non-affine full-plane LayerNorm, ReLU and residual reload. Each stage
has two trainable normalized residual weights. The new experiment initializes
the optical and skip branches at 0.35/0.65 instead of the old 0.10/0.90, which
made optical bypass too easy.

The final readout is:

```text
final amplitude
→ adaptive pool 20 × 20
→ affine LayerNorm(400)
→ Dropout(0.1), training only
→ Linear(400,128)
→ L2 normalization
```

No phase dropout is enabled. Dropout affects only the electronic embedding
readout and is deterministic across matched methods after resetting the same
seed.

## Transfer protocol

1. CIFAR-100 pretraining uses supervised contrastive loss over balanced P × K
   batches and two deterministic augmented views per image.
2. Actual CIFAR-10 is split into fine-tuning images, 100 prototype-support
   images per class and 200 validation images per class. Official CIFAR-10 test
   images remain test-only.
3. Fine-tuning uses SupCon plus 0.5 × leave-one-out prototype cross-entropy.
4. Evaluation constructs ten prototypes only from the fixed CIFAR-10 support
   images and classifies validation/test images by cosine similarity.
5. BP, FA-pretrained and FA-random share initialization, batches, optimizer,
   augmentation, dropout probability and seeds. NoFT builds prototypes without
   updating model parameters.

Test data never select checkpoints. The report contains both the prespecified
final epoch and the test result corresponding to the best validation checkpoint.

See [ARCHITECTURE.md](ARCHITECTURE.md) and
[commands/COMMANDS.md](commands/COMMANDS.md).
