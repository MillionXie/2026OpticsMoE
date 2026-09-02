# Architecture and tensor contract

## 1. Frozen Qwen input boundary

For one video (`B` denotes batch size):

```text
MP4
  -> frames at 10/37/63/90%                         4 RGB frames
  -> centered 65%-short-side crop -> 448x448
  -> Qwen processor, min=max=200704 pixels
  -> Qwen visual.patch_embed                        [B*4*784, 1024]
  +  Qwen visual.fast_pos_embed_interpolate
  -> deterministic contiguous 2x2 mean              [B,4,196,1024]

fixed five-level prompt
  -> frozen Qwen language backbone                  [1,L,2048], mask [1,L]
  -> broadcast at Dataset access                    [B,L,2048], mask [B,L]
```

The cache uses the actual Qwen patch embed, supported interpolated positional
embedding API, and language model. The 784→196 operation is a deterministic
pre-merger shrink chosen to keep the established 196-token LGVQ student
contract. It is not the learned native Qwen merger and contains no parameters.

## 2. Electronic boundary

```text
Vision:   [B,4,196,1024] -> LayerNorm -> Linear 1024->192
                                      -> [B,4,196,192]

Language: [B,L,2048]     -> LayerNorm -> Linear 2048->192
                                      -> [B,L,192]
```

## 3. Four optical feature layers

The student width is 192 throughout. Each optical branch first maps token
channels to a nonnegative amplitude row:

```text
[...,tokens,192] -> Linear 192->224 -> Softplus
-> zero-pad token rows into a 224x224 field -> per-frame RMS normalization
```

### Vision Block 1: expert layer

- Four frames occupy four fixed 478x478 lanes inside the centered 986x986
  active area of a 1024x1024 canvas.
- Every lane has a 2x2 MoE4: expert fields are 224x224 with pitch 254.
- Therefore one propagation applies 16 phase experts in parallel.
- The router weights duplicate each frame field into its selected expert
  positions; `power_l2` makes the selected weights' squared sum exactly one.
- After 10 cm angular-spectrum propagation, each complete 478x478 lane is read
  by mean normalization, relative clipping, `log1p`, adaptive pooling
  478x478→196x196, LayerNorm, Softplus, and Linear 196→192.
- Result: `[B,4,196,192]`.

### Vision Block 2: global layer

The fused Block-1 tokens are re-encoded to amplitude. A trainable 986x986 global
phase acts on the full four-lane active plane, followed by the same propagation
and full-lane readout. Result remains `[B,4,196,192]`.

### Language Blocks 1 and 2

Only after Vision expert and Vision global have both finished, the final Vision
tokens form the multimodal bridge:

```text
final Vision per-frame mean and max over 196 tokens
          [B,4,192] + [B,4,192] -> concat [B,4,384]
          -> LayerNorm -> Linear 384->192 -> GELU
          -> four sample-dependent image tokens [B,4,192]

Language Block 1 input = prepend(image tokens, prompt tokens)
                       = [B,L+4,192], L+4 <= 224
```

Thus the strict four-layer order is Vision expert → Vision global → merger →
Language expert → Language global, which preserves the intended sequential
hardware replacement chain. The `lightweight_frame_merger` replaces the native
single-image merger only for this four-frame parallel interface; it is not
DeepStack and adds no attention.

Language uses the unchanged serial geometry: 518x518 canvas, centered 478x478
active field, 2x2 MoE4 of 224x224 experts at pitch 254. Block 1 is the expert
phase layer and Block 2 is the 478x478 global phase layer. Both preserve
`[B,L+4,192]` after CCD readout.

All feature optical stages use 532 nm, 17 µm sampling, 10 cm propagation,
k-space limiting, zero-filled (non-wrapping) input/phase/CCD translations up to
±16 pixels, and 8x8 block phase bypass/dropout during training. These are
simulation robustness perturbations; evaluation disables random perturbation.

## 4. Electronic paths and balanced fusion

Vision's electronic path is an attention-free 5x5 depthwise Conv2D token mixer
plus pointwise/channel MLP. Language uses causal 5-wide depthwise Conv1D plus a
channel MLP. There are two blocks on each branch.

At each of the four feature stages, electronic `E` and optical `O` have the
same shape. With detached RMS statistics:

```text
rE = rms(E), rO = rms(O)
En = E/rE, On = O/rO
M  = (1-alpha)*En + alpha*On
F  = rE*M/rms(M)
```

`alpha` is independently learned for Vision-expert, Vision-global,
Language-expert, and Language-global, constrained to `[0.01,0.49]` and
initialized at `0.055`. Equal RMS prevents a numerically larger electronic
feature from hiding the optical feature. `(1-alpha)` makes the contribution
definition explicit. The final `F` post-rescale preserves the electronic
branch's stable layer scale; removing it changes the distribution presented to
the next block and is therefore not part of the fair router comparison.

## 5. E1/E2/E4 and O2 routers

Electronic routing pools each 224x224 input to 14x14, applies non-affine
LayerNorm and one Linear 196→4. The same frame router is shared across all four
frames; Language uses the equivalent one-field router.

Optical routing adds no learned electronic score head:

```text
centered 224x224 amplitude + deterministic four-spot phase initialization
 -> 10 cm propagation
 -> integrate four 59x59 windows inside each 478 lane:
    rows/cols [164,223) and [255,314)
 -> subtract four-region mean / region-energy RMS
 -> softmax -> top2 -> corrected STE -> power_l2 weights
```

The four Vision lanes are evaluated in parallel on the same 1024/986/478
geometry, producing `[B,4,4]` scores. The Language router produces `[B,4]`.
Router robustness includes independent zero-filled ±16 input/phase/CCD shifts,
8x8 block phase dropout, an energy epsilon, and a capture penalty weighted 0.10.
Capture is normalized by each lane's own total 478x478 energy, not by the whole
1024 detector. Feature readout uses the full 478 lane; router decisions use only
the four 59x59 detector windows. These two ROIs have different purposes.

## 6. Lightweight temporal/output head

```text
Vision [B,4,196,192]
 -> per-frame mean+max = [B,4,384]
 -> depthwise Conv1D over four frames -> Linear 384->128
 -> temporal mean+max = [B,256]

Language [B,L+4,192]
 -> masked mean+max = [B,384] -> Linear 384->128

concat [B,384] -> LayerNorm -> Linear 384->128 -> GELU -> Linear 128->2
              -> [spatial MOS, temporal MOS]
```

Targets are standardized from training-set statistics. Training uses Smooth-L1
regression plus one pairwise ranking loss, with small router balance/importance
regularizers. No alignment logit, target, or loss exists.

## 7. Learning-rate groups

- ordinary electronic parameters: `5e-5`
- feature expert/global phase masks: `6e-3`
- electronic router parameters: `1e-3`
- O2 router phase masks: `1e-2`

Optimizer construction matches the router module prefixes before generic
`raw_*phase` names, so optical-router phases cannot accidentally fall into the
feature-phase group. A uniqueness audit rejects missing or duplicated tensors.
