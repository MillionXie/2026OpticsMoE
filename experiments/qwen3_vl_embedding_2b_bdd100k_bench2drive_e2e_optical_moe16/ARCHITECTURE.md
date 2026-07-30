# Architecture

## 1. BDD100K pretraining graph

```text
RGB ───────────────┬─ frozen native Qwen Vision blocks
                   │  └─ final pre-merger hidden [ΣT,1024]
                   │     └─ fixed PCA [1024→224] ─ teacher target
                   │
                   └─ frozen Qwen patch/position stem [ΣT,1024]
                      └─ input adapter [1024→224]
                      └─ Optical MoE16 (one expert phase stage)
                      └─ global phase
                      └─ CCD [B,224,224]
                      └─ LN + Linear [224→224]
                         ├─ packed student target [ΣT,224]
                         └─ training-only road-structure head
```

Padded CCD token rows do not enter feature loss. Runtime `image_grid_thw` determines every valid token count and spatial shape; overflow is an error.

## 2. Deployment backbone

```text
saved = core_state_dict + recombiner_state_dict
removed = PCA + native teacher blocks + auxiliary heads
```

The saved core includes the input adapter, router, 16 expert phase masks, OEO parameters, global phase and propagation configuration. The recombiner contains the final LayerNorm and Linear(224,224).

## 3. Behavior cloning

```text
optical spatial tokens
→ mean pooling [B,224]
→ concat(speed/12, command one-hot, clipped target point/50)
→ Actor MLP
→ normalized tanh controls
→ physical steer [-1,1], throttle/brake [0,1]
```

Controls are supervised with component-wise SmoothL1 plus a small simultaneous throttle/brake penalty.

## 4. SAC

```text
BC Actor mean + learned Gaussian log_std
Twin Q critics
Entropy-regularized target
Polyak target update
```

Normalized SAC actions stay in `[-1,1]^3`; throttle and brake are decoded to `[0,1]`. Test/benchmark metrics are observational only and are never fed into updates.

## 5. No privileged deployment inputs

The deployed policy consumes only front RGB, ego speed, high-level navigation command and target point. BDD labels and simulator reward signals are training-only. Collision/offroad/red-light fields are used only by the closed-loop reward, not as Actor inputs.
