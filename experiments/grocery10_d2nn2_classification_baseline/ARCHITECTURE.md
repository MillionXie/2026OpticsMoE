# Architecture

## Trainable path

| Component | Shape | Trainable | Operation |
|---|---:|---:|---|
| Input amplitude | 224×224 | no | RGB luminance encoded as scalar amplitude |
| Local phase plane | 224×224 | yes | `exp(j·phase_1)` |
| Free space | 1026×1026 | no | angular spectrum, 0.10 m |
| Global phase plane | 986×986 | yes | `exp(j·phase_2)` |
| Free space | 1026×1026 | no | angular spectrum, 0.10 m |
| CCD | 986×986 | no | final `abs(field)^2` only |
| Ten regions | 3/4/3 | no | sum intensity in equal square areas |

There is no detector operation between the two phase planes. Consequently the
field remains complex and phase coherent over the complete two-plane path.

## Loss

Let `e_c` be the nonnegative energy returned directly by detector region `c`.

```text
probability_c = e_c / sum_j(e_j)
loss = -log(probability_target + eps)
prediction = argmax_c e_c
```

The logarithm exists only inside the scalar training loss, not in model
`forward()` and not as an electronic readout layer. No cosine similarity,
metric-learning embedding, teacher target, KD target, MSE hidden target or
retrieval prototype is used.

An independent optimization config can instead apply normalized full-plane
MSE. It matches the total prediction energy to a one-hot detector target and
then computes `scale * MSE(prediction, target_plane)` over the active CCD.
This changes only the scalar training objective: inference still returns the
same ten physical region energies and chooses their `argmax`.

## Detector geometry

Ten 120×120 regions are centered in three rows:

```text
      [0] [1] [2]
  [3] [4] [5] [6]
      [7] [8] [9]
```

The horizontal clear gap is 60 pixels and the vertical clear gap is 100
pixels. All regions lie inside the aligned 986×986 CCD aperture.
