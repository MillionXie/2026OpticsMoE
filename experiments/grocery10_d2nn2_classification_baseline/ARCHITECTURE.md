# Architecture

| Component | Shape | Trainable | Operation |
|---|---:|---:|---|
| Grayscale amplitude | 448×448 | no | fixed RGB-to-luminance conversion |
| Local phase plane | 448×448 | yes | `exp(j·phase_1)` |
| Free space | 1026×1026 | no | angular spectrum, 0.10 m |
| Global phase plane | 986×986 | yes | `exp(j·phase_2)` |
| Free space | 1026×1026 | no | angular spectrum, 0.10 m |
| Effective CCD | 986×986 | no | final `abs(field)^2` |
| Ten regions | 3 / 4 / 3 | no | sum intensity in equal square areas |

The field remains complex and phase coherent between the two phase planes.
There is no intermediate detector, normalization, activation, electronic
layer, routing operation, or similarity embedding.

For nonnegative region energy `e_c`:

```text
p_c = e_c / sum_j(e_j)
loss = -log(p_target + eps)
prediction = argmax_c(e_c)
```

The ten 120×120 detector regions are centered in the effective CCD:

```text
      [0] [1] [2]
  [3] [4] [5] [6]
      [7] [8] [9]
```
