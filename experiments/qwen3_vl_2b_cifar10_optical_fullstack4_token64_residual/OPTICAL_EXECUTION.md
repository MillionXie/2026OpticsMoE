# Optical execution details

For every sample, token rows are handled independently. A batch never shares one optical field.

1. The hidden dimension is projected to 64.
2. LayerNorm and Softplus produce non-negative optical values.
3. The values are copied into the leading rows of a zero-filled 64-by-64 tensor.
4. Four `OpticalConversion` modules process the whole field.
5. Only the originally valid rows are read.
6. A final Linear restores the Qwen boundary hidden size.
7. If enabled, the restored optical delta is combined with the input through independently configured identity and modulated scales.

There is no bilinear interpolation in either direction. Zero-padded rows are zero before the first propagation; physical diffraction can naturally move energy into them during later propagation.
