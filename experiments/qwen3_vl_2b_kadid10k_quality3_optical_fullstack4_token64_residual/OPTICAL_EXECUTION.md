# Optical execution

The optical implementation is copied from the token64 residual baseline. It uses direct token-row placement, strict zero padding, four padded angular-spectrum propagation and detection conversions, valid-row readout, and a configurable residual branch. Batch samples always use independent optical fields.

`optical_dim` must equal `optical_field_size`. Both default to 64, while propagation uses a 128-by-128 padded grid and an 8-micrometre pixel pitch.
