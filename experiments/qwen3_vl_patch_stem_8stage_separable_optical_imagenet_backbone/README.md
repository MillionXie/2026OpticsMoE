# P11: separable optical token/channel mixer backbone

P11 is a controlled optical-operator variant of P09. Its eight phase planes are
organized as four semantic optical macro blocks:

```text
token-axis 1-D diffraction -> channel-axis 1-D diffraction
```

Token stages propagate only along tensor rows while feature columns are ideally
relayed. Channel stages propagate only along feature columns while token rows
are ideally relayed. Before token-axis optics, the first 196 Qwen tokens are
permuted from Qwen 2x2 block-major order to true row-major order; the optical
output is returned to canonical Qwen order before electronic fusion.

The formal 90-epoch config and guarded launcher are prepared but have not been
run. See [ARCHITECTURE.md](ARCHITECTURE.md),
[OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md) and
[commands/COMMANDS.md](commands/COMMANDS.md).
