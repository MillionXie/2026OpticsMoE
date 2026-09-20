# T12 reproduction status

The source and CPU structure tests are reproducible now. Formal ABO feature
caches and trained checkpoints have not yet been produced. Do not report
simulation smoke output as image-generation quality.

Local verification on 2026-09-20 used the `qwen3vl-cifar10` Conda environment:
13 tests passed, both compact comparison rows completed forward/backward, and
the formal audited DC20 path completed a real 14x14-token forward to a
4x28x28 latent. The default/base Python environment was not used because its
PyTorch `c10.dll` failed to initialize.

Commands:

```powershell
python -m pytest LightGenV2/tasks/t12_text_to_image/tests -q
python -m LightGenV2.tasks.t12_text_to_image --profile smoke --phase smoke --device cpu
python -m LightGenV2.tasks.t12_text_to_image --profile lightgen --phase cache --device cuda
python -m LightGenV2.tasks.t12_text_to_image --profile lightgen --phase train --device cuda
python -m LightGenV2.tasks.t12_text_to_image --profile baseline --phase train --device cuda
```

The two formal rows must use the same three cache files. Those files record the
manifest, Qwen, VAE and preprocessing identities.
