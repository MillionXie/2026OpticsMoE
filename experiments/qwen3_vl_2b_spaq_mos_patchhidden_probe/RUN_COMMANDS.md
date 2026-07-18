# Commands

Run from the repository root. Commands use one line and contain no continuation backslashes.

```bash
CUDA_VISIBLE_DEVICES=6 python -m experiments.qwen3_vl_2b_spaq_mos_patchhidden_probe --config experiments/qwen3_vl_2b_spaq_mos_patchhidden_probe/configs/spaq_mos_patchhidden_probe.json --phase all
```

```bash
CUDA_VISIBLE_DEVICES=6 python -m experiments.qwen3_vl_2b_spaq_mos_patchhidden_probe --config experiments/qwen3_vl_2b_spaq_mos_patchhidden_probe/configs/spaq_mos_patchhidden_probe.json --phase extract_features
```

```bash
python -m experiments.qwen3_vl_2b_spaq_mos_patchhidden_probe --config experiments/qwen3_vl_2b_spaq_mos_patchhidden_probe/configs/spaq_mos_patchhidden_probe.json --phase train --device cpu
```

```bash
python -m pytest experiments/qwen3_vl_2b_spaq_mos_patchhidden_probe/tests -q
```

