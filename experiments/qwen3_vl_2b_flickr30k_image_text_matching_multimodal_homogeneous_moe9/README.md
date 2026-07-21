# Flickr30k image-text matching with Qwen3-VL-2B and homogeneous optical MoE9

This is a self-contained binary image-text matching experiment. It reuses the current physical and DeepStack-aware implementation from `qwen3_vl_2b_spaq_single_attribute_multimodal_homogeneous_moe9` without modifying that experiment.

Each input contains one RGB Flickr30k image and one caption. The per-sample prompt is:

```text
Determine whether the caption accurately describes the image.
Caption: {caption}
Match score:
```

The Qwen chat template is applied with `add_generation_prompt=True`. Classification uses the final language layer at the last non-padding prompt token. No generation, special token, vocabulary change, candidate list, or multi-score inference is used.

## Dataset and fixed pairing

The loader uses `nlphuji/flickr30k` through Hugging Face Datasets and understands its common layout: the public repository can expose one top-level `test` split while every record has an internal `split` field. With `validate_standard_counts=true`, it accepts only an explicitly known exact profile and checks five captions per image. The requested profile is 29,783/1,000/1,000. The repository revision validated on 2026-07-21 actually exposes the Karpathy profile 29,000/1,014/1,000 (31,014 total); that profile is accepted with a visible warning and recorded in `dataset.json`. Unknown count layouts are rejected. The loader never fabricates the 769 images absent from this repository's annotations.

The protocol in this experiment intentionally follows the latest user instruction: **there is no validation evaluation**. Official train images train the models; official test images are evaluated after every epoch and test AUROC selects the best checkpoint. The official 1,000-image validation split is checked for image-ID isolation but left unused. This makes the reported final test result selection-biased and is recorded in every relevant metrics file.

For every retained image, a stable SHA256-based choice selects one positive caption by default. A deterministic collision-checked Sattolo derangement supplies a caption from another image as the negative. Negative source IDs cannot equal the target ID, and negative text cannot equal any ground-truth caption of the target image. Positive and negative counts are equal. Pair manifests are written once under `pair_manifests/`; changed prompt, seed, sampling settings, dataset fingerprint, or manifest content is rejected instead of silently reusing stale caches.

Images are not copied. A dataset item combines the lazily loaded image, its fixed pair caption, prompt, binary label, and metadata.

## Models

Electronic teacher:

```text
RGB image + per-pair caption prompt
-> frozen full Qwen3-VL-2B vision stack
-> native vision merger and native three-point DeepStack injection
-> frozen full Qwen3-VL-2B language stack
-> final non-padding-token hidden [B,2048]
-> LayerNorm(2048) -> Linear(2048,1)
-> raw logit
```

Only the binary head is trained. `BCEWithLogitsLoss` receives raw logits. Sigmoid appears only in metric/probability computation.

The main optical student replaces both vision and language stacks with independent homogeneous MoE9 optical stacks. A diagnostic configuration replaces only vision while retaining the frozen electronic language stack. Both use exactly the same pair-manifest construction and binary head.

Student loss:

```text
L = 1.0 * normalized vision hidden MSE
  + 1.0 * normalized answer hidden MSE
  + 0.5 * SmoothL1(student raw logit, teacher raw logit)
  + 1.0 * BCEWithLogits(student raw logit, label)
  + 0.03 * router balance
  + 0.0 * router importance
```

Phase dropout is retained as a configuration feature but disabled by default.

## Automatic download and cache

The default config downloads/caches `nlphuji/flickr30k` under the configurable `data_root` and uses `https://hf-mirror.com`, which is reachable from the validated server. Set `hf_endpoint` to `null` to use the official endpoint, or provide `HF_ENDPOINT` through the environment. An existing offline cache can be used with `download=false` or `local_files_only=true`. The repository currently uses a legacy Hugging Face dataset script, so `requirements.txt` pins `datasets>=2.18,<3` and the loader explicitly trusts this configured repository. Dataset loading failures are explicit; this experiment never falls back to Flickr8k, synthetic data, or another dataset.

Teacher and processor cache identities include the dataset repository/revision/fingerprint, split, pair-manifest digest, prompt template, negative sampler/version, caption count, seed, model ID, and processor pixel budget. Sequences above 120 visual or language tokens raise an error—there is no hidden truncation or token-row remapping.

## Main outputs

- `config_resolved.json`, `environment.json`, `dataset.json`, `model.json`
- `pair_manifests/train.jsonl`, `pair_manifests/test.jsonl`, metadata and SHA256 digests
- sharded `processor_cache/` and `teacher_cache/`
- `teacher_cache/train_teacher_logits.pt`, `test_teacher_logits.pt` containing raw logits
- teacher/student histories, best-test summaries, inference JSON, prediction CSV and confusion matrices
- `metrics/comparison.json`
- teacher head and best/last optical student checkpoints

See [ARCHITECTURE.md](ARCHITECTURE.md) and [RUN_COMMANDS.md](RUN_COMMANDS.md).
