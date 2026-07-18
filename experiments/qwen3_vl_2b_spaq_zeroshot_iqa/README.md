# Qwen3-VL-2B SPAQ Zero-Shot IQA

This experiment asks the frozen, unmodified `Qwen/Qwen3-VL-2B-Instruct` model to directly output SPAQ quality scores. It performs no training, fine-tuning, feature-head fitting, hidden-state regression, or test-time adaptation.

Each SPAQ test image is paired with four task-specific prompts for MOS, Brightness, Colorfulness, and Contrast. Qwen runs its original processor, chat template, vision encoder, language decoder, and deterministic text generation. The generated assistant completion is parsed as a 0–100 number. Invalid, missing, and out-of-range answers are recorded as parse failures rather than silently replaced or clipped.

The experiment uses the same seed-42 90/10 image-level split as `qwen3_vl_2b_spaq_multitask_iqa`. It reports MAE, SRCC, PLCC, parse coverage, and the fraction of valid predictions within 5, 10, and 15 score points. Partial generations are appended to `zeroshot_predictions.jsonl`, so an interrupted run can resume without regenerating completed samples.

This is a zero-shot capability test, not a conventional SPAQ-trained IQA model. It answers whether the instruction-tuned VLM can directly calibrate human quality scores without dataset-specific supervision.

## Outputs

- `resolved_config.json`
- `data_split.json`
- `dataset.json`
- `model.json`
- `generation_metadata.json`
- `zeroshot_predictions.jsonl`
- `zeroshot_predictions.csv`
- `zeroshot_metrics.json`
- `zeroshot_vs_supervised_head.json` when reference metrics exist

