# Dataset-Switching Source Backbone

Before running prompt-only transfer, place these files in this directory:

- `source_best.pt`: best checkpoint from the dataset-switching OpticalMoE run.
- `source_config.yaml`: the dataset-switching config used to train that checkpoint.
- `source_architecture_report.json`: optional architecture report copied into each transfer run for record keeping.

The checkpoint must be a `learnable_route_moe` dataset-switching OpticalMoE with source tasks:

- `mnist`
- `fashionmnist`
- `emnist_letters`

The transfer scripts freeze the expert bank, global FC phase mask, source prompts, source readouts, and target readout. Only the new target optical prompt is trained by default.

