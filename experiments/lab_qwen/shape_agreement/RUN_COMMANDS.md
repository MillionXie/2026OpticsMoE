# Shape agreement acquisition order

Run every command from the standalone repository root. CCD output must use
the generated formal homography configuration. Each stage has one manual phase BMP.

## 1. phase_00_zero

Manually load the only BMP under:

`experiments\lab_qwen\shape_agreement\phase_00_zero\phase_to_play`

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\generated\formal_hardware.yaml `
  --stage-dir experiments\lab_qwen\shape_agreement\phase_00_zero `
  --clear-output
```

## 2. phase_01_circle_0p75turn

Manually load the only BMP under:

`experiments\lab_qwen\shape_agreement\phase_01_circle_0p75turn\phase_to_play`

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\generated\formal_hardware.yaml `
  --stage-dir experiments\lab_qwen\shape_agreement\phase_01_circle_0p75turn `
  --clear-output
```

## 3. phase_02_square_0p625turn

Manually load the only BMP under:

`experiments\lab_qwen\shape_agreement\phase_02_square_0p625turn\phase_to_play`

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\generated\formal_hardware.yaml `
  --stage-dir experiments\lab_qwen\shape_agreement\phase_02_square_0p625turn `
  --clear-output
```

## 4. phase_03_ring_0p5turn

Manually load the only BMP under:

`experiments\lab_qwen\shape_agreement\phase_03_ring_0p5turn\phase_to_play`

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\generated\formal_hardware.yaml `
  --stage-dir experiments\lab_qwen\shape_agreement\phase_03_ring_0p5turn `
  --clear-output
```

## 5. phase_04_cross_0p375turn

Manually load the only BMP under:

`experiments\lab_qwen\shape_agreement\phase_04_cross_0p375turn\phase_to_play`

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\generated\formal_hardware.yaml `
  --stage-dir experiments\lab_qwen\shape_agreement\phase_04_cross_0p375turn `
  --clear-output
```

## 6. phase_05_letter_L_0p25turn

Manually load the only BMP under:

`experiments\lab_qwen\shape_agreement\phase_05_letter_L_0p25turn\phase_to_play`

```powershell
python -m experiments.hardware_sdk.workflows.acquire_folder `
  --config experiments\lab_qwen\generated\formal_hardware.yaml `
  --stage-dir experiments\lab_qwen\shape_agreement\phase_05_letter_L_0p25turn `
  --clear-output
```

## Evaluate and plot

```powershell
python -m experiments.lab_qwen.shape_agreement evaluate `
  --session-dir experiments\lab_qwen\shape_agreement
```
