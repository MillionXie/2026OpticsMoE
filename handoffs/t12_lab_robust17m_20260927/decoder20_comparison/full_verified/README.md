# Full matched TEST comparison

2304 rows, six native256 lossless PNGs per row. Join by test_index/sample_id. Reference is input; target is expected edit; baseline/tuned are matched same-CCD predictions. Prompt and source identity retained. MSE/MAE RGB[0,1], PSNR range1, SSIM Gaussian11 sigma1.5 valid windows.

Figure shortlist selects up to8 best SSIM and8 largest PSNR improvements in each category/mode for presentation only, not weight selection or population metrics. All2304 remain in all_test_comparison; aggregate every row, not this shortlist. No fake upscale. FP32 predictions, TRAIN/TEST rawCCD and best/last retained in original runs.
