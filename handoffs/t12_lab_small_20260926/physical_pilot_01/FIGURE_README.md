# Figure data

All rows retained, including low-scoring images. Join using sample_id; source_sample_id and full prompt identify each edit. CSV opens in Excel; JSON preserves full precision. MSE/MAE use RGB [0,1], PSNR range1, SSIM Gaussian11/sigma1.5 valid windows, RGB mean. FID/KID are dataset-level, not per-image metrics.

PNG files are lossless native decoder outputs (256x256), not contact-sheet crops or JPEGs. They are not higher-resolution reconstructions; upscaling cannot recover missing detail. outputs.pt preserves unrounded FP32 predictions. Pilot results must not be reported as full-test metrics.
