@echo off
cd /d E:\code\guest\2026OpticsMoE\OpenMoji_Robust_Rank64_SHS_20261002
set PYTHONPATH=E:\code\guest\2026OpticsMoE\OpenMoji_Robust_Rank64_SHS_20261002\source
"E:\code\guest\2026OpticsMoE\ABO_Lab_SHS_8um\.venv_gpu\Scripts\python.exe" -u -m LightGenV2.tasks.t04_openmoji_robust_ablation.lab_split_rank_tune --project . --epochs 160 --device cuda > runs/splitrank48_tune.log 2>&1
exit /b %errorlevel%
