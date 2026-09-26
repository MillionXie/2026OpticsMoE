@echo off
set PROJECT=E:\code\guest\2026OpticsMoE\ABO_I2I_Lab_DVP_8um
E:\code\guest\2026OpticsMoE\ABO_Lab_SHS_8um\.venv_gpu\Scripts\python.exe -u "%PROJECT%\lab_dvp8um\tune_full_projection.py" --capture "%PROJECT%\runs\abo_i2i_20260926\shs_physical2400_w240_e400_x4" --output "%PROJECT%\runs\abo_i2i_20260926\full_projection50_trainonly" --epochs 50 > "%PROJECT%\runs\abo_i2i_20260926\full_projection50_trainonly.log" 2>&1
