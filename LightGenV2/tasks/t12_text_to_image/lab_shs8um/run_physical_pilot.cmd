@echo off
set PROJECT=E:\code\guest\2026OpticsMoE\T12_Small_Lab_SHS_8um
set PYTHONPATH=%PROJECT%\source
E:\code\guest\2026OpticsMoE\ABO_Lab_SHS_8um\.venv_gpu\Scripts\python.exe -u -m LightGenV2.tasks.t12_text_to_image.lab_shs8um.run_pilot --project "%PROJECT%" --abo-project E:\code\guest\2026OpticsMoE\ABO_I2I_Lab_DVP_8um --output "%PROJECT%\runs\physical_pilot_01" > "%PROJECT%\physical_pilot_01.log" 2>&1
