@echo off
cd /d E:\code\guest\2026OpticsMoE\T12_Small_Lab_SHS_8um\source
E:\code\guest\2026OpticsMoE\ABO_Lab_SHS_8um\.venv_gpu\Scripts\python.exe -u -m LightGenV2.tasks.t12_text_to_image.lab_shs8um.run_full --project E:\code\guest\2026OpticsMoE\T12_Small_Lab_SHS_8um --abo-project E:\code\guest\2026OpticsMoE\ABO_I2I_Lab_DVP_8um --output E:\code\guest\2026OpticsMoE\T12_Small_Lab_SHS_8um\runs\physical_full_test_20260926 --batch-size 6 >> E:\code\guest\2026OpticsMoE\T12_Small_Lab_SHS_8um\physical_full_test_20260926.log 2>&1
exit /b %errorlevel%
