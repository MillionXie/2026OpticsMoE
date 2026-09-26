from pathlib import Path
import abo_full_sensor_health as health

health.OUT = Path(r'E:\code\guest\2026OpticsMoE\ABO_I2I_Lab_DVP_8um\runs\abo_i2i_20260926\health_black_white')
if __name__ == '__main__':
    health.main()
