"""Local phase hold ONLY. Never captures or switches automatically to next plane."""
import argparse,os,sys,time,subprocess
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bmp',type=Path,required=True);p.add_argument('--bench-root',type=Path,required=True);p.add_argument('--link-config',type=Path,required=True);a=p.parse_args()
 sys.path.insert(0,str(a.bench_root.resolve()))
 from guarded_workflow import read
 from phase_hdmi import PhaseHDMI,load_native
 from phase_owner import message_pump
 from phase_display import DisplayOrigin
 if 'blinkhdmi.exe' in subprocess.check_output(['tasklist','/FI','IMAGENAME eq BlinkHdmi.exe','/FO','CSV'],text=True).lower():raise RuntimeError('Close Blink GUI before SDK ownership')
 c=read(a.link_config);load_native(a.bmp)
 lock=a.bench_root/'results/phase_sdk_owner.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
 try:
  with DisplayOrigin(bool(c.get('phase_display_align_top',False))):
   pump=message_pump()
   with PhaseHDMI(c['phase_sdk'],c['phase_lut'],c.get('phase_settle_s',1),pixel_format=c.get('phase_pixel_format','rgba')) as sdk:
    receipt=sdk.show(a.bmp,pump=pump);print('HOLDING',a.bmp,'SHA256',receipt['phase_sha256'],'PID',os.getpid(),flush=True);last=time.monotonic()
    while True:
     pump()
     if time.monotonic()-last>=1:sdk.repeat();last=time.monotonic()
     time.sleep(.01)
 except KeyboardInterrupt:pass
 finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
