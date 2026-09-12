"""Copy only the user-supplied HDMI SDK runtime into ignored vendor storage."""
import argparse,hashlib,json,shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    out=Path(__file__).resolve().parent/'vendor/phase_hdmi';out.mkdir(parents=True,exist_ok=True)
    files=['SDK/Blink_C_wrapper.dll','SDK/HdmiDisplay.dll','SDK/ImageGen.dll','SDK/sfml-graphics-2.dll',
           'SDK/sfml-system-2.dll','SDK/sfml-window-2.dll','SDK/vcruntime140.dll','SDK/python38.dll',
           'SDK/Blink_C_wrapper.h','SDK/BlinkSdkExample.py','1920 HDMI User Manual.pdf',
           'LUT Files/19x12_8bit_linearVoltage.lut']
    manifest={}
    for rel in files:
        src=a.source/rel;dest=out/src.name
        h=hashlib.sha256(src.read_bytes()).hexdigest()
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()!=h:raise FileExistsError(dest)
        if not dest.exists():shutil.copy2(src,dest)
        manifest[dest.name]={'sha256':h,'source':str(src)}
    (out/'IMPORT_MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(out)

if __name__=='__main__':main()
