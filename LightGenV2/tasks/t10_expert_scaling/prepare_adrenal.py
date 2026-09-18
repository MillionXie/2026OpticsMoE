"""Convert the fixed Adrenal grayscale split to the T10 RGB cache contract."""
import argparse, hashlib, json, subprocess
from pathlib import Path
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    z=np.load(a.source,allow_pickle=False); arrays={}; split_ids={}; supports={}
    for split in ['train','val','test']:
        x=np.asarray(z[split+'_images']); y=z[split+'_labels'].reshape(-1).astype(np.int64); ids=z[split+'_ids'].astype(str)
        assert x.ndim==3 and len(x)==len(y)==len(ids) and len(set(ids))==len(ids)
        x=np.clip(np.rint(x*255),0,255).astype(np.uint8)
        arrays[split+'_images']=np.repeat(x[...,None],3,-1); arrays[split+'_labels']=y; arrays[split+'_ids']=ids
        split_ids[split]=ids.tolist(); supports[split]=np.bincount(y,minlength=2).tolist()
    a.out.mkdir(parents=True,exist_ok=False); cache=a.out/'adrenal_t10_rgb.npz';np.savez(cache,**arrays)
    manifest=dict(dataset='adrenal_binary',classes=['negative','positive'],cache_sha256=sha(cache),split_ids=split_ids,supports=supports,training_ready=True,source_sha256=sha(a.source),source_license='CC-BY-4.0',rgb_conversion='grayscale repeated to RGB and float [0,1] scaled to uint8 [0,255]',official_split_preserved=True,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (a.out/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
