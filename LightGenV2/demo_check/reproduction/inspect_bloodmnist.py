"""Training/validation-only RGB data audit, with no torch dependency."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    arrays={};info={};sets={}
    with np.load(a.data,allow_pickle=False) as z:
        for split in ['train','val']:
            x=z[split+'_images'];y=z[split+'_labels'].reshape(-1);assert x.dtype==np.uint8 and x.shape[1:]==(28,28,3);arrays[split]=(x,y);sets[split]={hashlib.sha256(img.tobytes()).hexdigest() for img in x};info[split]=dict(shape=list(x.shape),support=np.bincount(y,minlength=8).tolist(),range=[int(x.min()),int(x.max())],unique_images=len(sets[split]))
    labels=['Basophil','Eosinophil','Erythroblast','Immature granulocytes','Lymphocyte','Monocyte','Neutrophil','Platelet'];fig,axes=plt.subplots(2,8,figsize=(16,4.5));x,y=arrays['train']
    for k in range(8):
        im=x[np.flatnonzero(y==k)[0]]/255.;tile=np.block([[im[:,:,0],im[:,:,1]],[im[:,:,2],im.mean(2)]]);axes[0,k].imshow(im);axes[0,k].set_title(labels[k],fontsize=9);axes[1,k].imshow(tile,cmap='gray',vmin=0,vmax=1)
        for ax in axes[:,k]:ax.axis('off')
    fig.suptitle('First training image per class; fixed [R,G;B,mean] layout (before resize)');fig.tight_layout();fig.savefig(a.out/'training_examples.png',dpi=160);plt.close(fig)
    report=dict(command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),data_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),splits=info,train_val_exact_overlap=len(sets['train']&sets['val']),test_read=False,patient_groups_available=False)
    (a.out/'data_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
