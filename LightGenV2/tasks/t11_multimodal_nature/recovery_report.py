"""Export figures and a compact scorecard from completed recovery runs."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=a.root/'report';out.mkdir(exist_ok=True);summary={}
 for task in ['sonyc','sen12ms']:
  fig,axs=plt.subplots(1,3,figsize=(12,3.4));summary[task]={}
  for j,arch in enumerate(['moe','d2nn']):
   root=a.root/(task+'_'+arch);result=json.loads((root/'results.json').read_text());history=json.loads((root/'history.json').read_text());summary[task][arch]=result
   key='macro_ap' if task=='sonyc' else 'accuracy';color=['#006ba4','#d85a52'][j]
   for split,style in [('train','-'),('val','--')]:axs[0].plot([r['epoch'] for r in history],[r[split][key] for r in history],style,color=color,label=arch+' '+split)
   vals=[result['metrics'][s]['default'][key] for s in ['train','val','test']];axs[1].bar(np.arange(3)+(j-.5)*.32,vals,.32,label=arch,color=color)
   route=result['metrics']['test']['default']['route_mean']
   if route is not None:axs[2].bar(range(1,5),route,color=color)
  axs[0].set(xlabel='Epoch',ylabel='Macro AP' if task=='sonyc' else 'Accuracy');axs[0].legend(fontsize=8)
  axs[1].set(xticks=range(3),xticklabels=['Train','Validation','Test'],ylim=(0,1));axs[1].legend()
  axs[2].set(xlabel='MoE expert',ylabel='Mean test power share',ylim=(0,1),xticks=range(1,5))
  fig.suptitle(task+' / seed 17 / fixed subset (not full benchmark)');fig.tight_layout();fig.savefig(out/(task+'_results.png'),dpi=180);fig.savefig(out/(task+'_results.pdf'));plt.close(fig)
  if task=='sen12ms':
   fig,axs=plt.subplots(1,2,figsize=(10,4))
   for ax,arch in zip(axs,['moe','d2nn']):
    cm=np.array(summary[task][arch]['metrics']['test']['default']['confusion_matrix']);im=ax.imshow(cm,cmap='Blues');ax.set(title=arch,xlabel='Predicted class (0-based)',ylabel='True class (0-based)',xticks=range(10),yticks=range(10));fig.colorbar(im,ax=ax)
   fig.tight_layout();fig.savefig(out/'sen12ms_confusion.png',dpi=180);plt.close(fig)
 (out/'verified_results.json').write_text(json.dumps(summary,indent=2))
 # Actual input examples, not synthetic imagery.
 import rasterio
 records=json.loads((a.root/'sen12ms_data/test_records.json').read_text());fig,axs=plt.subplots(2,4,figsize=(10,5))
 for j,r in enumerate(records[::max(1,len(records)//4)][:4]):
  cache=a.root/'sen12ms_data/paired_tiffs'
  with rasterio.open(cache/r['name']) as f:rgb=f.read([4,3,2]).astype(float)/3000
  with rasterio.open(cache/r['name'].replace('_s2_','_s1_')) as f:sar=f.read(1)
  axs[0,j].imshow(np.clip(rgb.transpose(1,2,0),0,1));axs[0,j].set_title('S2 RGB / class '+str(r['label']))
  axs[1,j].imshow(sar,cmap='gray',vmin=-25,vmax=0);axs[1,j].set_title('S1 VV (dB)')
  axs[0,j].axis('off');axs[1,j].axis('off')
 fig.suptitle('SEN12MS official source imagery / CC BY 4.0 / Schmitt et al.');fig.tight_layout();fig.savefig(out/'sen12ms_examples.png',dpi=180);plt.close(fig)
 print(out)
if __name__=='__main__':main()
