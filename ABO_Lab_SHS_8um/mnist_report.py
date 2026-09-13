"""Publish complete small-sample MNIST evidence, without changing inference data."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mnist_diagnostic import read,energy

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args();run=a.run
 c=read(run/'comparison.json');raw=read(run/'report.json');assert c['complete'] and raw['complete']
 m=raw['manifest'];ref=read(Path(m['reference'])/'reference.json');z=np.load(Path(m['reference'])/'reference.npz')
 rows=[r for r in c['rows'] if not r['repeat_of']];bounds=ref['detector_bounds'];conf=np.zeros((4,4),int)
 metas=[read(run/(r['name']+'.json')) for r in rows]
 for r in rows:conf[r['label'],r['measured_prediction']]+=1
 selected=[next(r for r in rows if r['label']==k) for k in range(4)]
 simmax=max(z['ccd'][r['reference_index']].max() for r in selected)
 fig,axs=plt.subplots(4,4,figsize=(13,12),constrained_layout=True)
 for i,r in enumerate(selected):
  idx=r['reference_index'];sim=z['ccd'][idx];real=np.load(run/(r['name']+'_canonical.npy'))
  axs[i,0].imshow(z['amplitude'][idx],cmap='gray',vmin=0,vmax=1);axs[i,0].set_title(f"Input: digit {r['label']} (test {ref['rows'][idx]['official_test_index']})")
  axs[i,1].imshow(sim,cmap='gray',vmin=0,vmax=simmax);axs[i,1].set_title(f"Simulation: prediction {r['simulation_prediction']}")
  axs[i,2].imshow(real,cmap='gray',vmin=0,vmax=255);axs[i,2].set_title(f"Measured: prediction {r['measured_prediction']} | PCC {r['pcc']:.3f}")
  for j in [1,2]:
   for k,(x0,y0,x1,y1) in enumerate(bounds):axs[i,j].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='red',lw=.6));axs[i,j].text(x0,y0-3,str(k),color='red',fontsize=8)
  en=np.array(r['raw_detector_energy']);se=np.array(energy(sim,bounds));xx=np.arange(4)
  axs[i,3].bar(xx-.18,se/se.sum(),.36,label='Simulation');axs[i,3].bar(xx+.18,en/en.sum(),.36,label='Measured')
  axs[i,3].set_xticks(xx);axs[i,3].set_ylim(0,1);axs[i,3].set_title('Energy fractions (plot only)');axs[i,3].legend(fontsize=7)
  for j in range(3):axs[i,j].set_axis_off()
 fig.suptitle('First held-out sample per class; not selected by outcome. Linear grayscale, no log / denoising.\nSimulation display uses one shared maximum; measured display is fixed 0..255. Raw sums determine predictions.',fontsize=11)
 fig.savefig(run/'simulation_vs_measured.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(5,4));ax.imshow(conf,cmap='Blues');ax.set_xticks(range(4));ax.set_yticks(range(4));ax.set_xlabel('Measured prediction');ax.set_ylabel('True digit');ax.set_title(f"Held-out real CCD: {sum(r['label']==r['measured_prediction'] for r in rows)}/{len(rows)}")
 for i in range(4):
  for j in range(4):ax.text(j,i,str(conf[i,j]),ha='center',va='center',color='red')
 fig.tight_layout();fig.savefig(run/'confusion_matrix.png',dpi=160);plt.close(fig)
 timing={k:dict(mean_ms=float(np.mean([v[k] for v in metas])),p95_ms=float(np.percentile([v[k] for v in metas],95))) for k in ['capture_total_ms','settle_actual_ms','final_fresh_ms']}
 summary=dict(n=len(rows),correct=sum(r['label']==r['measured_prediction'] for r in rows),accuracy=float(np.mean([r['label']==r['measured_prediction'] for r in rows])),
     simulation_accuracy=float(np.mean([r['label']==r['simulation_prediction'] for r in rows])),confusion_matrix=conf.tolist(),
     mean_pcc=float(np.mean([r['pcc'] for r in rows])),repeat_pcc=c['repeat_pcc'],max_saturation_fraction=max(r['saturation_fraction'] for r in rows),
     camera_exposures=sorted({v['camera']['ExposureTime']['value'] for v in metas}),camera_gain=sorted({v['camera']['Gain']['value'] for v in metas}),
     incomplete_frames=sum(v['incomplete'] for v in metas),timing=timing,per_class_accuracy=[float(conf[k,k]/conf[k].sum()) for k in range(4)],
     counts_include_repeat=False,formal_full_dataset_accuracy=False,scope='Fixed small sample, orientation selected on separate pilot inputs; no photometric calibration certification')
 (run/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
