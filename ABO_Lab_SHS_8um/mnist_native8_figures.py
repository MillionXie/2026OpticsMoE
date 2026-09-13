"""Scientific plots only; never writes inference pixels or changes masks."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def run(root):
    result=json.loads((root/'result.json').read_text());history=json.loads((root/'history.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    epochs=[r['epoch'] for r in history]
    axes[0].plot(epochs,[r['validation']['accuracy']*100 for r in history],label='B validation')
    axes[0].axhline(result['baseline_A']['validation']['accuracy']*100,ls='--',label='A validation')
    axes[0].set_xlabel('Epoch');axes[0].set_ylabel('Validation accuracy (%)');axes[0].legend()
    axes[1].plot(epochs,[r['phase_delta_rms_rad'] for r in history]);axes[1].set_xlabel('Epoch');axes[1].set_ylabel('Phase update RMS (rad/epoch)')
    values=[result['baseline_A']['test']['accuracy']*100,result['test_B']['accuracy']*100]
    axes[2].bar(['A fixed resampling','B native8'],values);axes[2].set_ylim(0,100);axes[2].set_ylabel('Full test accuracy (%)')
    for i,v in enumerate(values):axes[2].text(i,v+1,f'{v:.4f}%',ha='center')
    fig.suptitle('Simulation only | MNIST 0/1/2/3 | identical 4157-image test set')
    fig.savefig(root/'training_and_test.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,5),constrained_layout=True)
    for ax,arm,name in zip(axes,['A','B'],['A_old_fixed_native8','B_native8_best']):
        phase=np.load(root/(name+'_phase_rad.npy'))
        im=ax.imshow(phase,cmap='twilight',vmin=0,vmax=2*np.pi,interpolation='nearest');ax.set_title(arm+' logical phase, radians');ax.set_axis_off()
    fig.colorbar(im,ax=axes,shrink=.7);fig.suptitle('Logical arrays, NOT display BMPs; BMPs separately flip XY and invert gray')
    fig.savefig(root/'phase_comparison.png',dpi=150);plt.close(fig)
    pair=root/'paired';ref=json.loads((pair/'pair_reference.json').read_text());z=np.load(pair/'pair_reference.npz')
    indices=[next(i for i,r in enumerate(ref['rows']) if r['label']==k and r['split']=='heldout_measurement') for k in range(4)]
    vmax=max(z['ccd_'+arm][indices].max() for arm in ['A','B'])
    fig,axes=plt.subplots(4,3,figsize=(9,12),constrained_layout=True)
    for row,i in enumerate(indices):
        axes[row,0].imshow(z['amplitude'][i],cmap='gray',vmin=0,vmax=1);axes[row,0].set_title(f"Input digit {ref['rows'][i]['label']}")
        for col,arm in enumerate(['A','B'],1):
            axes[row,col].imshow(z['ccd_'+arm][i],cmap='gray',vmin=0,vmax=vmax)
            axes[row,col].set_title(f"{arm} simulated CCD; prediction {ref['predictions'][arm][i]}")
            for k,(x0,y0,x1,y1) in enumerate(ref['detector_bounds']):
                axes[row,col].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='r',lw=.5))
        for ax in axes[row]:ax.set_axis_off()
    fig.suptitle('First held-out input per class, not selected by outcome; shared linear CCD display scale')
    fig.savefig(root/'paired_simulated_CCD.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run(a.run)
