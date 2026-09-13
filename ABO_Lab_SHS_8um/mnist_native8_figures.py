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

def hardware(root, captured):
    """Fixed first held-out digit per class; display scaling never changes metrics."""
    report=json.loads((captured/'comparison.json').read_text())
    if not report.get('complete'):raise ValueError('Both capture arms must be complete')
    ref=json.loads((root/'paired/pair_reference.json').read_text())
    with np.load(root/'paired/pair_reference.npz') as sim:
        selected={arm:[next(r for r in report['arms'][arm]['rows']
            if r['label']==k and not r['repeat_of']) for k in range(4)] for arm in ['A','B']}
        vmax=max(float(sim['ccd_'+arm][r['reference_index']].max())
                 for arm in ['A','B'] for r in selected[arm])
        fig,axes=plt.subplots(4,4,figsize=(12,12),constrained_layout=True)
        for arm,start in [('A',0),('B',2)]:
            with np.load(captured/arm/'canonical.npz') as measured:
                for row,r in enumerate(selected[arm]):
                    axes[row,start].imshow(sim['ccd_'+arm][r['reference_index']],cmap='gray',vmin=0,vmax=vmax)
                    axes[row,start].set_title(f"{arm} sim: digit {r['label']}, pred {r['simulation_prediction']}")
                    axes[row,start+1].imshow(measured[r['name']],cmap='gray',vmin=0,vmax=255)
                    axes[row,start+1].set_title(f"{arm} CCD: pred {r['prediction']}, PCC {r['pcc']:.3f}")
                    for ax in axes[row,start:start+2]:
                        for k,(x0,y0,x1,y1) in enumerate(ref['detector_bounds']):
                            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='red',lw=.7))
                            ax.text(x0,y0,str(k),color='red',fontsize=8)
                        ax.set_axis_off()
        fig.suptitle('First predeclared held-out sample per class; no outcome selection\n'
                     'Simulation: shared linear scale | measured: fixed 0-255, no contrast normalization')
        fig.savefig(captured/'paired_hardware_CCD.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    for offset,key,label in [(-.18,'simulation_accuracy','Simulation on these 40'),(.18,'accuracy','Fresh hardware captures')]:
        values=[report['arms'][arm][key]*100 for arm in ['A','B']]
        bars=ax.bar(np.arange(2)+offset,values,.36,label=label)
        ax.bar_label(bars,fmt='%.1f%%',padding=3)
    ax.set_xticks([0,1],['A old resampled mask','B native 8 um mask']);ax.set_ylim(0,110)
    ax.set_ylabel('Accuracy (%)');ax.legend(loc='lower right')
    ax.set_title('Same 40 fixed inputs; not full-test hardware accuracy')
    fig.savefig(captured/'paired_accuracy.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--hardware-run',type=Path);a=p.parse_args()
    if a.hardware_run:hardware(a.run,a.hardware_run)
    else:run(a.run)
