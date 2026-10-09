"""Small validation-only CCD diagnostic, using saved selected checkpoints on CPU."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import load_domain,sha256
from .model_four import FourRouterClassification,CCD_EDGES
from .train import encode

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-prefix',required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(4)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    task=Path(__file__).resolve().parent
    first=task/'runs'/'simulation'/f'{args.run_prefix}_optical'
    cfg=json.loads((first/'config.json').read_text())
    data,_=load_domain(Path(cfg['data']),Path(cfg['manifest']))
    indices=[int(np.flatnonzero(data['val_labels']==k)[0]) for k in (0,8)]
    x=encode(data['val_images'][indices],'four_top2_ccd')
    fig,axes=plt.subplots(2,4,figsize=(16,8))
    metadata=dict(split='validation',indices=indices,labels=data['val_labels'][indices].tolist(),
                  scale='per-panel log10 intensity relative to panel maximum',models={})
    for r,index in enumerate(indices):
        axes[r,0].imshow(x[r].numpy(),cmap='gray')
        axes[r,0].set_title(f'Fixed RGB input; val {index}; true {metadata["labels"][r]}')
    for c,variant in enumerate(('optical','electronic','d2nn'),1):
        run=task/'runs'/'simulation'/f'{args.run_prefix}_{variant}'
        config=json.loads((run/'config.json').read_text())
        if config['profile']!='four_top2_ccd':raise ValueError(config['profile'])
        if config['data_sha256']!=cfg['data_sha256']:raise ValueError('dataset mismatch')
        result=json.loads((run/'result.json').read_text())
        checkpoint=run/'best_checkpoint.pt'
        if sha256(checkpoint)!=result['checkpoint_sha256']:raise ValueError('checkpoint mismatch')
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=FourRouterClassification(variant,'ccd_grid')
        model.load_state_dict(ck['model']);model.eval()
        with torch.no_grad():out=model(x,return_debug=True)
        power=out['class_energies']/out['class_energies'].sum(1,keepdim=True)
        metadata['models'][variant]=dict(checkpoint_sha256=result['checkpoint_sha256'],
            selected_epoch=ck['epoch'],predictions=out['logits'].argmax(1).tolist(),
            region_power=power.tolist(),route_power=out['route_power'].tolist()
                if out['route_power'] is not None else None)
        for r in range(2):
            ax=axes[r,c]
            field=out['post_oeo_intensity'][r,20:498,20:498].numpy()
            relative=field/field.max()
            im=ax.imshow(np.maximum(relative,1e-5),cmap='inferno',norm=LogNorm(1e-5,1))
            for k in range(9):
                y,z=divmod(k,3);lo_y,hi_y=CCD_EDGES[y:y+2];lo_x,hi_x=CCD_EDGES[z:z+2]
                ax.add_patch(Rectangle((lo_x-.5,lo_y-.5),hi_x-lo_x,hi_y-lo_y,
                                      fill=False,edgecolor='cyan',linewidth=.8))
                ax.text((lo_x+hi_x)/2,(lo_y+hi_y)/2,f'{k}: {float(power[r,k])*100:.1f}%',
                        ha='center',color='white',fontsize=9,
                        bbox=dict(facecolor='black',alpha=.5,pad=1))
            ax.set_title(f'{variant}; selected epoch {ck["epoch"]}; predict '
                         f'{metadata["models"][variant]["predictions"][r]}')
            fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
    for ax in axes.flat:ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Nine fixed CCD regions after second OEO; two validation samples; '
                 'log display only, classification uses linear energy sums',fontsize=12)
    fig.tight_layout()
    args.out.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out/'detector_fields.png',dpi=160);plt.close(fig)
    (args.out/'detector_fields.json').write_text(json.dumps(metadata,indent=2)+'\n')

if __name__=='__main__':main()
