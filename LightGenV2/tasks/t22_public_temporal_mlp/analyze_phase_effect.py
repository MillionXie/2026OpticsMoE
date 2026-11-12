"""Measure learned phase-pattern contributions with fixed electronic weights."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__:
    from .data import load_inputs, sha256
    from .model import load_lightgen
    from .phase_regularization import phase_statistics, illumination_support
    from .run import evaluate, write
    from .phase_plotting import phase_display_norm,add_phase_colorbar,PHASE_DISPLAY
else:
    from data import load_inputs, sha256
    from model import load_lightgen
    from phase_regularization import phase_statistics, illumination_support
    from run import evaluate, write
    from phase_plotting import phase_display_norm,add_phase_colorbar,PHASE_DISPLAY


STAGES = ('frame_global','video_global')


def phase_information(stage):
    phase = 2*math.pi*torch.sigmoid(stage.raw_global_phase.detach())
    support = illumination_support(stage.frame_level).to(phase.device)
    active = phase[:,support]
    resultant = torch.complex(active.cos().mean(-1),active.sin().mean(-1))
    centers = resultant.angle()
    spread = (-2*resultant.abs().clamp(1e-12,1).log()).sqrt()
    centered = torch.atan2((phase-centers[:,None,None]).sin(),
                           (phase-centers[:,None,None]).cos())
    statistics = phase_statistics(stage)
    statistics['mean_illuminated_circular_std_rad'] = float(spread.mean())
    return phase,centered,centers,statistics


def constant_modulation(stage,centers,zero=False):
    phase = torch.zeros_like(centers) if zero else centers
    coefficient = stage.cfg.alpha_eval
    values = coefficient*torch.exp(1j*phase)+(1-coefficient)
    return values[:,None,None].expand_as(stage.raw_global_phase)


def plot_phases(records,arrays,path):
    labels = list(records)
    limit = max(float(np.abs(arrays[label][name+'_centered'][0]).max())
                for label in labels for name in STAGES)
    limit = max(limit,1e-6)
    fig,axes=plt.subplots(4,len(labels),figsize=(4.3*len(labels),12),
                          squeeze=False,layout='constrained')
    for column,label in enumerate(labels):
        for stage_index,name in enumerate(STAGES):
            row = 2*stage_index
            phase = arrays[label][name][0]
            centered = arrays[label][name+'_centered'][0]
            spread = records[label]['phase_statistics'][name]['mean_illuminated_circular_std_rad']
            title = 'Vision' if stage_index==0 else 'Vision-language'
            artist=axes[row,column].imshow(phase,cmap='twilight',norm=phase_display_norm())
            axes[row,column].set_title(f'{label}: {title}\nActual phase, slot 0')
            add_phase_colorbar(fig,artist,axes[row,column])
            artist=axes[row+1,column].imshow(centered,cmap='RdBu_r',vmin=-limit,vmax=limit)
            axes[row+1,column].set_title(f'Phase relative to illuminated circular mean\nActive circular SD, mean of 16 slots: {spread:.4f} rad')
            fig.colorbar(artist,ax=axes[row+1,column],shrink=.7,label='Relative phase (rad)')
    fig.suptitle('Absolute and centered phase use common color limits across checkpoints',fontsize=15)
    fig.savefig(path,dpi=170)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints',type=Path,nargs='+',required=True)
    parser.add_argument('--labels',nargs='+',required=True)
    parser.add_argument('--paths',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if len(args.labels)!=len(args.checkpoints) or len(set(args.labels))!=len(args.labels):
        raise ValueError('Provide one unique label per checkpoint')
    torch.set_num_threads(4)
    payload=load_inputs(json.loads(args.paths.read_text('utf-8')))
    args.out.mkdir(parents=True)
    records={};arrays={}
    for label,checkpoint in zip(args.labels,args.checkpoints):
        model,metadata=load_lightgen(checkpoint,args.device)
        model.eval()
        if model.cfg.orm_mode!='amplitude':
            raise ValueError('Amplitude-ORM checkpoint required')
        if metadata['dataset']['manifest']['sha256']!=payload['identity']['manifest']['sha256']:
            raise ValueError('Dataset identity mismatch')
        statistics={};centers={};arrays[label]={}
        for name in STAGES:
            phase,centered,centers[name],statistics[name]=phase_information(getattr(model,name))
            arrays[label][name]=phase.cpu().numpy()
            arrays[label][name+'_centered']=centered.cpu().numpy()
        record={'checkpoint_SHA256':sha256(checkpoint),'phase_statistics':statistics,'conditions':{}}
        cases={'learned':(), 'flat_vision':('frame_global',),
               'flat_joint':('video_global',),'flat_both':STAGES,
               'zero_vision':('frame_global',),'zero_joint':('video_global',),'zero_both':STAGES}
        for case,names in cases.items():
            with ExitStack() as stack:
                for name in names:
                    value=constant_modulation(getattr(model,name),centers[name],zero=case.startswith('zero_'))
                    stack.enter_context(patch.object(getattr(model,name),'modulation',lambda value=value:value))
                dest=args.out/label/case
                dest.mkdir(parents=True)
                record['conditions'][case]=evaluate(model,payload,args.device,4,dest)
        records[label]=record
        write(args.out/'phase_effect.json',records)
        print(json.dumps({'completed':label,'srcc':{case:metric['srcc'] for case,metric in record['conditions'].items()}}),flush=True)
    plot_phases(records,arrays,args.out/'phase_texture.png')
    np.savez_compressed(args.out/'phase_arrays.npz',
                        **{label+'_'+name:value for label,items in arrays.items() for name,value in items.items()})
    write(args.out/'protocol.json',{'data_identity':payload['identity'],
          'role':'post_training_phase_pattern_ablation_on_development_data',
          'electronic_weights_fixed':True,'constant_phase':'circular mean within each geometrically illuminated tile',
          'zero_phase':'phase zero in both stages, modulation recomputed by the existing model rule',
          'phase_display':PHASE_DISPLAY,
          'hardware_measurement':False,'script_SHA256':sha256(Path(__file__)),
          'files':{p.name:sha256(p) for p in args.out.iterdir() if p.is_file()}})


if __name__=='__main__':main()
