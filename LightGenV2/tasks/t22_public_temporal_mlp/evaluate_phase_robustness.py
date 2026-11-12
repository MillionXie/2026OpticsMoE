"""Compare clean accuracy, periodic mask roughness and controlled perturbations."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
import json
import math
from pathlib import Path
import platform
import subprocess
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__:
    from .data import load_inputs, sha256
    from .model import load_lightgen, translated, Propagation
    from .phase_regularization import phase_statistics
    from .run import evaluate
else:
    from data import load_inputs, sha256
    from model import load_lightgen, translated, Propagation
    from phase_regularization import phase_statistics
    from run import evaluate


def conditions():
    result=[{'kind':'clean','level':0.,'trial':0}]
    for kind in ('phase_shift','ccd_shift'):
        for level in (1,2):
            for trial,(dy,dx) in enumerate([(level,0),(-level,0),(0,level),(0,-level)]):
                result.append({'kind':kind,'level':level,'trial':trial,'dy':dy,'dx':dx})
    for level in (.02,.05):
        for trial in range(3):
            result.append({'kind':'phase_noise','level':level,'trial':trial})
    result.append({'kind':'quantization','level':8,'trial':0})
    for factor in (.99,1.01):
        result.append({'kind':'distance','level':factor,'trial':0})
    return result


def modulation_for_condition(stage, condition, stage_index):
    original = stage.modulation
    if condition['kind']=='phase_shift':
        def modulation():
            return translated(original(),condition['dy'],condition['dx'],fill=1+0j)
        return modulation
    phase=2*math.pi*torch.sigmoid(stage.raw_global_phase.detach())
    if condition['kind']=='phase_noise':
        generator=torch.Generator(device=phase.device).manual_seed(3701+100*stage_index+condition['trial'])
        phase=phase+condition['level']*torch.randn(phase.shape,device=phase.device,generator=generator)
    elif condition['kind']=='quantization':
        maximum=2**int(condition['level'])-1
        phase=(phase/(2*math.pi)*maximum).round()/maximum*(2*math.pi)
    else:
        raise ValueError('Unsupported modulation perturbation')
    # The configured amplitude mixing remains private and is identical in all comparisons.
    coefficient=stage.cfg.alpha_eval
    mask=coefficient*torch.exp(1j*phase)+(1-coefficient)
    return lambda:mask


def evaluate_condition(model,payload,device,condition):
    with ExitStack() as stack:
        for index,stage in enumerate((model.frame_global,model.video_global)):
            if condition['kind'] in ('phase_shift','phase_noise','quantization'):
                stack.enter_context(patch.object(stage,'modulation',
                    modulation_for_condition(stage,condition,index)))
            elif condition['kind']=='ccd_shift':
                def shift(module,inputs,output):
                    return translated(output,condition['dy'],condition['dx'])
                handle=stage.propagation.register_forward_hook(shift)
                stack.callback(handle.remove)
            elif condition['kind']=='distance':
                altered=Propagation(replace(stage.cfg,distance_m=stage.cfg.distance_m*condition['level']))
                transfer=altered.transfer.to(stage.propagation.transfer.device)
                original=stage.propagation.transfer
                stage.propagation.transfer=transfer
                stack.callback(setattr,stage.propagation,'transfer',original)
        return evaluate(model,payload,device,4)


def plot_curves(records,out):
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    for label,record in records.items():
        clean=record['conditions'][0]['metrics']['srcc']
        for ax,kind,title,levels in zip(axes,('phase_shift','ccd_shift','phase_noise'),
                ('Phase-mask shift','CCD registration shift','Static independent phase noise'),
                ([1,2],[1,2],[.02,.05])):
            means=[clean];lower=[clean];upper=[clean]
            for level in levels:
                values=[row['metrics']['srcc'] for row in record['conditions']
                        if row['condition']['kind']==kind and row['condition']['level']==level]
                means.append(float(np.mean(values)));lower.append(min(values));upper.append(max(values))
            x=[0]+levels
            ax.plot(x,means,marker='o',label=label)
            ax.fill_between(x,lower,upper,alpha=.12)
            ax.set(title=title,ylabel='Development SRCC',xlabel='Simulation pixels' if kind!='phase_noise' else 'Standard deviation (rad)')
    axes[-1].legend(frameon=False)
    fig.savefig(out,dpi=200)
    plt.close(fig)


def plot_masks(records,phases,out):
    fig,axes=plt.subplots(2,len(records),figsize=(4*len(records),7),squeeze=False,layout='constrained')
    for column,(label,value) in enumerate(phases.items()):
        for row,name in enumerate(('frame_global','video_global')):
            image=axes[row,column].imshow(value[name],cmap='twilight',vmin=0,vmax=2*np.pi)
            rough=records[label]['phase_statistics'][name]['curvature_penalty']
            axes[row,column].set_title(f'{label}\n{name.replace("_"," ")}, curvature {rough:.4f}')
            fig.colorbar(image,ax=axes[row,column],shrink=.7,label='Phase (rad)')
    fig.savefig(out,dpi=180)
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
        raise ValueError('A unique label is required for each checkpoint')
    torch.set_num_threads(4)
    payload=load_inputs(json.loads(args.paths.read_text('utf-8')))
    args.out.mkdir(parents=True)
    records={};phases={}
    for label,path in zip(args.labels,args.checkpoints):
        model,metadata=load_lightgen(path,args.device)
        if model.cfg.orm_mode!='amplitude':
            raise ValueError('Current amplitude-ORM model required')
        if payload['identity']['manifest']['sha256']!=metadata['dataset']['manifest']['sha256']:
            raise ValueError('Checkpoint dataset identity differs')
        model.eval()
        record={'checkpoint':str(path),'sha256':sha256(path),'conditions':[],
                'phase_statistics':{name:phase_statistics(getattr(model,name)) for name in ('frame_global','video_global')}}
        phases[label]={name:(2*math.pi*torch.sigmoid(getattr(model,name).raw_global_phase.detach()))[0].cpu().numpy()
                       for name in ('frame_global','video_global')}
        for condition in conditions():
            metrics=evaluate_condition(model,payload,args.device,condition)
            record['conditions'].append({'condition':condition,'metrics':metrics})
        if record['sha256']!=sha256(path):
            raise ValueError('Source checkpoint file changed during evaluation')
        records[label]=record
        (args.out/'comparison.json').write_text(json.dumps(records,indent=2)+'\n','utf-8')
        print(json.dumps({'completed':label,'clean_srcc':record['conditions'][0]['metrics']['srcc']}),flush=True)
        del model
    plot_curves(records,args.out/'robustness_curves.png')
    plot_masks(records,phases,args.out/'phase_comparison.png')
    np.savez_compressed(args.out/'slot0_phase_arrays.npz',
                        **{label+'_'+name:array for label,values in phases.items() for name,array in values.items()})
    root=Path(__file__).resolve().parent
    (args.out/'protocol.json').write_text(json.dumps({
        'data_identity':payload['identity'],'phase_shift_directions':['+y','-y','+x','-x'],
        'noise_trials':3,'noise_is_static_across_dataset':True,'same_noise_seeds_for_all_checkpoints':True,
        'conditions_are_numerical_assumptions':True,'hardware_measurement':False,
        'scope':'phase-mask/CCD shifts, independent phase noise, 8-bit numerical phase quantization, distance +/-1%',
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'source_SHA256':{name:sha256(root/name) for name in ('model.py','evaluate_phase_robustness.py','phase_regularization.py')},
        'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'device':args.device}},indent=2)+'\n')


if __name__=='__main__':
    main()
