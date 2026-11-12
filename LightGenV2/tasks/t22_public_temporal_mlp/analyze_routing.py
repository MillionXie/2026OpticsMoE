"""Count real Top2 decisions and visualize the selected electronic experts."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import torch

if __package__:
    from .plot_style import save_figure
    from .phase_plotting import phase_display_norm, add_phase_colorbar
    from .data import batch, groups, load_inputs, sha256
    from .model import load_lightgen
    from .phase_regularization import phase_statistics, illumination_support
else:
    from plot_style import save_figure
    from phase_plotting import phase_display_norm, add_phase_colorbar
    from data import batch, groups, load_inputs, sha256
    from model import load_lightgen
    from phase_regularization import phase_statistics, illumination_support


ROOT = Path(__file__).resolve().parent
# Approximate raster fill colors sampled from the submitted Fig.2d.
COLORS = ['#86A2E1','#F5BA92','#E2E2E2','#FFEDB8']
EXPERTS = ['E1','E2','E3','E4']


def write_csv(path, rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_shares(shares, out, ylabel='Optical expert share (%)'):
    fig,ax = plt.subplots(figsize=(3.35,3.2),layout='constrained')
    bottom = np.zeros(2)
    for expert,color in enumerate(COLORS):
        height = np.array([shares[name][expert]*100 for name in ('frame','video')])
        ax.bar([0,1],height,bottom=bottom,width=.64,color=color,label=EXPERTS[expert])
        for x,y,h in zip([0,1],bottom,height):
            ax.text(x,y+h/2,f'{h:.2f}',ha='center',va='center',fontsize=11)
        bottom += height
    ax.set(ylim=(0,100),xticks=[0,1],xticklabels=['Vision\nstage','Vision-language\nstage'],
           ylabel=ylabel)
    ax.legend(frameon=False,bbox_to_anchor=(1.02,1),loc='upper left')
    ax.spines[['top','right']].set_visible(False)
    save_figure(fig,out)
    plt.close(fig)


def plot_slot_shares(slot_shares, out):
    fig,axes = plt.subplots(1,2,figsize=(6.0,4.4),layout='constrained')
    for ax,name in zip(axes,('frame','video')):
        values = slot_shares[name]*100
        artist=ax.imshow(values,cmap='Blues',vmin=0,vmax=50,aspect='auto')
        for row in range(16):
            for column in range(4):
                ax.text(column,row,f'{values[row,column]:.1f}',ha='center',va='center',
                        color='white' if values[row,column]>32 else '#182737',fontsize=8)
        ax.set(title='Vision router' if name=='frame' else 'Vision-language router',xticks=range(4),xticklabels=EXPERTS,
               yticks=range(16),ylabel='Physical video slot')
        fig.colorbar(artist,ax=ax,shrink=.6,aspect=14,label='Optical expert share (%)')
    save_figure(fig,out)
    plt.close(fig)


def plot_expert_operation(model, tokens, weights, out):
    matrices = np.stack([expert.weight.detach().cpu().numpy()
                         for expert in model.frame_experts.experts])
    outputs = np.einsum('ti,koi->kto',tokens.numpy(),matrices)
    limit = max(float(np.percentile(np.abs(matrices),99.5)),1e-6)
    output_limit = max(float(np.percentile(np.abs(outputs),99.5)),1e-6)
    energies = np.sqrt(np.mean((outputs*weights[:,None,None])**2,axis=-1)).reshape(4,7,7)
    energy_limit = max(float(energies.max()),1e-6)
    fig,axes = plt.subplots(3,4,figsize=(7.6,5.6),layout='constrained')
    for i,color in enumerate(COLORS):
        axes[0,i].imshow(matrices[i],cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
        axes[0,i].set(title=f'{EXPERTS[i]} matrix, 192 × 192',xlabel='Input channel',ylabel='Output channel')
        axes[1,i].imshow(outputs[i],cmap='RdBu_r',vmin=-output_limit,vmax=output_limit,aspect='auto')
        axes[1,i].set(title='Expert output, 49 × 192',xlabel='Channel',ylabel='Token')
        cmap=LinearSegmentedColormap.from_list(EXPERTS[i],['white',color,'#333333'])
        axes[2,i].imshow(energies[i],cmap=cmap,vmin=0,vmax=energy_limit)
        state='Selected' if weights[i]>0 else 'Inactive'
        axes[2,i].set(title=f'{state}: weighted RMS',xlabel='Token column',ylabel='Token row')
        for spine in axes[2,i].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3 if weights[i]>0 else 1)
    fig.suptitle('Vision experts in electronic feature space')
    save_figure(fig,out)
    plt.close(fig)


def plot_virtual_activation(tokens, weights, out):
    energy=tokens.square().mean(-1).sqrt().numpy().reshape(7,7)
    fig,axes=plt.subplots(2,2,figsize=(4.2,4.0),layout='constrained')
    high=max(float(energy.max()*weights.max()),1e-6)
    for i,ax in enumerate(axes.flat):
        cmap=LinearSegmentedColormap.from_list(EXPERTS[i],['white',COLORS[i],'#333333'])
        ax.imshow(energy*weights[i],cmap=cmap,vmin=0,vmax=high)
        ax.set_title(f'{EXPERTS[i]}  '+('Selected' if weights[i]>0 else 'Inactive'))
        ax.set(xticks=[],yticks=[])
    fig.suptitle('Virtual expert layout, weighted input-token RMS')
    save_figure(fig,out)
    plt.close(fig)


def plot_router_sensitivity(model,tokens,out):
    value=tokens.detach().clone().requires_grad_(True)
    routing=model.frame_router(value[None])
    maps=[]
    for expert in range(4):
        gradient=torch.autograd.grad(routing['probabilities'][0,expert],value,retain_graph=True)[0]
        maps.append(gradient.square().mean(-1).sqrt().detach().cpu().numpy().reshape(7,7))
    high=max(float(np.max(maps)),1e-12)
    fig,axes=plt.subplots(1,4,figsize=(7.2,2.3),layout='constrained')
    for expert,ax in enumerate(axes):
        image=ax.imshow(maps[expert],cmap='viridis',vmin=0,vmax=high)
        probability=float(routing['probabilities'][0,expert].detach())
        ax.set_title(f'{EXPERTS[expert]}  p={probability:.3f}')
        ax.set(xlabel='Token column',ylabel='Token row')
        fig.colorbar(image,ax=ax,shrink=.6,aspect=14)
    fig.suptitle('Vision router sensitivity, gradient RMS of softmax probability')
    save_figure(fig,out)
    plt.close(fig)


def plot_layout(out):
    fig,axes=plt.subplots(1,2,figsize=(7.2,4.0),layout='constrained')
    for frame_level,ax in zip((True,False),axes):
        ax.add_patch(Rectangle((0,0),518,518,facecolor='#f3f3f3',edgecolor='none'))
        ax.add_patch(Rectangle((20,20),478,478,facecolor='white',edgecolor='#222222',lw=1.2))
        for slot in range(16):
            row,column=divmod(slot,4);y=23+119*row;x=23+119*column
            ax.add_patch(Rectangle((x,y),115,115,fill=False,edgecolor='#9c9c9c',lw=.8))
            ax.text(x+3,y+9,f'V{slot:02d}',fontsize=8)
            if frame_level:
                for frame,(dy,dx) in enumerate([(0,0),(0,59),(59,0),(59,59)]):
                    ax.add_patch(Rectangle((x+dx+14,y+dy+14),27,27,facecolor='#75b2ce',edgecolor='none'))
                    ax.text(x+dx+27.5,y+dy+27.5,f'F{frame+1}',ha='center',va='center',fontsize=8)
            else:
                ax.add_patch(Rectangle((x+29,y+29),56,56,facecolor='#e2e2e2',edgecolor='none'))
                ax.add_patch(Rectangle((x+29,y+29),56,42,facecolor='#75b2ce',edgecolor='none'))
        ax.set(xlim=(0,518),ylim=(518,0),aspect='equal',xlabel='x (simulation pixel)',ylabel='y (simulation pixel)')
        ax.set_title('Vision global\n16 videos × 4 frames, 8 × 8 patches' if frame_level
                     else 'Vision-language global\n16 video fields, 4 × 4 patches')
    fig.suptitle('518 × 518 canvas, 478 × 478 active region')
    save_figure(fig,out)
    plt.close(fig)


def plot_phase_support(model, out):
    fig,axes=plt.subplots(2,3,figsize=(7.2,4.6),layout='constrained')
    for row,name in enumerate(('frame_global','video_global')):
        stage=getattr(model,name)
        phase=stage.phase(wrapped=True).detach()[0].cpu().numpy()
        support=illumination_support(stage.frame_level).numpy()
        visible=np.where(support,phase,np.nan)
        phase_curvature=np.angle(np.exp(1j*(phase[:,2:]-2*phase[:,1:-1]+phase[:,:-2])))
        for column,array in enumerate((phase,visible,phase_curvature)):
            stage_title='Vision' if row==0 else 'Vision-language'
            axes[row,column].set_title(stage_title+'\n'+['Full phase tile','Illumination support','Horizontal curvature'][column])
            axes[row,column].set(xticks=[],yticks=[])
            if column<2:
                image=axes[row,column].imshow(array,cmap='twilight',norm=phase_display_norm())
                add_phase_colorbar(fig,image,axes[row,column])
            else:
                image=axes[row,column].imshow(array,cmap='RdBu_r',vmin=-np.pi,vmax=np.pi)
                bar=fig.colorbar(image,ax=axes[row,column],shrink=.6,aspect=14,
                                 label='Curvature (rad)',ticks=[-np.pi,0,np.pi])
                bar.ax.set_yticklabels(['−π','0','π'])
    save_figure(fig,out)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--paths',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(2)
    model,metadata=load_lightgen(args.weights,args.device)
    payload=load_inputs(json.loads(args.paths.read_text('utf-8')))
    if payload['identity']['manifest']['sha256']!=metadata['dataset']['manifest']['sha256']:
        raise ValueError('Dataset manifest identity differs')
    indices,valid=groups(payload,'test')
    counts={name:np.zeros(4,dtype=np.int64) for name in ('frame','video')}
    per_slot={name:np.zeros((16,4),dtype=np.int64) for name in counts}
    weight_mass={name:np.zeros(4,dtype=np.float64) for name in counts}
    rows=[];sample={}

    def capture(module,inputs):
        if not sample:
            sample['tokens']=inputs[0][0,0,0].detach().cpu().clone()

    handle=model.frame_experts.register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            for group in range(len(indices)):
                item=batch(payload,indices[group:group+1],valid[group:group+1],args.device)
                result=model(item['vision'])
                if group==0:
                    sample['weights']=result['routing']['frame']['weights'][0,0,0].cpu().numpy()
                for slot,index in enumerate(indices[group].tolist()):
                    if not bool(valid[group,slot]):
                        continue
                    for name in counts:
                        route=result['routing'][name]
                        selected=route['selected'][0,slot].cpu().numpy().reshape(-1,4)
                        weights=route['weights'][0,slot].cpu().numpy().reshape(-1,4)
                        if not np.all(selected.sum(-1)==2):
                            raise ValueError('Expected exactly two selected experts')
                        counts[name]+=selected.sum(0)
                        per_slot[name][slot]+=selected.sum(0)
                        weight_mass[name]+=weights.sum(0)
                        for frame,(chosen,weight) in enumerate(zip(selected,weights)):
                            row={'stage':name,'sample_id':payload['ids'][index],
                                 'physical_slot':slot,'frame':frame if name=='frame' else ''}
                            row.update({f'E{i+1}_selected':int(chosen[i]) for i in range(4)})
                            row.update({f'E{i+1}_weight':float(weight[i]) for i in range(4)})
                            rows.append(row)
    finally:
        handle.remove()
    shares={name:value/value.sum() for name,value in counts.items()}
    slot_shares={name:value/value.sum(-1,keepdims=True) for name,value in per_slot.items()}
    args.out.mkdir(parents=True)
    write_csv(args.out/'routing_decisions.csv',rows)
    summary=[];slot_rows=[];stats={}
    for name in counts:
        n=int(counts[name].sum()//2)
        stats[name]={'decisions':n,'total_top2_selections':2*n,'counts':counts[name].tolist(),
                     'share_percent':(shares[name]*100).tolist(),
                     'routing_weight_mass_percent':(weight_mass[name]/n*100).tolist(),
                     'share_entropy_nats':float(-(shares[name]*np.log(shares[name])).sum()),
                     'uniform_share_percent':25.}
        for expert in range(4):
            summary.append({'stage':name,'expert':EXPERTS[expert],'selected_count':int(counts[name][expert]),
                            'total_selected_count':2*n,'Optical expert share (%)':float(shares[name][expert]*100)})
            for slot in range(16):
                slot_rows.append({'stage':name,'physical_slot':slot,'expert':EXPERTS[expert],
                                  'count':int(per_slot[name][slot,expert]),'share_percent':float(slot_shares[name][slot,expert]*100)})
    if stats['frame']['decisions']!=558*4 or stats['video']['decisions']!=558:
        raise ValueError('Incorrect decision population')
    write_csv(args.out/'expert_share.csv',summary)
    write_csv(args.out/'expert_share_by_slot.csv',slot_rows)
    plot_shares(shares,args.out/'optical_expert_share.png')
    plot_shares({name:weight_mass[name]/(counts[name].sum()/2) for name in counts},
                args.out/'mean_routing_weight.png','Mean routing weight (%)')
    plot_slot_shares(slot_shares,args.out/'expert_share_by_slot.png')
    plot_expert_operation(model,sample['tokens'],sample['weights'],args.out/'expert_matrix_and_tokens.png')
    plot_virtual_activation(sample['tokens'],sample['weights'],args.out/'virtual_expert_activation.png')
    plot_router_sensitivity(model,sample['tokens'].to(args.device),args.out/'router_sensitivity.png')
    plot_layout(args.out/'layout_4x4_and_8x8.png')
    plot_phase_support(model,args.out/'phase_illumination_support.png')
    stats['definition']='share = selected count / all Top2 selection counts; each decision contributes two selections'
    stats['weights_sha256']=sha256(args.weights)
    stats['phase_statistics']={name:phase_statistics(getattr(model,name)) for name in ('frame_global','video_global')}
    stats['figure2d_palette']=dict(zip(EXPERTS,COLORS))
    stats['data_identity']=payload['identity']
    stats['example']={'sample_id':payload['ids'][int(indices[0,0])],'physical_slot':0,'frame':0}
    stats['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    stats['source_SHA256']={name:sha256(ROOT/name) for name in ('model.py','analyze_routing.py','phase_regularization.py')}
    stats['output_SHA256']={p.name:sha256(p) for p in args.out.iterdir() if p.is_file()}
    (args.out/'routing_analysis.json').write_text(json.dumps(stats,indent=2)+'\n','utf-8')
    print(json.dumps({name:stats[name] for name in ('frame','video')}),flush=True)


if __name__=='__main__':
    main()
