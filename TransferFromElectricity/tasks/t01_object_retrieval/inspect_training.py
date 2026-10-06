"""Auditable epoch timing and physical expert-layer assembly for control_v3."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import torch
from .protocol import common_anchor, physical_phase
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.geometry import MoEGeometry

TASK = Path(__file__).resolve().parent
METHODS = ('direct', 'qwen_lora', 'clip_lora')
NAMES = {'direct':'Direct phase', 'qwen_lora':'Qwen LoRA', 'clip_lora':'CLIP LoRA'}
DATASETS = {'caltech':'Caltech101 (10 classes)', 'cifar':'CIFAR-100 (fixed 10)', 'imagenette':'Imagenette (10 classes)'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value,indent=2),encoding='utf-8',newline='\n')


def csv_write(path, rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def geometry():
    g=MoEGeometry(canvas_size=518,active_size=478,expert_size=224,expert_pitch=254,num_experts=4,grid_rows=2,grid_cols=2)
    g.validate()
    return g


def assemble(phase):
    """Identity modulation outside expert apertures: phase zero, exp(i*0)=1."""
    if tuple(phase.shape)!=(2,4,224,224):raise ValueError('Expected [2,4,224,224] physical phase')
    g=geometry();canvas=phase.new_zeros((2,g.canvas_size,g.canvas_size))
    for index,a in enumerate(g.expert_apertures):canvas[:,a.y0:a.y1,a.x0:a.x1]=phase[:,index]
    return canvas


def phase_layers(output, artifacts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    g=geometry();artifacts.mkdir(parents=True,exist_ok=True);checks=[];initial_checks=[]
    initial=physical_phase(common_anchor(42,dc_power=.25));initial_canvas=assemble(initial)
    support=assemble(torch.ones_like(initial)).bool()
    for dataset in DATASETS:
        fixed_run=TASK/'runs/simulation'/f'20260908_v3_formal_{dataset}_fixed_s42'
        fixed_path=fixed_run/'expert_bank.pt'
        fixed_sha=hashlib.sha256(fixed_path.read_bytes()).hexdigest()
        assert fixed_sha==read(fixed_run/'final_report.json')['expert_bank_sha256']
        fixed=torch.load(fixed_path,map_location='cpu',weights_only=True)
        raw_delta=float((fixed['raw_phase']-common_anchor(42,dc_power=.25)).abs().max())
        phase_delta=float((fixed['phase_rad']-initial).abs().max())
        assert raw_delta<=2e-7 and phase_delta<=1e-6
        initial_checks.append({'run_id':fixed_run.name,'expert_bank_sha256':fixed_sha,
                               'raw_reconstruction_max_error':raw_delta,'physical_reconstruction_max_error_rad':phase_delta})
        banks={'initial':initial};sources={}
        for method in METHODS:
            run=TASK/'runs/simulation'/f'20260908_v3_formal_{dataset}_{method}_s42'
            cfg=read(run/'protocol.json');r=read(run/'final_report.json')
            assert cfg['seed']==42 and cfg['initial_expert_dc_power']==.25
            assert cfg['method']==r['method']==method
            p=run/'expert_bank.pt';digest=hashlib.sha256(p.read_bytes()).hexdigest()
            assert digest==r['expert_bank_sha256']
            payload=torch.load(p,map_location='cpu',weights_only=True)
            assert payload['selected_epoch']==r['selected_epoch'] and payload['git_sha']==r['git_sha']
            banks[method]=payload['phase_rad'];sources[method]={'run_id':run.name,'expert_bank_sha256':digest}
        fig,axes=plt.subplots(2,4,figsize=(15,8.4),layout='constrained')
        for col,(method,bank) in enumerate(banks.items()):
            canvas=assemble(bank)
            for i,a in enumerate(g.expert_apertures):torch.testing.assert_close(canvas[:,a.y0:a.y1,a.x0:a.x1],bank[:,i],rtol=0,atol=0)
            assert float(canvas[~support].abs().max())==0
            path=artifacts/f'{dataset}_{method}_assembled.pt'
            payload={'phase_canvas_rad':canvas,'phase_active_rad':canvas[:,20:498,20:498],
                     'modality_order':['vision','language'],'geometry':g.__dict__,
                     'expert_order':[[0,1],[2,3]],'outside_expert_phase_rad':0.,'source':sources.get(method,'common_anchor(seed=42, dc_power=.25)')}
            if path.exists():
                old=torch.load(path,map_location='cpu',weights_only=True)
                torch.testing.assert_close(old['phase_canvas_rad'],canvas,rtol=0,atol=0)
            else:torch.save(payload,path)
            checks.append({'dataset':dataset,'method':method,'artifact':str(path.relative_to(TASK)),
                           'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'reconstruction_max_error':0.})
            for row in range(2):
                ax=axes[row,col];im=ax.imshow(canvas[row],cmap='twilight',vmin=0,vmax=2*torch.pi,interpolation='nearest')
                for i,a in enumerate(g.expert_apertures):
                    ax.add_patch(Rectangle((a.x0-.5,a.y0-.5),224,224,fill=False,edgecolor='white',lw=.4))
                    ax.text(a.x0+8,a.y0+19,f'E{i}',color='white',fontsize=8,bbox={'facecolor':'black','alpha':.55,'edgecolor':'none'})
                ax.set_xticks([0,259,517]);ax.set_yticks([0,259,517])
                if row==0:ax.set_title('Shared initialization' if method=='initial' else NAMES[method])
                if col==0:ax.set_ylabel(('Vision' if row==0 else 'Language')+' | y pixel')
                if row==1:ax.set_xlabel('x pixel')
        fig.colorbar(im,ax=axes,label='Absolute physical phase (rad)',ticks=[0,torch.pi,2*torch.pi],shrink=.8)
        fig.suptitle(DATASETS[dataset]+' | assembled expert layers\n518 canvas = 20 guard + 478 active + 20 guard; 224 experts, 30 gap; gaps/guard phase = 0')
        fig.savefig(output/f'{dataset}_assembled_expert_layers.png',dpi=145);plt.close(fig)
    write(output/'mask_assembly_manifest.json',{'geometry':g.__dict__,'expert_apertures':[a.__dict__ for a in g.expert_apertures],
        'initial_phase':{'minimum_rad':float(initial.min()),'maximum_rad':float(initial.max()),'mean_rad':float(initial.mean()),'std_rad':float(initial.std()),
                         'raw_min':float(common_anchor(42).min()),'raw_max':float(common_anchor(42).max())},
        'initial_reconstruction_checks':initial_checks,'artifacts':checks})


def historical_timing(output):
    rows=[];summaries=[];targets=[];evidence=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for column,dataset in enumerate(DATASETS):
        for method in METHODS:
            run=TASK/'runs/simulation'/f'20260908_v3_formal_{dataset}_{method}_s42'
            h=read(run/'history.json');steps=read(run/'steps.json');assert len(h)==30
            local=[]
            for i,record in enumerate(h):
                batch=[s for s in steps if s['epoch']==record['epoch']];assert len(batch)==120
                span=batch[-1]['elapsed_seconds']-batch[0]['elapsed_seconds'];assert span>0
                row={'dataset':dataset,'method':method,'run_id':run.name,'epoch':record['epoch'],'stage':record['stage'],
                     'record_elapsed_seconds':record['elapsed_seconds'],
                     'record_interval_seconds':record['elapsed_seconds']-h[i-1]['elapsed_seconds'] if i else None,
                     'measured_steps_2_to_120_seconds':span,'estimated_120_step_seconds':span/119*120,
                     'after_last_step_to_record_seconds':record['elapsed_seconds']-batch[-1]['elapsed_seconds']}
                rows.append(row);local.append(row)
            summaries.append({'dataset':dataset,'method':method,'record_interval_mean_epochs_2_to_30_seconds':statistics.mean(r['record_interval_seconds'] for r in local[1:]),
                              'estimated_120_step_mean_seconds':statistics.mean(r['estimated_120_step_seconds'] for r in local)})
            line=axes[0,column].plot(range(1,31),[r['estimated_120_step_seconds'] for r in local],label=NAMES[method])[0]
            axes[1,column].plot([r['elapsed_seconds']/60 for r in h],[r['live_validation']['top1_retrieval_accuracy']*100 for r in h],color=line.get_color(),label=NAMES[method])
            for threshold in {'caltech':(70,75),'cifar':(40,45,50),'imagenette':(25,30,35)}[dataset]:
                hit=next((r for r in h if r['live_validation']['top1_retrieval_accuracy']*100>=threshold-1e-5),None)
                targets.append({'dataset':dataset,'method':method,'validation_top1_target_percent':threshold,'first_epoch':hit['epoch'] if hit else None,'historical_elapsed_seconds':hit['elapsed_seconds'] if hit else None})
            for file in ('history.json','steps.json','environment.json','protocol.json'):
                digest=hashlib.sha256((run/file).read_bytes()).hexdigest()
                receipt={r['file']:r for r in read(run/'transfer_manifest.json')}
                assert receipt[file]['sha256']==digest
                evidence.append({'run_id':run.name,'file':file,'sha256':digest})
        axes[0,column].set_title(DATASETS[dataset]);axes[0,column].set_xlabel('Epoch');axes[0,column].set_ylabel('120-step estimate (s), first step excluded')
        axes[1,column].set_xlabel('Historical elapsed time (minutes)');axes[1,column].set_ylabel('Validation Top-1 (%)')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=3)
    fig.suptitle('Historical shared-server observations | not an isolated speed benchmark')
    fig.savefig(output/'historical_epoch_time_and_validation.png',dpi=150);plt.close(fig)
    csv_write(output/'historical_epoch_timing.csv',rows);csv_write(output/'historical_timing_summary.csv',summaries);csv_write(output/'historical_time_to_validation_targets.csv',targets)
    write(output/'historical_timing_evidence.json',evidence)
    return summaries


def measured_timing(output):
    rows=[];contracts=set();evidence=[]
    execution=read(output/'timing_execution.json')
    assert hashlib.sha256((output/'timing_execution.json').read_bytes()).hexdigest()==read(output/'timing_execution_transfer_manifest.json')['sha256']
    assert execution['complete'] and len(execution['records'])==3
    assert [r['method'] for r in execution['records']]==list(METHODS)
    for record in execution['records']:
        assert record['status']=='complete' and record['returncode']==0 and record['own_gpu_seen'] and not record['foreign_pids']
        assert record['run_id']==f'20260908_v3_timing_cifar_{record["method"]}_s42'
    for left,right in zip(execution['records'],execution['records'][1:]):
        assert left['finished']<=right['started']
    for method in METHODS:
        run=TASK/'runs/smoke'/f'20260908_v3_timing_cifar_{method}_s42'
        assert read(run/'status.json')['status']=='complete'
        t=read(run/'epoch_timing.json');cfg=read(run/'protocol.json');env=read(run/'environment.json');result=read(run/'final_report.json')
        assert result['evaluation_split']=='validation' and result['selected_live_test'] is None and result['export_max_error']==0
        assert result['method']==cfg['method']==method
        assert t['git_sha']==env['git_sha']==result['git_sha']
        comparable={k:v for k,v in cfg.items() if k!='method'}
        contracts.add((t['git_sha'],env['cuda_visible_devices'],json.dumps(comparable,sort_keys=True),result['split_sha256']))
        assert 'RTX 4090' in env['device'] and len(t['epochs'])==6
        assert execution['gpu_uuid']==env['cuda_visible_devices'] and execution['git_sha']==t['git_sha']
        assert all(h['frozen_parameter_max_change']==0 for h in read(run/'history.json'))
        receipt={r['file']:r for r in read(run/'transfer_manifest.json')}
        for filename in ('epoch_timing.json','protocol.json','environment.json','final_report.json','history.json'):
            digest=hashlib.sha256((run/filename).read_bytes()).hexdigest()
            assert receipt[filename]['sha256']==digest
            evidence.append({'run_id':run.name,'file':filename,'sha256':digest,'source':receipt[filename]['source']})
        for r in t['epochs']:
            assert r['steps']==120
            assert abs(sum(r[k] for k in ('setup_seconds','train_loop_seconds','evaluation_and_audit_seconds','checkpoint_and_logs_seconds'))-r['epoch_wall_seconds'])<1e-6
            rows.append({'method':method,'run_id':run.name,'gpu_uuid':env['cuda_visible_devices'],**r})
    assert len(contracts)==1
    csv_write(output/'measured_epoch_timing.csv',rows)
    selected=[r for r in rows if r['epoch'] in (2,4,6)]
    reference={r['stage']:r['train_loop_seconds'] for r in selected if r['method']=='direct'}
    summaries=[{'method':r['method'],'stage':r['stage'],'epoch':r['epoch'],'train_loop_seconds':r['train_loop_seconds'],
                'epoch_wall_seconds':r['epoch_wall_seconds'],'train_time_ratio_to_direct':r['train_loop_seconds']/reference[r['stage']]} for r in selected]
    csv_write(output/'measured_stage_summary.csv',summaries)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,5.2),layout='constrained')
    components=(('train_loop_seconds','Train loop','#3676b5'),('evaluation_and_audit_seconds','Validation / audit','#edac50'),
                ('checkpoint_and_logs_seconds','Checkpoint / logs','#77b68c'),('setup_seconds','Setup','#b698ca'))
    for ax,stage in zip(axes,('experts','optics','joint')):
        stage_rows=[r for r in selected if r['stage']==stage]
        for i,r in enumerate(stage_rows):
            bottom=0
            for field,label,color in components:
                value=r[field];ax.bar(i,value,bottom=bottom,color=color,label=label if i==0 else None);bottom+=value
            ratio=r['train_loop_seconds']/reference[stage]
            ax.text(i,bottom+1.5,f'{bottom:.1f}s total\n{r["train_loop_seconds"]:.1f}s train\n{ratio:.2f}x train time',ha='center',fontsize=9)
        ax.set_xticks(range(3),['Direct','Qwen LoRA','CLIP LoRA']);ax.set_ylabel('Seconds / 120 batches');ax.set_title(stage+' | epoch '+str(stage_rows[0]['epoch']))
        ax.set_ylim(0,max(r['epoch_wall_seconds'] for r in stage_rows)*1.3)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=4)
    fig.suptitle('Same RTX 4090, sequential runs | second epoch of each stage\nStage-end validation included; training-loop ratio is the primary comparison')
    fig.savefig(output/'measured_epoch_timing.png',dpi=150);plt.close(fig)
    write(output/'measured_timing_evidence.json',evidence)
    write(output/'measured_timing_summary.json',{'git_sha':next(iter(contracts))[0],'epochs':rows,'second_epoch_per_stage':summaries,
          'execution_sha256':hashlib.sha256((output/'timing_execution.json').read_bytes()).hexdigest(),
          'limitations':['One sequential run per method on one GPU; not a distribution of repeat trials','GPU process isolation sampled every five seconds; shared host CPU/storage are not isolated','Six timing epochs use 2/2/2 stages, not the formal 4/8/18 optimization schedule','Second epochs are also stage ends, whose validation includes an additional initial-expert intervention','Same precision as prior implementations: Qwen frozen BF16, CLIP frozen FP32','Training loop includes preprocessing, EMA and gradient/update audits; not GPU kernel time alone']})
    return summaries


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--timing',action='store_true');args=parser.parse_args()
    out=TASK/'reports/timing_and_masks_20260908';out.mkdir(parents=True,exist_ok=True)
    phase_layers(out,TASK/'runs/smoke/20260908_v3_assembled_masks')
    print('Historical timings:',historical_timing(out))
    if args.timing:print('Measured timings:',measured_timing(out))


if __name__=='__main__':main()
