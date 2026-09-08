"""Report all paired spatial_v4 runs; refuse incomplete or incompatible evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import torch

from .inspect_training import assemble
from .run_spatial_suite import METHODS

TASK=Path(__file__).resolve().parent
DATASETS={'caltech':'Caltech101 ten classes','cifar':'CIFAR-100 fixed ten','imagenette':'Imagenette'}
NAMES={'direct':'Direct','clip_vision_lora':'CLIP vision','qwen_vision_pooled_lora':'Qwen pooled',
       'qwen_vision_lora':'Qwen spatial','qwen_vision_global_lora':'Qwen spatial + global'}


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8',newline='\n')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--prefix',default='20260908_v4_formal')
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--smoke',action='store_true',help='Validate real smoke evidence without publishing a formal report')
    args=parser.parse_args()
    if args.smoke and not args.validate_only:raise ValueError('Smoke evidence is only allowed with --validate-only')
    output=TASK/'reports/spatial_v4_20260908';output.mkdir(parents=True,exist_ok=True)
    metrics=[];timings=[];stage_times=[];thresholds=[];diagnostics=[];interventions=[];evidence=[];all_histories={};banks={};contracts=set();references={}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for dataset in DATASETS:
        shared=set();starts=[]
        for method in METHODS:
            run=TASK/'runs'/('smoke' if args.smoke else 'simulation')/f'{args.prefix}_{dataset}_{method}_s42'
            if read(run/'status.json')['status']!='complete':raise ValueError(f'Incomplete run {run.name}')
            cfg=read(run/'protocol.json');r=read(run/'final_report.json');env=read(run/'environment.json')
            hist=read(run/'history.json');timer=read(run/'epoch_timing.json');gpu=read(run/'gpu_execution.json')
            init=read(run/'zero_initialization.json');architecture=read(run/'architecture.json')
            assert cfg['phase_initialization']==r['phase_initialization']=='zero_raw_all_optical'
            assert len(init)==12 and all(x['raw_max_abs']==0 and abs(x['physical_min_rad']-torch.pi)<1e-6 and abs(x['physical_max_rad']-torch.pi)<1e-6 for x in init.values())
            assert len(hist)==len(timer['epochs'])==(3 if args.smoke else 30) and all(x['frozen_parameter_max_change']==0 for x in hist)
            assert r['evaluation_split']==('validation' if args.smoke else 'test') and not r['test_used_for_selection'] and cfg['validation_only']==args.smoke
            assert r['export_max_error']==0 and r['expert_bank_sha256']==digest(run/'expert_bank.pt')
            assert gpu['status']=='complete' and gpu['returncode']==0 and gpu['own_gpu_seen']
            assert 'RTX' in env['device'] and 'A100' not in env['device'] and gpu['gpu_uuid']==env['cuda_visible_devices']
            assert r['git_sha']==env['git_sha']==gpu['git_sha']==timer['git_sha']
            assert all(abs(a-.6)<1e-6 for h in hist for values in h['fusion'].values() for a in values)
            shared.add((r['split_sha256'],env['cuda_visible_devices']))
            starts.append(read(run/'initial_validation.json'))
            contract={k:v for k,v in cfg.items() if k not in {'method','dataset','generator','validation_per_class','development_selection'}}
            contract['generator']={k:v for k,v in cfg['generator'].items() if k!='task_description'}
            contracts.add((r['git_sha'],json.dumps(contract,sort_keys=True)))
            receipt={x['file']:x for x in read(run/'transfer_manifest.json')}
            critical=['final_report.json','history.json','epoch_timing.json','protocol.json','environment.json',
                      'zero_initialization.json','expert_bank.pt','gpu_execution.json','architecture.json',
                      'initial_validation.json','initialization.json','split.json','data_hashes.json','config.yaml']
            if method!='direct':critical+=['fixed_references.json','generator_source_manifest.json']
            for file in critical:
                assert digest(run/file)==receipt[file]['sha256']
                evidence.append({'run_id':run.name,**receipt[file]})
            if method!='direct':
                refs=read(run/'fixed_references.json')['references'];split=read(run/'split.json')
                train_ids={x['sample_id'] for x in split['train']}
                assert len(refs)==10 and all(x['sample_id'] in train_ids for x in refs)
                ref_contract=[(x['sample_id'],x['sha256']) for x in refs]
                if dataset in references:assert references[dataset]==ref_contract
                references[dataset]=ref_contract
                assert not architecture['spatial_generator']['language_tower_executed']
            if method=='qwen_vision_global_lora':
                assert all(h['global_phase']['rms_change_rad']==0 for h in hist if h['stage']=='experts')
            bank=torch.load(run/'expert_bank.pt',weights_only=True,map_location='cpu')
            assert tuple(bank['phase_rad'].shape)==(2,4,224,224) and tuple(bank['global_phase_rad'].shape)==(2,478,478)
            banks[dataset,method]=bank;all_histories[dataset,method]=hist
            selected=r['selected_metrics'];training=[e['train_loop_seconds'] for e in timer['epochs']]
            totals=[e['epoch_wall_seconds'] for e in timer['epochs']]
            selected_history=next(h for h in hist if h['epoch']==r['selected_epoch'])
            metrics.append({'dataset':dataset,'method':method,'run_id':run.name,'selected_epoch':r['selected_epoch'],
                'top1_percent':100*selected['top1_retrieval_accuracy'],'top3_percent':100*selected['top3_retrieval_accuracy'],'mrr':selected['mrr'],
                'train_seconds_mean':statistics.mean(training),'epoch_seconds_mean':statistics.mean(totals),
                'train_seconds_total':sum(training),'epoch_seconds_total':sum(totals),'gpu_uuid':env['cuda_visible_devices'],'gpu_model':env['device'],
                'run_wall_seconds':gpu['finished']-gpu['started'],
                'foreign_gpu_pids':json.dumps(gpu['foreign_pids']),'peak_memory_gib':r['peak_memory_gib'],
                'expert_rms_rad':r['selected_expert_phase']['rms_change_rad'],'global_rms_rad':r['selected_global_phase']['rms_change_rad'],
                'vision_experts_used_selected_epoch':sum(x>0 for x in selected_history['expert_selection_counts']['vision']),
                'language_experts_used_selected_epoch':sum(x>0 for x in selected_history['expert_selection_counts']['language']),
                'generator_trainable_parameters':architecture['generator_trainable']})
            for name,result in r['ablations'].items():
                if 'top1_retrieval_accuracy' not in result:continue
                interventions.append({'dataset':dataset,'method':method,'intervention':name,
                    'top1_percent':100*result['top1_retrieval_accuracy'],'top3_percent':100*result['top3_retrieval_accuracy'],
                    'top1_change_pp':100*(result['top1_retrieval_accuracy']-selected['top1_retrieval_accuracy'])})
            for h in hist:
                chain=h['task_gradient_chain']
                diagnostics.append({'dataset':dataset,'method':method,'epoch':h['epoch'],'stage':h['stage'],
                    'expert_rms_rad':h['expert_phase']['rms_change_rad'],'global_rms_rad':h['global_phase']['rms_change_rad'],
                    'expert_saturated_fraction':h['expert_phase']['sigmoid_saturated_fraction'],
                    'global_saturated_fraction':h['global_phase']['sigmoid_saturated_fraction'],
                    'per_expert_rms_rad':json.dumps(h['expert_phase']['per_expert_rms_change_rad']),
                    'expert_selection_counts':json.dumps(h['expert_selection_counts']),
                    'task_to_expert':chain.get('task_to_expert',''),'task_to_lora_b':chain.get('task_to_lora_b',''),
                    'task_to_global':chain.get('task_to_global','')})
            for epoch in timer['epochs']:
                assert epoch['steps']==(2 if args.smoke else 120)
                assert abs(sum(epoch[k] for k in ('setup_seconds','train_loop_seconds','evaluation_and_audit_seconds','checkpoint_and_logs_seconds'))-epoch['epoch_wall_seconds'])<1e-6
                timings.append({'dataset':dataset,'method':method,'run_id':run.name,'gpu_uuid':env['cuda_visible_devices'],**epoch})
            for stage in ('experts','optics','joint'):
                rows=[x for x in timer['epochs'] if x['stage']==stage]
                stage_times.append({'dataset':dataset,'method':method,'stage':stage,'epochs':len(rows),
                    'train_seconds_mean':statistics.mean(x['train_loop_seconds'] for x in rows),'epoch_seconds_mean':statistics.mean(x['epoch_wall_seconds'] for x in rows)})
            # Fixed, explicit thresholds; misses remain misses, with no test-driven selection.
            for threshold in (50,60,70,80):
                seconds=0.;hit=None
                if starts[-1]['top1_retrieval_accuracy']*100+1e-5>=threshold:hit=(0,0.)
                for h,t in zip(hist,timer['epochs']):
                    assert h['epoch']==t['epoch']
                    seconds+=t['epoch_wall_seconds']
                    if hit is None and h['live_validation']['top1_retrieval_accuracy']*100+1e-5>=threshold:hit=(h['epoch'],seconds)
                thresholds.append({'dataset':dataset,'method':method,'validation_top1_threshold_percent':threshold,
                    'reached':hit is not None,'first_epoch':hit[0] if hit else '',
                    'cumulative_epoch_seconds':hit[1] if hit else ''})
        assert len(shared)==1,shared
        assert all(start==starts[0] for start in starts),'Methods do not share initial validation outputs'
    assert len(contracts)==1,contracts
    if args.validate_only:
        print(f'Validated {len(metrics)} paired runs and {len(timings)} epoch records; no result tables or figures written')
        return
    write_csv(output/'metrics.csv',metrics);write_csv(output/'epoch_times.csv',timings);write_csv(output/'stage_times.csv',stage_times)
    write_csv(output/'validation_time_to_threshold.csv',thresholds)
    write_csv(output/'phase_and_gradient_diagnostics.csv',diagnostics)
    write_csv(output/'component_interventions.csv',interventions)
    write(output/'summary.json',{'source_git_sha':next(iter(contracts))[0],'metrics':metrics,'fixed_reference_contracts':references})
    write(output/'evidence_manifest.json',evidence)
    fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
    for col,dataset in enumerate(DATASETS):
        for method in METHODS:
            hist=all_histories[dataset,method]
            line=axes[0,col].plot([x['epoch'] for x in hist],[x['live_validation']['top1_retrieval_accuracy']*100 for x in hist],label=NAMES[method])[0]
            rows=[x for x in timings if x['dataset']==dataset and x['method']==method]
            axes[1,col].plot([x['epoch'] for x in rows],[x['train_loop_seconds'] for x in rows],color=line.get_color())
        axes[0,col].set_title(DATASETS[dataset]);axes[0,col].set_ylabel('Validation Top-1 (%)')
        axes[1,col].set_ylabel('Train loop seconds / 120 batches')
        for row in range(2):axes[row,col].set_xlabel('Epoch')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=5)
    fig.suptitle('Zero raw phase initialization | spatial vision generators | one optimization seed')
    fig.savefig(output/'validation_and_epoch_time.png',dpi=150);plt.close(fig)
    for dataset in DATASETS:
        fig,axes=plt.subplots(4,6,figsize=(20,14),layout='constrained')
        for col,method in enumerate(['initial',*METHODS]):
            if method=='initial':expert=torch.full((2,4,224,224),torch.pi);global_=torch.full((2,478,478),torch.pi)
            else:expert=banks[dataset,method]['phase_rad'];global_=banks[dataset,method]['global_phase_rad']
            expert_canvas=assemble(expert);global_canvas=torch.zeros(2,518,518);global_canvas[:,20:498,20:498]=global_
            for i,canvas in enumerate((expert_canvas,global_canvas)):
                for modality in range(2):
                    ax=axes[i*2+modality,col];im=ax.imshow(canvas[modality],cmap='twilight',vmin=0,vmax=2*torch.pi,interpolation='nearest')
                    ax.set_xticks([]);ax.set_yticks([])
                    if col==0:ax.set_ylabel(('Vision' if modality==0 else 'Language')+(' expert' if i==0 else ' global'))
                    if i==0 and modality==0:ax.set_title('Initial raw=0' if method=='initial' else NAMES[method],fontsize=10)
        fig.colorbar(im,ax=axes,label='Absolute physical phase (rad)',shrink=.7)
        fig.suptitle(DATASETS[dataset]+' | selected experts and global planes\nExpert gaps and outer guard have phase 0; trainable areas start at pi')
        fig.savefig(output/f'{dataset}_expert_global_layers.png',dpi=120);plt.close(fig)
    lines=['# 零初值空间生成：完整结果','',
           '五组均从相同零 raw 光学相位开始。表中 Top-1/Top-3 是 validation 所选模型的正式 test 结果；时间为全部30轮的算术均值，含各阶段首轮。',
           '训练循环包含数据处理、生成器、光电计算、反传及审计；完整 epoch 还包含阶段设置、验证和保存。模型加载与训练后干预不在 epoch 内。','',
           '| 数据集 | 方法 | 选择轮次 | Top-1 | Top-3 | 训练秒/轮 | 总秒/轮 | 专家 RMS/rad | global RMS/rad |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in metrics:
        lines.append(f'| {DATASETS[row["dataset"]]} | {NAMES[row["method"]]} | {row["selected_epoch"]} | {row["top1_percent"]:.2f}% | {row["top3_percent"]:.2f}% | {row["train_seconds_mean"]:.2f} | {row["epoch_seconds_mean"]:.2f} | {row["expert_rms_rad"]:.4f} | {row["global_rms_rad"]:.4f} |')
    lines+=['','所有逐轮数据见 [epoch_times.csv](epoch_times.csv)，分阶段均值见 [stage_times.csv](stage_times.csv)。',
            '相同数据集的五组使用相同物理RTX顺序训练；不同数据集可使用不同型号，不直接横比跨数据集耗时。CPU/磁盘属于共享资源。只有一个优化seed，不能凭单次排序证明稳定优势。',
            'GPU外部PID情况在metrics.csv和执行审计中保留；若出现外部PID，相关时间需标为受共享GPU负载影响。',
            '固定validation阈值的首次达到时间见 [validation_time_to_threshold.csv](validation_time_to_threshold.csv)，未达到的阈值不填估算值；累计时间不含加载与初始验证，首次达到不等于稳定保持。',
            '本轮参考图来自训练集且固定，新增了静态视觉条件；与旧文本生成或随机相位初值的成绩不构成单变量比较。']
    (output/'完整结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
