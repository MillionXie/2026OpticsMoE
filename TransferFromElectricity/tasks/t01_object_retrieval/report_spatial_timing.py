"""Summarize six-epoch validation-only timing runs separately from formal accuracy."""
import argparse
import json

from .report_spatial import TASK,NAMES,read,digest,write,write_csv
from .run_spatial_suite import METHODS


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--prefix',default='20260908_v4_timing');args=parser.parse_args()
    output=TASK/'reports/spatial_v4_20260908'
    rows=[];epochs=[];evidence=[];contracts=set();initial=[];formal_initial=[]
    for method in METHODS:
        run=TASK/'runs/smoke'/f'{args.prefix}_cifar_{method}_s42'
        r=read(run/'final_report.json');cfg=read(run/'protocol.json');env=read(run/'environment.json')
        gpu=read(run/'gpu_execution.json');timer=read(run/'epoch_timing.json');history=read(run/'history.json')
        assert read(run/'status.json')['status']=='complete' and gpu['status']=='complete' and gpu['returncode']==0 and gpu['own_gpu_seen']
        assert cfg['validation_only'] and not cfg['smoke'] and r['evaluation_split']=='validation' and not r['test_used_for_selection']
        assert cfg['phase_initialization']=='zero_raw_all_optical' and r['export_max_error']==0
        zero=read(run/'zero_initialization.json');assert len(zero)==12 and all(v['raw_max_abs']==0 for v in zero.values())
        assert len(history)==len(timer['epochs'])==6 and all(h['frozen_parameter_max_change']==0 for h in history)
        assert [t['stage'] for t in timer['epochs']]==['experts','experts','optics','optics','joint','joint']
        assert [t['epoch'] for t in timer['epochs']]==list(range(1,7)) and all(t['steps']==120 for t in timer['epochs'])
        assert env['git_status'].strip() in {'','?? data'} and 'RTX' in env['device'] and 'A100' not in env['device']
        assert r['git_sha']==gpu['git_sha']==env['git_sha']==timer['git_sha']
        assert gpu['gpu_uuid']==env['cuda_visible_devices']
        formal=TASK/'runs/simulation'/f'20260908_v4_formal_cifar_{method}_s42'
        formal_cfg=read(formal/'protocol.json')
        formal_device=read(formal/'environment.json')['device']
        assert {k:v for k,v in cfg.items() if k not in {'stages','validation_only'}}=={k:v for k,v in formal_cfg.items() if k not in {'stages','validation_only'}}
        assert r['split_sha256']==read(formal/'final_report.json')['split_sha256']
        assert digest(run/'data_hashes.json')==digest(formal/'data_hashes.json')
        assert read(run/'initialization.json')['sha256']==read(formal/'initialization.json')['sha256']
        assert read(run/'architecture.json')==read(formal/'architecture.json')
        if method!='direct':
            assert read(run/'generator_source_manifest.json')['files']==read(formal/'generator_source_manifest.json')['files']
            assert read(run/'fixed_references.json')==read(formal/'fixed_references.json')
        contracts.add((r['git_sha'],r['split_sha256'],digest(run/'data_hashes.json'),gpu['gpu_uuid'],json.dumps({k:v for k,v in cfg.items() if k!='method'},sort_keys=True)))
        initial.append(read(run/'initial_validation.json'))
        formal_initial.append(read(formal/'initial_validation.json'))
        receipt={x['file']:x for x in read(run/'transfer_manifest.json')}
        checkpoints=read(run/'checkpoint_hashes.json')
        assert {x['file'] for x in checkpoints}=={'best_checkpoint.pt','last_checkpoint.pt'}
        assert all(len(x['sha256'])==64 and x['bytes']>0 and not x['downloaded'] for x in checkpoints)
        evidence.extend({'run_id':run.name,'evidence_type':'remote_checkpoint_sha256',**x} for x in checkpoints)
        for name in ('protocol.json','history.json','epoch_timing.json','final_report.json','environment.json','gpu_execution.json','zero_initialization.json','initial_validation.json'):
            assert digest(run/name)==receipt[name]['sha256'];evidence.append({'run_id':run.name,**receipt[name]})
        for t in timer['epochs']:
            epochs.append({'method':method,'run_id':run.name,'gpu_uuid':gpu['gpu_uuid'],'foreign_gpu_pids':json.dumps(gpu['foreign_pids']),**t})
        for index in (1,3,5):
            t=timer['epochs'][index]
            rows.append({'method':method,'stage':t['stage'],'epoch':t['epoch'],'train_loop_seconds':t['train_loop_seconds'],
                'epoch_wall_seconds':t['epoch_wall_seconds'],'gpu_model':env['device'],'gpu_uuid':gpu['gpu_uuid'],
                'foreign_gpu_pids':json.dumps(gpu['foreign_pids'])})
    assert len(contracts)==1 and all(x==initial[0] for x in initial) and all(x==formal_initial[0] for x in formal_initial)
    for row in rows:
        base=next(x for x in rows if x['method']=='direct' and x['stage']==row['stage'])
        row['train_ratio_to_direct']=row['train_loop_seconds']/base['train_loop_seconds']
    write_csv(output/'timing_recheck_stages.csv',rows);write_csv(output/'timing_recheck_epochs.csv',epochs)
    write(output/'timing_recheck_evidence.json',evidence)
    initial_comparison={kind:{key:100*value[key] for key in ('top1_retrieval_accuracy','top3_retrieval_accuracy')}
        for kind,value in [('timing_validation',initial[0]),('formal_validation',formal_initial[0])]}
    write(output/'timing_recheck_summary.json',{'git_sha':next(iter(contracts))[0],'policy':'Validation only, six epochs, report the second epoch of each stage; retain all 30 epoch records',
        'initial_validation_percent':initial_comparison,'rows':rows})
    lines=['# 同卡短计时复测','',
           'CIFAR固定十类、零raw相位，每组experts/optics/joint各两轮、每轮120个batch，沿用正式训练的模型、精度与学习率；仅使用validation。以下采用每阶段第二轮（2/4/6），避开阶段解锁首轮；不是正式test准确度实验。','',
           '| 方法 | experts训练秒 | optics训练秒 | joint训练秒 | joint/direct |',
           '|---|---:|---:|---:|---:|']
    for method in METHODS:
        selected=[r for r in rows if r['method']==method]
        mark='†' if json.loads(selected[0]['foreign_gpu_pids']) else ''
        lines.append(f'| {NAMES[method]}{mark} | '+ ' | '.join(f'{r["train_loop_seconds"]:.2f}' for r in selected)+f' | {selected[-1]["train_ratio_to_direct"]:.3f}× |')
    lines+=['',f'五组均使用 {rows[0]["gpu_model"]}，UUID `{rows[0]["gpu_uuid"]}`。',
            f'短复测五组的初始validation均完全相同，Top-1/Top-3为{initial_comparison["timing_validation"]["top1_retrieval_accuracy"]:.2f}%/{initial_comparison["timing_validation"]["top3_retrieval_accuracy"]:.2f}%；正式CIFAR五组的共同初始值为{initial_comparison["formal_validation"]["top1_retrieval_accuracy"]:.2f}%/{initial_comparison["formal_validation"]["top3_retrieval_accuracy"]:.2f}%。正式CIFAR使用{formal_device}，本次短计时使用{rows[0]["gpu_model"]}；相同源码、配置与权重不意味着跨GPU型号输出逐位相同，具体数值差异的算子来源尚未定位。这里仅比较短复测内部的时间，不拼接两批validation准确度。',
            '† 表示检测到同卡外部计算PID，不能作为严格独占GPU计时。CPU/磁盘仍为共享资源，每阶段只有一个第二轮观测值，不能据小差值建立稳定加速结论。',
            '全部30个epoch及分项时间见 [timing_recheck_epochs.csv](timing_recheck_epochs.csv)，不是挑选各组最快epoch。该短协议不用于比较最终准确度或收敛优势。']
    (output/'同卡短计时复测.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
