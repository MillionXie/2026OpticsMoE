"""Audit all predefined unseen-class runs and summarize paired support draws."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

TASK = Path(__file__).resolve().parent
METHODS = ['direct', 'qwen_vision_lora']


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def csv_write(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--prefix', default='20260909_unseen_formal')
    p.add_argument('--output', default=str(TASK/'reports/unseen_v1_20260909'))
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True,exist_ok=True)
    metrics = []; timings = []; evidence = []; contracts = {}; source_hashes = {}; training_shas = set()
    source_classes = {3,13,14,17,28,31,35,81,86,94}
    expected_sets = {'a':{22,25,50,52,60,66,68,72,77,80}, 'b':{5,18,20,21,26,32,34,55,85,91}}
    for task in ('a','b'):
      for shots in (5,20):
       for seed in (101,202,303):
        for method in METHODS:
            run_id = f'{args.prefix}_{task}_{method}_{shots}shot_s{seed}'
            run = TASK/'runs/simulation'/run_id
            receipt = {r['file']:r for r in read(run/'transfer_manifest.json')}
            files = ['protocol.json','status.json','split.json','source_checkpoint.json','environment.json','history.json',
                     'initial_metrics.json','final_report.json','gpu_execution.json','expert_bank.pt','data_hashes.json',
                     'frozen_source_bank_predictions.json','adapted_predictions.json']
            if method.startswith('qwen'): files += ['qwen_reference_swap_only_predictions.json','fixed_references.json']
            for name in files:
                digest = hashlib.sha256((run/name).read_bytes()).hexdigest()
                assert digest == receipt[name]['sha256'], (run_id,name)
                evidence.append({'run_id':run_id, **receipt[name]})
            weights = read(run/'checkpoint_hashes.json')
            assert {r['file'] for r in weights} == {'best_checkpoint.pt','last_checkpoint.pt'}
            evidence.extend({'run_id':run_id, **r} for r in weights)
            assert read(run/'status.json')['status'] == 'complete'
            report = read(run/'final_report.json'); split = read(run/'split.json'); history = read(run/'history.json')
            gpu = read(run/'gpu_execution.json'); env = read(run/'environment.json'); source = read(run/'source_checkpoint.json')
            assert report['method']==method and report['shots']==shots and report['support_seed']==seed and report['class_set']==task
            assert report['epochs']==5 and report['steps_per_epoch']==20 and len(history)==5 and not report['smoke']
            assert report['export_max_error'] <= 1e-5 and not report['query_used_for_gradient']
            assert source['git_sha']=='c024b9280433f6e7fe31fc0122a1b8aadf342b38'
            assert source_hashes.setdefault(method,source['sha256']) == source['sha256'] == report['source_checkpoint_sha256']
            assert set(split['source_class_ids'])==source_classes and set(split['novel_class_ids'])==expected_sets[task]
            assert not source_classes & expected_sets[task]
            assert len(split['support'])==10*shots and len(split['query'])==1000
            assert all(s['source_split']=='official_train' for s in split['support'])
            assert all(s['source_split']=='official_test' for s in split['query'])
            ids = {s['sample_id'] for s in split['support']}; query_ids = {s['sample_id'] for s in split['query']}
            assert not ids & query_ids
            contract = (sorted(ids), sorted(query_ids), gpu['gpu_uuid'])
            assert contracts.setdefault((task,shots,seed),contract)==contract
            if method.startswith('qwen'):
                refs=read(run/'fixed_references.json')['references']
                assert len(refs)==10 and {s['sample_id'] for s in refs} <= ids
            assert gpu['returncode']==0 and gpu['status']=='complete' and gpu['own_gpu_seen']
            assert 'RTX' in env['device'] and 'A100' not in env['device'] and gpu['gpu_uuid']==env['gpu_uuid']
            assert all(abs(x-.6)<1e-6 for a in report['fusion'].values() for x in a)
            training_shas.add(report['git_sha']); assert env['git_sha']==report['git_sha']
            for h in history:
                assert h['steps']==20 and h['frozen_parameter_max_change']==0
                assert min(h['task_gradient_chain'].values())>0
                timings.append({'run_id':run_id,'class_set':task,'shots':shots,'support_seed':seed,'method':method,
                    'epoch':h['epoch'],'train_loop_seconds':h['train_loop_seconds'],'epoch_wall_seconds':h['epoch_wall_seconds'],
                    'gpu_uuid':gpu['gpu_uuid'],'foreign_pids':json.dumps(gpu['foreign_pids'])})
            for mode in ('frozen_source_bank','qwen_reference_swap_only','adapted'):
                m=report[mode]
                if m is None: continue
                preds=read(run/f'{mode}_predictions.json')
                assert {r['sample_id'] for r in preds}==query_ids and len(preds)==1000
                for key, field in [('top1_retrieval_accuracy','top1_correct'),('top3_retrieval_accuracy','top3_correct')]:
                    assert abs(sum(r[field] for r in preds)/1000-m[key])<1e-6
                metrics.append({'run_id':run_id,'class_set':task,'shots':shots,'support_seed':seed,'method':method,'mode':mode,
                    'top1_percent':100*m['top1_retrieval_accuracy'],'top3_percent':100*m['top3_retrieval_accuracy'],
                    'mrr':m['mrr'],'mean_train_seconds':statistics.mean(h['train_loop_seconds'] for h in history),
                    'mean_epoch_seconds':statistics.mean(h['epoch_wall_seconds'] for h in history),
                    'phase_rms_change_rad':report['expert_phase']['rms_change_rad'], 'gpu_uuid':gpu['gpu_uuid'],
                    'foreign_pids':json.dumps(gpu['foreign_pids'])})
    for task in ('a','b'):
        for seed in (101,202,303):
            assert set(contracts[(task,5,seed)][0]) <= set(contracts[(task,20,seed)][0])
    assert len(training_shas)==1
    summary=[]
    for task in ('a','b'):
      for shots in (5,20):
       for method in METHODS:
        for mode in ('frozen_source_bank','qwen_reference_swap_only','adapted'):
            rows=[r for r in metrics if (r['class_set'],r['shots'],r['method'],r['mode'])==(task,shots,method,mode)]
            if not rows:continue
            assert len(rows)==3
            summary.append({'class_set':task,'shots':shots,'method':method,'mode':mode,
                **{f'{key}_{stat}':fn(r[key] for r in rows) for key in ('top1_percent','top3_percent','mean_epoch_seconds')
                   for stat,fn in [('mean',statistics.mean),('sd',statistics.stdev)]}})
    gains=[]
    for task in ('a','b'):
      for shots in (5,20):
       for method in METHODS:
        pairs=[]
        for seed in (101,202,303):
            rows={r['mode']:r for r in metrics if (r['class_set'],r['shots'],r['support_seed'],r['method'])==(task,shots,seed,method)}
            pairs.append(rows['adapted']['top1_percent']-rows['frozen_source_bank']['top1_percent'])
        gains.append({'class_set':task,'shots':shots,'method':method,'top1_gain_pp_mean':statistics.mean(pairs),
                      'top1_gain_pp_sd':statistics.stdev(pairs),'three_paired_gains_pp':json.dumps(pairs)})
    csv_write(out/'metrics.csv',metrics);csv_write(out/'epoch_times.csv',timings)
    csv_write(out/'summary.csv',summary);csv_write(out/'paired_adaptation_gains.csv',gains)
    (out/'evidence_manifest.json').write_text(json.dumps({'training_git_sha':list(training_shas)[0],
        'source_checkpoint_hashes':source_hashes,'verified_artifacts':evidence},indent=2),encoding='utf-8')
    lines=['# 未见类别：固定专家迁移与少样本适配','',
           'CIFAR源十类之外的两组十类任务；均值 ± 样本标准差来自三次支持集抽样，不是三个源模型训练seed。Top-1/Top-3均为百分比。', '',
           '| 任务 | 每类支持数 | 方法 | 评估方式 | Top-1 | Top-3 |', '|---|---:|---|---|---:|---:|']
    labels={'direct':'直接相位','qwen_vision_lora':'Qwen视觉空间'}
    modes={'frozen_source_bank':'源mask不更新','qwen_reference_swap_only':'只换参考图，不反传','adapted':'仅expert适配100步'}
    for r in summary:
        lines.append(f"| {r['class_set'].upper()} | {r['shots']} | {labels[r['method']]} | {modes[r['mode']]} | {r['top1_percent_mean']:.2f} ± {r['top1_percent_sd']:.2f} | {r['top3_percent_mean']:.2f} ± {r['top3_percent_sd']:.2f} |")
    lines += ['', '支持集同时作为gallery；5/20-shot差异也包含类别原型样本数量变化。未见指未进入源CIFAR任务训练，不能保证预训练概念未见，历史其他实验曾评估CIFAR全百类。',
              '两方法各自继承源模型，电子权重和源相位并不相同。适配时它们全部冻结，仅expert或Qwen视觉LoRA/解码器更新；需同时看各自适配增量。',
              'Qwen更换条件后的直接生成沿用源offset，是单源任务训练生成器的外推诊断；正式适配重新居中到源mask，不能把该诊断与适配起点混为一谈。',
              '全部适配固定末轮，不使用新类别query选模。源训练从raw0开始，适配为保留已学相位的续训。', '',
              '[逐次结果](metrics.csv)、[120轮时间](epoch_times.csv)、[配对增量](paired_adaptation_gains.csv)、[证据](evidence_manifest.json)。',
              '时间为CUDA同步后的训练循环/含审计保存的epoch时间；源训练、模型加载、初始/最终检索评估及时间文件写入不计入epoch。共享GPU影响由foreign_pids字段保留。',
              '[唯一复现入口](../reproduction/README.md)。']
    (out/'完整结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharey='row')
    for col,task in enumerate(('a','b')):
      for row,key in enumerate(('top1_percent','top3_percent')):
        ax=axes[row,col]
        for method,color in [('direct','#2878b5'),('qwen_vision_lora','#c45d2d')]:
          for mode,style in [('frozen_source_bank','--'),('adapted','-')]:
            rows=[next(r for r in summary if (r['class_set'],r['shots'],r['method'],r['mode'])==(task,k,method,mode)) for k in (5,20)]
            ax.errorbar([5,20],[r[key+'_mean'] for r in rows],yerr=[r[key+'_sd'] for r in rows],
                        color=color,linestyle=style,marker='o',capsize=4,label=f"{'Direct' if method=='direct' else 'Qwen'} / {'frozen' if mode=='frozen_source_bank' else 'adapted'}")
        ax.set_title(f"Novel task {task.upper()} / {'Top-1' if row==0 else 'Top-3'}")
        ax.set_xticks([5,20]);ax.set_xlabel('Labeled support images per class');ax.set_ylabel('Accuracy (%)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Class-disjoint transfer: 3 support draws, mean +/- sample SD')
    fig.tight_layout();fig.savefig(out/'unseen_accuracy.png',dpi=180);plt.close(fig)
    print(json.dumps({'runs':24,'metric_rows':len(metrics),'epoch_rows':len(timings),'summary':summary,'gains':gains},indent=2))


if __name__=='__main__':main()
