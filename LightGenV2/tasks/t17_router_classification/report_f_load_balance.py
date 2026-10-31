"""Per-modality actual hard Top-2 usage, bound to the selected joint F PT."""
import hashlib
import html
import json
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def usage_statistics(power, dense):
    power=np.asarray(power,dtype=np.float64);dense=np.asarray(dense,dtype=np.float64)
    if power.ndim!=2 or power.shape[1]!=4 or dense.shape!=power.shape or not len(power):raise ValueError('Nonempty four-expert routes required')
    if not np.isfinite(power).all() or not np.isfinite(dense).all() or (power<0).any() or (dense<0).any():raise ValueError('Invalid route power')
    chosen=power>0
    if not (chosen.sum(1)==2).all() or not np.allclose(power.sum(1),1,atol=1e-5) or not np.allclose(dense.sum(1),1,atol=1e-5):raise ValueError('Normalized actual hard Top-2 required')
    counts=chosen.sum(0);frequency=counts/len(power);share=counts/(2*len(power));nonzero=share[share>0]
    pairs={}
    for mask in chosen:
        pair='+'.join(str(i+1) for i in np.flatnonzero(mask));pairs[pair]=pairs.get(pair,0)+1
    entropy=float(-(nonzero*np.log(nonzero)).sum()/np.log(4))
    return dict(images=len(power),total_expert_selections=2*len(power),expert_selection_counts=counts.tolist(),
        per_image_selection_frequency=frequency.tolist(),selection_slot_share=share.tolist(),
        post_Top2_mean_power=power.mean(0).tolist(),pre_Top2_mean_dense_power=dense.mean(0).tolist(),
        selected_expert_share_target=[.25]*4,per_image_frequency_uniform_reference=[.5]*4,
        unused_experts=[int(i+1) for i in np.flatnonzero(counts==0)],
        normalized_selection_entropy=entropy,effective_experts=float(np.exp(-(nonzero*np.log(nonzero)).sum())),
        selection_share_CV=float(share.std()/share.mean()),maximum_share_deviation_from_uniform=float(np.abs(share-.25).max()),
        pair_counts=pairs,pair_frequencies={k:v/len(power) for k,v in sorted(pairs.items())},
        fraction_outside_most_common_pair=1-max(pairs.values())/len(power))


def main():
    task=Path(__file__).resolve().parent;main_report=json.loads((task/'reports/f_joint_dual_teacher_s17_20261101/results.json').read_text())
    run=task/main_report['selected_joint_run'];result=json.loads((run/'result.json').read_text());checkpoint=run/'best_checkpoint.pt'
    if sha(checkpoint)!=result['checkpoint_sha256']:raise ValueError('Selected joint PT SHA mismatch')
    # Locate a completed remote/local transfer manifest covering this exact run.
    receipts=list((task/'reports').glob('*/completed_transfer.json'));bound={}
    for receipt in receipts:
        value=json.loads(receipt.read_text())
        if not value.get('verified_all_remote_and_local_SHA256'):continue
        for item in value.get('files',[]):
            relative=item['path'].replace('\\','/');wanted=str(run.relative_to(task)).replace('\\','/')+'/'
            if '/'+wanted in relative:bound[Path(relative).name]=dict(**item,receipt=str(receipt.relative_to(task)))
    parts={};evidence={}
    for split in ('train','val','test'):
        p=run/f'{split}_predictions.npz';expected=bound.get(p.name)
        if not expected or sha(p)!=expected['sha256']:raise ValueError('Prediction file lacks matching completed SHA receipt')
        evidence[split]=dict(path=str(p.relative_to(task)),sha256=sha(p),receipt=expected['receipt'])
        with np.load(p,allow_pickle=False) as z:
            if not np.isin(z['domains'],[0,1]).all():raise ValueError('Unexpected modality')
            rows={}
            for d,name in [(0,'RGB'),(1,'SAR')]:
                mask=z['domains']==d;entry=usage_statistics(z['route_power'][mask],z['dense_route_power'][mask])
                entry['accuracy']=float((z['predictions'][mask]==z['labels'][mask]).mean())
                entry['by_class']={str(int(c)):usage_statistics(z['route_power'][mask&(z['labels']==c)],z['dense_route_power'][mask&(z['labels']==c)]) for c in np.unique(z['labels'][mask])}
                rows[name]=entry
            parts[split]=rows
    out=task/'reports/f_joint_load_balance_s17_20261101';out.mkdir(exist_ok=True)
    unused={name:value['unused_experts'] for name,value in parts['test'].items()}
    has_unused=any(unused.values())
    status=('当前尚未均衡：'+str(unused)+'为各模态零选择专家。') if has_unused else '当前无零选择专家；是否足够均衡需结合完整分布与准确率评估，尚未预设通过阈值。'
    data=dict(role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT',selected_joint_run=str(run.relative_to(task)),checkpoint_sha256=sha(checkpoint),
        definitions=dict(selection_frequency='P(expert is in hard Top2); four entries sum to 2, balanced reference 50% each',
            selection_slot_share='Expert selections / (2 * images); four entries sum to 1, balanced reference 25% each',
            post_Top2_mean_power='Mean normalized optical power after the actual hard gate; sum=1; not selection frequency',
            pre_Top2_mean_dense_power='Mean dense normalized detector power before hard gating; cannot demonstrate actual selection balance'),
        balance_requirement='Inspect RGB and SAR separately, all four physical experts, preserve true input-dependent Top2; uniform reference is a diagnostic goal, not a forced per-image dense route or a guarantee.',
        balance_achieved=False if has_unused else None,balance_status=status,parts=parts,prediction_SHA_evidence=evidence)
    # Explicit status avoids declaring uniform soft powers equivalent to balanced hard usage.
    rows=[]
    for split,domains in parts.items():
        for name,value in domains.items():
            percentages=lambda key:' / '.join(f'{100*v:.2f}%' for v in value[key])
            rows.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [split,name,value['images'],percentages('per_image_selection_frequency'),percentages('selection_slot_share'),percentages('post_Top2_mean_power'),f"{value['normalized_selection_entropy']:.3f}"] )+'</tr>')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>F联合LG实际专家占比</title><style>body{font:16px/1.7 sans-serif;max-width:1400px;margin:30px auto;padding:0 16px}td,th{padding:9px;border-bottom:1px solid #ddd}table{border-collapse:collapse;width:100%}.note{background:#fff0cf;padding:14px}</style><h1>F：同一联合LG在RGB/SAR上的真实专家使用</h1><p class="note">当前尚未均衡：第四专家在两域都未进入Top-2，不能将归一化功率或联合平均当均衡。所有TEST值为ORIGINAL_TEST_EXPOSED_DEVELOPMENT。</p><p>每张图选两个专家。被选频率=选中图数/图数，四项合计200%，均衡参考各50%；选择次数占比=选中次数/(2×图数)，合计100%，均衡参考各25%。光功率份额是另一个指标，不代表负载频率。RGB与SAR分别检查，不能合并掩盖单域偏载；不要求每张图四路均匀。</p>'''
    page=page.replace('当前尚未均衡：第四专家在两域都未进入Top-2，不能将归一化功率或联合平均当均衡。',html.escape(status)+'不能将归一化功率或联合平均当均衡。')
    page+='<p>同一模型SHA256：'+sha(checkpoint)+'</p><img src="expert_usage.svg" style="width:100%"><table><tr><th>划分</th><th>模态</th><th>N</th><th>E1/E2/E3/E4被选频率</th><th>选择次数占比</th><th>Top2后平均功率份额</th><th>选择熵/ln4</th></tr>'+''.join(rows)+'</table><p>完整逐类统计、Top-2组合分布、有效专家数与SHA证据见<a href="results.json">JSON</a>。后续训练需兼顾两域准确率与每域实际硬选择负载，保留原模型及失败；不固定域专家、不使用GT路由。</p></html>'
    (out/'results.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');(out/'results.html').write_text(page,encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams['svg.fonttype']='none';colors=['#4477aa','#66ccee','#228833','#cc6677'];fig,axes=plt.subplots(1,2,figsize=(11.2,4.8));names=['RGB','SAR']
    for ax,key,title in [(axes[0],'selection_slot_share','Actual Top-2 selection share'),(axes[1],'post_Top2_mean_power','Mean optical power after Top-2')]:
        bottom=np.zeros(2)
        for i,color in enumerate(colors):
            values=np.array([parts['test'][n][key][i]*100 for n in names]);ax.bar(names,values,bottom=bottom,color=color,label=f'E{i+1}',width=.55)
            for x,(v,y) in enumerate(zip(values,bottom)):
                if v>=3:ax.text(x,y+v/2,f'{v:.2f}%',ha='center',va='center',fontsize=10)
            bottom+=values
        ax.set(title=title,ylim=(0,100),ylabel='Share (%)');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,-.09));fig.suptitle('F / one joint LG / original TEST exposed development\n'+('Unused expert(s) remain; balance not achieved' if has_unused else 'Per-modality usage; inspect load and accuracy together'),fontsize=12);fig.tight_layout(rect=(0,.04,1,.9));fig.savefig(out/'expert_usage.svg');fig.savefig(out/'expert_usage.png',dpi=160);plt.close(fig)
    svg=out/'expert_usage.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print(json.dumps({n:{k:v for k,v in parts['test'][n].items() if k in ('per_image_selection_frequency','selection_slot_share','unused_experts')} for n in names}))


if __name__=='__main__':main()
