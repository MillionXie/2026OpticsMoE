import torch
from .core import *

def report():
    authorize();cfg=plan();split=prepare();root=ROOT/'runs/merge_seed42'
    records={stage:json.loads((root/stage/'final.json').read_text()) for stage in cfg['stages']}
    merge_info=json.loads((root/'merge_checks.json').read_text())
    state=load_checkpoint(root/'router/selected.pt',split);shared=load_checkpoint(root/'shared/selected.pt',split)
    loaded,s=setup(cfg['seed'],root);r,h=build(loaded,s);M.restore(r,h,state);original_digest=M.digest(r,h)
    automatic=evaluate(loaded,r,h,s,split);uniform=evaluate(loaded,r,h,s,split,'uniform');isolated=evaluate(loaded,r,h,s,split,'isolated')
    for label in ('A','B'):
        if isolated[label]['accuracy']!=merge_info['isolated_validation'][label]['accuracy']:raise RuntimeError('Router training changed isolated expert predictions')
    ablations={}
    for domain,name in ((0,'restore_A_initial_phase'),(1,'restore_B_initial_phase')):
        M.restore(r,h,state)
        with torch.no_grad():
            for n,p in M.named(r,h).items():
                i=expert_number(n)
                if i is not None and i//2==domain:p.copy_(state_tensors(shared)[n].to(p.device))
        ablations[name]=evaluate(loaded,r,h,s,split)
    M.restore(r,h,state);assert original_digest==M.digest(r,h),'Ablation restoration changed normal state'
    imgs=[Images(split,d,'validation')[i][0] for i in range(5) for d in (0,1)]
    with torch.no_grad(),E.autocast(loaded,s):
        a=forward(loaded,r,h,E.base._prepare(loaded,imgs,s)).float();b=forward(loaded,r,h,E.base._prepare(loaded,list(reversed(imgs)),s)).float().flip(0)
    torch.testing.assert_close(a,b,atol=.02,rtol=.002);assert torch.equal(a.argmax(1),b.argmax(1))
    drops={d:isolated[d]['accuracy']-automatic[d]['accuracy'] for d in ('A','B')}
    task_pass=all(isolated[d]['accuracy']>=cfg['targets']['minimum_isolated_accuracy'] and drops[d]<=cfg['targets']['maximum_merge_drop'] for d in ('A','B'))
    result=dict(status='complete',experiment_type=cfg['experiment_type'],partition='validation',test_evaluated=False,stage_results=records,
        independent_experts=isolated,merged_uniform=uniform,merged_automatic=automatic,ablations=ablations,merge_drop=drops,task_target_passed=task_pass,
        same_checkpoint_for_both_domains=True,mixed_order_invariant=True,source_merge_checks=merge_info,
        mechanism_accuracy_drop={d:automatic[d]['accuracy']-ablations['restore_'+d+'_initial_phase'][d]['accuracy'] for d in ('A','B')},
        next_action='Report results and wait for user; no D1 or further training is scheduled.')
    atomic_json(root/'results.json',result)
    rows=[('共享底座／组隔离',records['shared']['validation']),('独立专家／组隔离',isolated),('合并／固定均匀路由',uniform),('合并／自动路由',automatic),('恢复 A 专家初始相位',ablations['restore_A_initial_phase']),('恢复 B 专家初始相位',ablations['restore_B_initial_phase'])]
    lines=['# Office-Home 分域专家合并结果','', '全部为验证集结果；共享部分使用过 A+B，不属于严格顺序迁移。','', '| 条件 | A 准确率 | B 准确率 | A Macro-F1 | B Macro-F1 | 两域准确率均值 |','|---|---:|---:|---:|---:|---:|']
    for name,record in rows:lines.append('| '+name+' | '+' | '.join(f'{100*v:.2f}%' for v in (record['A']['accuracy'],record['B']['accuracy'],record['A']['macro_f1'],record['B']['macro_f1'],record['mean']))+' |')
    lines+=['',f"目标通过：{task_pass}。A/B 合并下降分别为 {100*drops['A']:.2f} / {100*drops['B']:.2f} 个百分点。",'', '组隔离是独立专家诊断条件；正常最终推理只使用一个检查点、固定相位及自动四路权重，不接收域标签。D1 未启动。']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');r.close()

if __name__=='__main__':report()
