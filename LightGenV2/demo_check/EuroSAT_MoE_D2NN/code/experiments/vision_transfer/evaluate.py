"""Official reporting includes failed/blocked arms and their actual budgets."""
import json
from pathlib import Path
import numpy as np
import torch
from . import model as M
from . import final_evaluation as F
from .data import atomic_json, official_b
from .protocol import require_authorization, output_root, load_policy, PROJECT_ROOT
from .evaluation_plan import make_evaluation_plan, seal_plan


def main():
    require_authorization()
    out=output_root();plan=make_evaluation_plan(out)
    marker=Path(load_policy()['official_download_marker'])
    verified=json.loads(marker.read_text()) if marker.exists() else {}
    if (verified.get('status'),verified.get('size'),verified.get('md5'))!=('complete',2918471680,'56bf5dcef84df0e2308c6dcbcbbd8499'):
        raise RuntimeError('Official CIFAR-10-C download and integrity verification are incomplete')
    seal_plan(out,plan)
    s=F.load_settings(PROJECT_ROOT/'experiments/vision_transfer/config.yaml');s.output_dir=out
    torch.set_num_threads(8);F.seed_everything(42)
    bundle=F.prepare_cifar10(s,persist=False);loaded=F.load_backbone(s,torch.device('cuda'));results={}
    for name,item in plan.items():
        if not item['checkpoint']:continue
        dest=out/'evaluation'/name;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'metrics.json').exists():
            old=json.loads((dest/'metrics.json').read_text())
            if old.get('checkpoint_file_sha256')!=item['checkpoint_file_sha256']:raise RuntimeError('Cached evaluation checkpoint mismatch')
            results[name]=old;continue
        arch='d2nn' if name.startswith('d2nn') else 'moe'
        F.seed_everything(42);r,h=M.build(loaded,s,arch)
        checkpoint=torch.load(item['checkpoint'],map_location='cpu',weights_only=False)
        M.restore(r,h,checkpoint);before=M.digest(r,h)
        try:
            a=F.evaluate(loaded,r,h,bundle.test,s,predictions_path=dest/'clean.npz')
            b=F.evaluate_corrupted(loaded,r,h,s,official_b(),dest)
            ac,bc=F.correct_vectors(dest)
            result=dict(**item,checkpoint_state_sha256=before,selected_epoch=checkpoint['epoch'],clean=a,corrupted=b,
                        ci95_clean=F.ci(ac),ci95_corrupted=F.ci(bc),ci95_equal_domain_average=F.ci((ac+bc)/2),
                        corrupted_seen_severity_1_3=float(np.mean([v['accuracy'] for v in b['conditions'] if v['severity']<=3])),
                        corrupted_unseen_severity_4_5=float(np.mean([v['accuracy'] for v in b['conditions'] if v['severity']>=4])),
                        worst_condition=min(b['conditions'],key=lambda v:v['accuracy']))
            if name.endswith('_B'):
                source_name='d2nn_A' if arch=='d2nn' else 'moe_A'
                source=torch.load(plan[source_name]['checkpoint'],map_location='cpu',weights_only=False)
                F.mechanism(loaded,r,h,s,bundle,source,checkpoint,dest)
                clean_images=[bundle.test[i][0] for i in range(20)]
                _,_,corrupted=next(iter(official_b()));corrupt_images=[corrupted[i][0] for i in range(20)]
                def forward(images):
                    inputs=F.base._prepare(loaded,images,s)
                    with torch.no_grad(),F.autocast(loaded,s):return M.predict(loaded,r,h,inputs).float()
                separate=torch.cat([forward(clean_images),forward(corrupt_images)])
                order=np.random.default_rng(241).permutation(40);images=clean_images+corrupt_images
                mixed=forward([images[i] for i in order])[torch.as_tensor(np.argsort(order),device='cuda')]
                passed=torch.equal(separate.argmax(1),mixed.argmax(1)) and torch.allclose(separate,mixed,atol=.025,rtol=.003)
                result['mixed_single_model_check']=dict(passed=bool(passed),maximum_logit_difference=float((separate-mixed).abs().max()),
                                                        no_domain_id=True,no_test_time_updates=True)
                if not passed:raise RuntimeError('Mixed/separate fixed-model prediction contract failed')
                forward([bundle.validation[0][0]]);F.export_visuals(r,h,checkpoint['origin_phases'],dest/'final_visuals')
            if M.digest(r,h)!=before:raise RuntimeError('Evaluation changed model state')
            atomic_json(dest/'metrics.json',result);results[name]=result
        finally:r.close()
    matrix=[]
    for label,src,end in [('M1 reserved MoE','moe_A','moe_reserved_B'),('M2 all-expert MoE','moe_A','moe_all_B'),('D1 D2NN hybrid','d2nn_A','d2nn_d2nn_B')]:
        row=dict(model=label,training_status=plan[end]['training_status'],B_epochs_completed=plan[end]['epochs_completed'],B_epochs_planned=plan[end]['planned_epochs'],
                 routing_qualified=plan[end]['routing_qualified'],retention_satisfied=plan[end]['retention_satisfied'],protocol_succeeded=plan[end]['protocol_succeeded'],reason=plan[end]['reason'])
        for key in ('A_after_A','B_after_A','A_after_B','B_after_B','A_forgetting','B_gain','equal_domain_average'):row[key]=None
        if src in results:row.update(A_after_A=results[src]['clean']['accuracy'],B_after_A=results[src]['corrupted']['accuracy'])
        if end in results:
            aa,ab=row['A_after_A'],row['B_after_A'];ba=results[end]['clean']['accuracy'];bb=results[end]['corrupted']['accuracy']
            sa,sb=F.correct_vectors(out/'evaluation'/src);ea,eb=F.correct_vectors(out/'evaluation'/end)
            row.update(A_after_B=ba,B_after_B=bb,A_forgetting=aa-ba,B_gain=bb-ab,equal_domain_average=(ba+bb)/2,
                       ci95_A_forgetting=F.ci(sa-ea),ci95_B_gain=F.ci(eb-sb))
        matrix.append(row)
    comparisons={}
    if 'd2nn_d2nn_B' in results:
        da,db=F.correct_vectors(out/'evaluation/d2nn_d2nn_B')
        for name in ('moe_reserved_B','moe_all_B'):
            if name not in results:continue
            a,b=F.correct_vectors(out/'evaluation'/name)
            comparisons[name+'_minus_d2nn']=dict(clean_difference=float((a-da).mean()),corrupted_difference=float((b-db).mean()),
                ci95_equal_domain_difference=F.ci((a-da+b-db)/2),moe_B_epochs=plan[name]['epochs_completed'],d2nn_B_epochs=plan['d2nn_d2nn_B']['epochs_completed'],
                equal_completed_epoch_budget=plan[name]['epochs_completed']==plan['d2nn_d2nn_B']['epochs_completed'])
    success=all(row['protocol_succeeded'] for row in matrix)
    report=dict(status='complete' if success else 'complete_with_protocol_failures',matrix=matrix,paired_comparisons=comparisons,checkpoints=results,
                training_outcomes=plan,seed=42,test_used_for_selection=False,bootstrap_unit='original image ID',
                caveat='Failed/blocked arms and actual epochs are retained. MoE automatic gradient probes add diagnostic compute. Single seed; D2NN is a hybrid baseline.')
    atomic_json(out/'final_performance.json',report)
    text=['# Vision-only CIFAR-10 Clean → Corrupted 结果','',
          '每个模型固定一个验证集选出的检查点进行自动推理。失败组、未运行组及实际训练轮数均保留。官方测试没有用于调整超参数或选择检查点。','',
          '| 模型 | 状态 | B轮数 | A后A | A后B | B后A | B后B | 遗忘pp | B变化pp | 两域平均 |',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in matrix:
        values=['未运行' if row[key] is None else f'{row[key]*100:.2f}' for key in ('A_after_A','B_after_A','A_after_B','B_after_B','A_forgetting','B_gain','equal_domain_average')]
        text.append(f"| {row['model']} | {row['training_status']} | {row['B_epochs_completed']}/{row['B_epochs_planned']} | "+' | '.join(values)+' |')
    text+=['','协议验收、保持约束和失败原因见final_performance.json。提前停止组不能作为等训练预算比较，也不能由数值成绩替代路由验收。MoE诊断有额外前向/梯度计算；D2NN保留电子接口及四次CCD边界。']
    (out/'性能报告.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    F.plot_results(out,matrix)
