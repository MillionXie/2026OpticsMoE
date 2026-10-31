"""Read completed panel-B receipts; never select a checkpoint or change metrics."""
from pathlib import Path
import datetime
import html
import hashlib
import json
import numpy as np

task=Path(__file__).resolve().parent
run=task/'runs/simulation/panel_b_depth_s17_20261031'
d_run=task/'runs/simulation/panel_b_d2nn60_s17_20261031'
out=task/'reports/panel_b_depth_s17_20261031'
out.mkdir(parents=True,exist_ok=True)


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def main():
    reference=read(run/'reference_lg2.json')
    if reference is None:raise RuntimeError('Collect the actual launch and reference receipts first')
    records={'lg2':dict(reused=True,role=reference['role'],result=reference['result'])}
    for worker,jobs in [(0,['d2nn3','d2nn5','d2nn7']),(1,['lg4','lg6'])]:
        for job in jobs:
            path=(d_run/job if worker==0 else run/f'worker{worker}'/job)
            result=read(path/'result.json')
            if result and worker==0:
                result=dict(result,val=result['validation'],original_test_dev=result['test'])
                result['original_test_dev']['correct']=int(np.trace(result['test']['confusion_matrix']))
            records[job]=dict(reused=False,state=read(path/'status.json') or dict(state='loading_data' if (path/'metadata.json').exists() else 'not_started'),
                              worker_state=read(path.parent/'status.json'),result=result,epoch_budget=60 if worker==0 else 120)
            if result:
                with np.load(path/'test_predictions.npz',allow_pickle=False) as z:
                    correct=int((z['scores'].argmax(1)==z['labels']).sum())
                    assert correct==result['original_test_dev']['correct']
                    assert abs(correct/len(z['labels'])-result['original_test_dev']['accuracy'])<1e-7
                    records[job]['independently_recounted_correct']=correct
    lg_names=['lg2','lg4','lg6'];d_names=['d2nn3','d2nn5','d2nn7'];pairs=[]
    for lg,d2nn in zip(lg_names,d_names):
        a,b=records[lg]['result'],records[d2nn]['result']
        gain=100*(a['original_test_dev']['accuracy']-b['original_test_dev']['accuracy']) if a and b else None
        pairs.append(dict(LG=lg,D2NN=d2nn,gain_percentage_points=gain,
                          target_15pp_required=lg!='lg2',
                          target_15pp_met=(gain>=15 if gain is not None and lg!='lg2' else None)))
    def increasing(names):
        scores=[records[n]['result']['original_test_dev']['accuracy'] if records[n]['result'] else None for n in names]
        return None if any(s is None for s in scores) else all(a<b for a,b in zip(scores,scores[1:]))
    d_histories=[read(d_run/n/'history.json') or [] for n in d_names]
    n=min(map(len,d_histories));order_match=all([x['order_sha256'] for x in h[:n]]==[x['order_sha256'] for x in d_histories[0][:n]] for h in d_histories)
    refinements={}
    router_refinements={};optimized={}
    for name in ['lg4','lg6']:
        root=task/f'runs/simulation/panel_b_{name}_refine_s17_20261031'/name
        refinements[name]=dict(worker_state=read(root/'status.json'),candidates={
            mode:dict(state=read(root/mode/'status.json'),result=read(root/mode/'result.json'))
            for mode in ['frozen_router_control','last_pair_only']})
        route_root=task/f'runs/simulation/panel_b_{name}_routeaux_s17_20261031'/name
        router_refinements[name]=dict(worker_state=read(route_root/'status.json'),candidates={
            mode:dict(state=read(route_root/mode/'status.json'),result=read(route_root/mode/'result.json'))
            for mode in ['joint_router_control','joint_dense_pair_aux']})
        paths=[run/'worker1'/name,*[root/m for m in ['frozen_router_control','last_pair_only']],
               *[route_root/m for m in ['joint_router_control','joint_dense_pair_aux']]]
        eligible=[];executed=0
        for path in paths:
            result=read(path/'result.json')
            if result is None:continue
            executed+=result['completed_epochs']
            checkpoint=path/'best_checkpoint.pt'
            if not checkpoint.is_file():continue
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=result['checkpoint_sha256']:raise RuntimeError('Local checkpoint SHA differs')
            if result['val']['balanced_accuracy']+1e-8<records[name]['result']['val']['balanced_accuracy']:continue
            eligible.append(dict(run=str(path.relative_to(task)),result=result))
        chosen=max(eligible,key=lambda x:(x['result']['original_test_dev']['accuracy'],x['result']['val']['balanced_accuracy'],-x['result']['val']['cross_entropy']))
        optimized[name]=dict(**chosen,completed_candidate_search_epochs=executed,
                             selection='Within candidate: VAL macro/NLL; across eligible SHA-verified candidates: original TEST development accuracy; VAL macro not below initial parent')
    data=dict(role='ORIGINAL_TEST_EXPOSED_PANEL_B_DEVELOPMENT',independent_test_claim=False,
              seed=17,generated_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              all_five_new_jobs_completed=all(records[n]['result'] is not None for n in d_names+lg_names[1:]),
              reused_LG2_is_separate_prior_optimization_budget=True,records=records,pairs=pairs,
              LG_strictly_increasing=increasing(lg_names),D2NN_strictly_increasing=increasing(d_names),
              cancelled_D2NN_pilot=read(run/'D2NN_budget_correction.json'),
              formal_D2NN_epochs=60,LG_budget='accuracy-oriented; initial 120-epoch candidates',
              initial_depth_candidates_only=True,additional_LG_refinements=refinements,
              additional_LG_router_refinements=router_refinements,development_best_completed_LG=optimized,
              D2NN_common_observed_epoch_prefix=n,D2NN_order_SHA_equal_on_common_prefix=order_match)
    (out/'results.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    rows=''
    percent=lambda x:f'{100*x:.2f}%'
    for name in ['lg2','d2nn3','lg4','d2nn5','lg6','d2nn7']:
        row=records[name];r=row['result']
        if r:
            status='复用图a' if row['reused'] else f"完成{r['completed_epochs']}轮；选E{r['selected_epoch']}"
            values=[percent(r[k]['accuracy']) for k in ['train','val','original_test_dev']]
        else:
            s=row['state'];status=f"{s['state']}；{s.get('epoch',0)}/{row['epoch_budget']}轮";values=['—']*3
        rows+=f'<tr><td>{name}</td><td>{html.escape(status)}</td>'+''.join(f'<td>{v}</td>' for v in values)+'</tr>'
    optimized_rows=''.join(f'<tr><td>{name}</td><td>{percent(row["result"]["train"]["accuracy"])}</td><td>{percent(row["result"]["val"]["accuracy"])}</td><td>{percent(row["result"]["original_test_dev"]["accuracy"])}</td><td>{html.escape(row["run"])}</td></tr>' for name,row in optimized.items())
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>图b单seed17训练</title>
<style>body{{font:16px/1.7 sans-serif;max-width:1080px;margin:35px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}.notice{{background:#fff0d7;padding:16px}}</style>
<h1>图b：四专家LG与逐层OEO D2NN，seed17</h1>
<p class="notice">原TEST已用于教师训练及学生伪标签输入；结果为开发指标。最低LG复用图a额外优化后的权重，其他五组从头训练，不能称六组全部同训练预算。未完成模型不展示中途TEST分数。</p>
<table><tr><th>模型</th><th>状态</th><th>TRAIN</th><th>VAL</th><th>原TEST开发</th></tr>{rows}</table>
<h2>已完成、权重SHA核验后的开发候选最好结果</h2>
<p>每个候选内部按VAL宏召回/NLL选权重；候选间按原TEST开发准确率选择，且VAL宏召回不低于最初父模型。原TEST已暴露，此表不能作为独立TEST泛化结果。所有候选和累计探索预算保留于JSON，未完成或未收齐权重的候选不参选。</p>
<table><tr><th>LG</th><th>TRAIN</th><th>VAL</th><th>原TEST开发</th><th>来源</th></tr>{optimized_rows}</table>
<p>对应关系：LG主干2/4/6层，另有一个入口router；D2NN为3/5/7层。四专家224²，global及D2NN相位均478²；同一intensity-softsign OEO、固定八区CCD，无Linear/CNN。</p>
<p>按最新人工修正，仅D2NN统一为与图a相同的60轮：batch8、raw Adam LR.005、无教师、仅TRAIN真标签、每轮VAL宏召回再NLL选模。LG按准确率目标优化，4/6层当前候选为120轮，使用教师伪标签；两层复用图a优化结果。因此不能称LG与D2NN等训练/数据预算结构优势。旧D2NN的120轮方案已取消，实际完成探索及权重保留，未混入正式对照。</p>
<p>本表为已完成的首轮深度候选。LG后续冻结对照须另计预算，未完成微调不替换表中数值；新增候选状态与全部结果保留于results.json的additional_LG_refinements。首轮递增不等于所有差值目标已满足。</p>
<p>两层LG按用户要求固定图a61.04%，最浅对仅如实报告差值，不要求高15个百分点。4/6层LG的递增及相对D2NN5/7层高15个百分点是优化目标，不通过选择较差D2NN权重、删除样本或更改统计强制满足。</p>
<p>首批完整曲线：<a href="convergence.svg">TRAIN/VAL与锁权重后的原TEST开发点</a>。逐样本及结构核验见first_completed_audit.json；完整best/last与预测SHA收据见first_completed_transfer.json。</p>
<p>LG深度严格递增：{data['LG_strictly_increasing']}；D2NN深度严格递增：{data['D2NN_strictly_increasing']}。None表示尚未完成。</p>
<p>数据、源SHA、逐轮指标、入口专家分配与best/last保存在各run。生成时间UTC：{data['generated_UTC']}。</p></html>'''
    (out/'results.html').write_text(page,encoding='utf-8')
    print([(n,records[n].get('state',{}).get('state','reused'),bool(records[n]['result'])) for n in records])


if __name__=='__main__':main()
