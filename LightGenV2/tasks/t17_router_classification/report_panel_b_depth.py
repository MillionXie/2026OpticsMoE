"""Read completed panel-B receipts; never select a checkpoint or change metrics."""
from pathlib import Path
import datetime
import html
import json
import numpy as np

task=Path(__file__).resolve().parent
run=task/'runs/simulation/panel_b_depth_s17_20261031'
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
            path=run/f'worker{worker}'/job;result=read(path/'result.json')
            records[job]=dict(reused=False,state=read(path/'status.json') or dict(state='not_started'),
                              worker_state=read(path.parent/'status.json'),result=result)
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
                          target_15pp_met=(gain>=15 if gain is not None else None)))
    def increasing(names):
        scores=[records[n]['result']['original_test_dev']['accuracy'] if records[n]['result'] else None for n in names]
        return None if any(s is None for s in scores) else all(a<b for a,b in zip(scores,scores[1:]))
    d_histories=[read(run/'worker0'/n/'history.json') or [] for n in d_names]
    n=min(map(len,d_histories));order_match=all([x['order_sha256'] for x in h[:n]]==[x['order_sha256'] for x in d_histories[0][:n]] for h in d_histories)
    data=dict(role='ORIGINAL_TEST_EXPOSED_PANEL_B_DEVELOPMENT',independent_test_claim=False,
              seed=17,generated_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              all_five_new_jobs_completed=all(records[n]['result'] is not None for n in d_names+lg_names[1:]),
              reused_LG2_is_separate_prior_optimization_budget=True,records=records,pairs=pairs,
              LG_strictly_increasing=increasing(lg_names),D2NN_strictly_increasing=increasing(d_names),
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
            s=row['state'];status=f"{s['state']}；{s.get('epoch',0)}/120轮";values=['—']*3
        rows+=f'<tr><td>{name}</td><td>{html.escape(status)}</td>'+''.join(f'<td>{v}</td>' for v in values)+'</tr>'
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>图b单seed17训练</title>
<style>body{{font:16px/1.7 sans-serif;max-width:1080px;margin:35px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}.notice{{background:#fff0d7;padding:16px}}</style>
<h1>图b：四专家LG与逐层OEO D2NN，seed17</h1>
<p class="notice">原TEST已用于教师训练及学生伪标签输入；结果为开发指标。最低LG复用图a额外优化后的权重，其他五组从头训练，不能称六组全部同训练预算。未完成模型不展示中途TEST分数。</p>
<table><tr><th>模型</th><th>状态</th><th>TRAIN</th><th>VAL</th><th>原TEST开发</th></tr>{rows}</table>
<p>对应关系：LG主干2/4/6层，另有一个入口router；D2NN为3/5/7层。四专家224²，global及D2NN相位均478²；同一intensity-softsign OEO、固定八区CCD，无Linear/CNN。</p>
<p>五组基础预算均120轮：前60轮边界CE，后60轮从各自VAL最优继续focal gamma2；D2NN三组使用相同优化器、初始化、采样、教师数据政策和选模规则，不早停。</p>
<p>深度单调及LG高15个百分点是待检验的目标，不通过选择较差D2NN权重、删除样本或更改统计强制满足。最低LG固定61.04%时，D2NN3最多346/752正确才能达到15个百分点差值。</p>
<p>LG深度严格递增：{data['LG_strictly_increasing']}；D2NN深度严格递增：{data['D2NN_strictly_increasing']}。None表示尚未完成。</p>
<p>数据、源SHA、逐轮指标、入口专家分配与best/last保存在各run。生成时间UTC：{data['generated_UTC']}。</p></html>'''
    (out/'results.html').write_text(page,encoding='utf-8')
    print([(n,records[n].get('state',{}).get('state','reused'),bool(records[n]['result'])) for n in records])


if __name__=='__main__':main()
