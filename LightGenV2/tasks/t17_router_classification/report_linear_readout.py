"""Keep new Linear-readout results distinct from historical CCD-only LG runs."""
from pathlib import Path
import datetime
import html
import json


def read(path):return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def main():
    task=Path(__file__).resolve().parent;root=task/'runs/simulation/panel_ab_linear_readout_s17_20261031'
    cfg=read(task/'configs/panel_ab_linear_readout_s17_20261031.json');records={}
    for worker,names in enumerate(cfg['worker_jobs']):
        for name in names:
            path=root/f'worker{worker}'/name;records[name]=dict(run=str(path.relative_to(task)),profile=cfg['jobs'][name],
                status=read(path/'status.json') or dict(state='not_started'),result=read(path/'result.json'))
    oldA=read(task/'reports/panel_a_original_test_development_20261031/results.json')
    baselineA={r['dataset']:r['D2NN_development_reference']['original_test_dev']['accuracy'] for r in oldA['cases']}
    baselineB={depth:read(task/f'runs/simulation/panel_b_d2nn60_s17_20261031/d2nn{depth}/result.json')['test']['accuracy'] for depth in (3,5,7)}
    rows=''
    layouts=[('a','a_kather2016',baselineA['kather2016']),('a','a_bloodmnist128',baselineA['bloodmnist128']),
             ('a','a_crc9',baselineA['crc9']),('b: main2','a_kather2016',baselineB[3]),
             ('b: main4','b_kather4',baselineB[5]),('b: main6','b_kather6',baselineB[7])]
    comparisons=[]
    for panel,name,baseline in layouts:
        record=records[name];result=record['result'];job=record['profile'];gain=100*(result['original_test_dev']['accuracy']-baseline) if result else None
        comparisons.append(dict(panel=panel,LG_job=name,D2NN_original_test_dev=baseline,gain_pp=gain))
        values=[f'{100*result[k]["accuracy"]:.2f}%' for k in ('train','val','original_test_dev')] if result else ['—']*3
        phase=50176+429188*(job['main_layers']//2);head=784*job['num_classes']
        state=f"完成{result['completed_epochs']}轮头训练，选E{result['selected_epoch']}" if result else record['status']['state']
        rows+=f'<tr><td>{panel}</td><td>{name}</td><td>{phase}+{head}</td><td>{html.escape(state)}</td>'+''.join(f'<td>{x}</td>' for x in values)+f'<td>{100*baseline:.2f}%</td><td>{gain:+.2f} pp</td></tr>' if gain is not None else f'<tr><td>{panel}</td><td>{name}</td><td>{phase}+{head}</td><td>{html.escape(state)}</td>'+''.join(f'<td>{x}</td>' for x in values)+f'<td>{100*baseline:.2f}%</td><td>—</td></tr>'
    out=task/'reports/panel_ab_linear_readout_20261031';out.mkdir(exist_ok=True)
    data=dict(role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT',independent_test_claim=False,records=records,comparisons=comparisons,
              B2_reuses_A1_same_checkpoint=True,D2NN_unchanged=True,generated_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (out/'results.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>图a/b：统一末端Linear</title>
<style>body{{font:16px/1.7 sans-serif;max-width:1280px;margin:30px auto;padding:0 16px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}.note{{background:#fff1d5;padding:14px}}</style>
<h1>图a/b：LG末端Linear新版本，seed17</h1><p class="note">新架构：光学CCD→28×28平均池化→固定均值归一→单层无偏置Linear(784,C)。无新MLP/可训练前端。旧无Linear准确率仅作为历史保留，不能移植到新表。光学父权重已受原TEST开发影响，因此仍是开发结果。</p>
<p>首阶段冻结光学相位，只用原TRAIN真标签训练读出，VAL选模；不把新TEST标签或教师用于头训练。最终整模型重新推理全部TEST并与绑定SHA的特征缓存比较。后续若联合微调另计预算。</p>
<table><tr><th>图</th><th>LG任务</th><th>相位+电子参数</th><th>状态</th><th>TRAIN</th><th>VAL</th><th>原TEST开发</th><th>固定D2NN</th><th>差值</th></tr>{rows}</table>
<p>图a的D2NN三层478²，无OEO、无Linear；图b的D2NN3/5/7层478²，逐层同OEO、无Linear。所有D2NN保持已完成60轮。B两层LG直接复用A Kather同一新权重。</p>
<p>LG仍为四专家224²、入口Top2、gap30、global478、无可训前端；每个E/G后原intensity_softsign OEO，含末层。新增电子读出及不同训练预算公开列示，不能称纯光学或等预算因果消融。</p>
<p>图d按人工要求暂缓；图f的数据范围待回复，其他图的要求及旧实现差异记录于architecture_contract.json。生成UTC：{data['generated_UTC']}。</p></html>'''
    (out/'results.html').write_text(page,encoding='utf-8');print([(n,r['status']['state']) for n,r in records.items()])


if __name__=='__main__':main()
