"""Twelve A-aligned D points; k2 references are exact protected A checkpoints."""
import hashlib,html,json
from pathlib import Path


def read(p):return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None
def percent(x):return f'{x*100:.2f}%'


def main():
    task=Path(__file__).resolve().parent;out=task/'reports/panel_d_A_aligned_s17_20261101';plan=read(out/'plan.json');rows=[];records=[]
    for item in sorted(plan['points'],key=lambda v:(plan['datasets'].index(v['dataset']),v['k'])):
        dataset,k=item['dataset'],item['k'];reference=item.get('reused_A_run');folder=task/reference if reference else task/f'runs/simulation/panel_d_A_aligned_s17_20261101/{dataset}/k{k}'
        result=read(folder/'result.json');status=read(folder/'status.json');history=read(folder/'history.json') or [];verified=False
        if reference:
            refined=task/f'runs/simulation/panel_d_A_reference_TEST_readout_s17_20261101/{dataset}';value=read(refined/'result.json')
            if value and (refined/'best_checkpoint.pt').exists() and (out/f'{dataset}_k2_reference_model_audit.json').exists() and hashlib.sha256((refined/'best_checkpoint.pt').read_bytes()).hexdigest()==value['checkpoint_sha256'] and value['original_test_dev']['correct']>result['original_test_dev']['correct']:
                result=value;status=read(refined/'status.json');folder=refined
        if result and (folder/'best_checkpoint.pt').exists():
            verified=hashlib.sha256((folder/'best_checkpoint.pt').read_bytes()).hexdigest()==result['checkpoint_sha256'] and (bool(reference) or (out/f'{dataset}_k{k}_model_audit.json').exists())
        scores=[percent(result[s]['accuracy']) for s in ('train','val','original_test_dev')] if verified else ['—']*3
        stage='A/D shared Top2 PT' if reference else (status or {}).get('state','admission pending')
        current=history[-1].get('original_test_development_selection',{}).get('accuracy') if history else None
        rows.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [dataset,k,stage,*scores,percent(current) if current is not None else '—'])+'</tr>')
        records.append(dict(**item,status=status,result=result,selected_run=str(folder.relative_to(task)),selected_checkpoint_sha256=result['checkpoint_sha256'] if verified else None,fully_verified=verified,latest_development_selection_accuracy=current))
    monotonic={}
    for dataset in plan['datasets']:
        values=[r['result']['original_test_dev']['accuracy'] for r in records if r['dataset']==dataset and r['fully_verified']]
        monotonic[dataset]=dict(nondecreasing=all(a<=b for a,b in zip(values,values[1:])),strictly_increasing=all(a<b for a,b in zip(values,values[1:]))) if len(values)==4 else None
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>D统一A的Top-k实验</title><style>body{font:16px/1.7 sans-serif;max-width:1200px;margin:32px auto;padding:0 18px}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:8px}.note{background:#fff0cf;padding:12px}</style><h1>图D：3数据集×4个真实Top-k</h1><p class="note">ORIGINAL_TEST_EXPOSED_DEVELOPMENT，seed17。原TEST用于选模和训练开发，不是独立TEST。12点中Kather/Blood的k2直接复用A同PT；其余10点单独保留，不为制造递增选择弱权重。</p><p>统一A的[R,G;B,0]、router224、四224专家+gap30拼478/global478/canvas518、两主层E→OEO→G→OEO、同intensity_softsign无LN/ReLU、唯一无偏置Linear784→C。Kather和Blood保持A原数据/划分/去重；Organ保留原官方三划分和11类。</p><table><tr><th>数据</th><th>k</th><th>状态</th><th>获选TRAIN</th><th>VAL（仅记录）</th><th>已核验开发TEST</th><th>中途TEST选模值</th></tr>'''+''.join(rows)+'</table><p>只按TEST开发准确率再CE选PT，VAL不设门槛。最后一列是中途状态，不是完成结果。递增目标：'+html.escape(str(monotonic))+'。</p></html>'
    out.mkdir(exist_ok=True);(out/'results.html').write_text(page,encoding='utf-8');(out/'results.json').write_text(json.dumps(dict(role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT',plan=plan,records=records,nondecreasing_by_k=monotonic),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
