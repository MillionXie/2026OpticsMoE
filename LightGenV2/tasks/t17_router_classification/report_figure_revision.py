"""One schedule for the human-approved figure revision, without mixing old metrics."""
from pathlib import Path
import datetime
import html
import json
import statistics


def read(path):return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def main():
    task=Path(__file__).resolve().parent;out=task/'reports/figure_revision_schedule_20261031';board=read(out/'schedule.json');run=task/'runs/simulation/figure_revision_s17_20261031';rows=[];records={}
    for name in ['panel_b_linear_joint4_s17_20261031','panel_b_linear_joint6_s17_20261031','panel_f_lg_s17_20261031','panel_f_d2nn_s17_20261031']:
        root=run/name;status=read(root/'status.json') or dict(state='not_started');history=read(root/'history.json') or [];result=read(root/'result.json');cfg=read(task/'configs'/f'{name}.json')
        timing=statistics.median([r['seconds'] for r in history[-5:]]) if history else None
        remaining=(cfg['epochs']-len(history))*timing/60 if timing and status['state']!='complete' else None
        records[name]=dict(run=str(root.relative_to(task)),status=status,result=result,completed_epoch_timing_seconds=timing,max_budget_remaining_minutes=remaining)
        progress=f"{status.get('epoch',len(history))}/{cfg['epochs']}" if status['state']!='complete' else f"完成{result['completed_epochs']}，选E{result['selected_epoch']}"
        metrics=' / '.join(f"{100*result[k]['accuracy']:.2f}%" for k in ['train','val','original_test_dev']) if result else '完成并锁权重后填写'
        resource='4090 / 两组共享' if name.startswith('panel_b') else 'A100 / 两组共享'
        rows.append(f'<tr><td>{name}</td><td>{resource}</td><td>{html.escape(status["state"])}</td><td>{progress}</td><td>{metrics}</td><td>{remaining:.0f}分钟以内（不计后续测试）</td></tr>' if remaining is not None else f'<tr><td>{name}</td><td>{resource}</td><td>{html.escape(status["state"])}</td><td>{progress}</td><td>{metrics}</td><td>—</td></tr>')
    report=dict(role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT',independent_test_claim=False,generated_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),records=records,schedule=board)
    convex_root=task/'runs/simulation/figure_convex_readout_s17_20261031/worker0';convex_rows=[];report['additional_convex_readout']={}
    for name in ['f_lg_convex','b_lg4_convex','b_lg6_convex']:
        result=read(convex_root/name/'result.json');report['additional_convex_readout'][name]=result
        if result:
            values=' / '.join(f"{100*result[k]['accuracy']:.2f}%" for k in ['train','val','original_test_dev'])
            convex_rows.append(f'<tr><td>{name}</td><td>{result["completed_inner_steps"]}内部迭代/选第{result["selected_epoch"]}段</td><td>{values}</td><td>{100*result["final_training_head_fit"]["accuracy"]:.2f}%</td><td>{result["final_training_gradient_max_abs"]:.3g}</td></tr>')
    (out/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>图a—j改版排程</title><style>body{font:16px/1.7 sans-serif;max-width:1350px;margin:30px auto;padding:0 16px}table{border-collapse:collapse;width:100%}td,th{padding:9px;text-align:left;border-bottom:1px solid #ddd}.note{background:#fff1d5;padding:14px}</style><h1>图a—j改版排程</h1><p class="note">所有新LG均使用同一末端单层Linear；既往TEST已经参与开发，以下均为开发结果，不是独立TEST泛化。A/B原D2NN完整60轮结果保护，不重训来压低参照。</p><table><tr><th>图</th><th>安排</th></tr><tr><td>a</td><td>三组LG补Linear已完成保护；D2NN三层无OEO/Linear。Kather新两层LG作为b两层的唯一权重来源。</td></tr><tr><td>b</td><td>仅LG四/六层末E/G+Linear有限微调；冻结入口路由/早期相位。D2NN3/5/7逐层OEO无Linear，不再训练。</td></tr><tr><td>d</td><td>按人工要求暂缓，后续需要新Linear与同配置点对齐。</td></tr><tr><td>f</td><td>保留原RGB/SAR十分类双域子集和空间划分，从头LG、D2NN各一组；同OEO，无前端，LG有Linear、D2NN无Linear。</td></tr><tr><td>g</td><td>只读核验，保护历史终身学习结果；同源EuroSAT双模态联合输入，不等于f单模态双域。</td></tr><tr><td>j</td><td>按正式最终源码继续只读核验，不改现场硬件/旧结果。</td></tr></table><h2>本轮四个并行任务</h2><table><tr><th>任务</th><th>资源</th><th>实际状态</th><th>轮数</th><th>TRAIN / VAL / 原TEST开发</th><th>按近期完整epoch估算剩余</th></tr>'''+''.join(rows)+'''</table><h2>图f与g的雷达任务</h2><table><tr><th>项目</th><th>图f原协议与此次重跑</th><th>图g实际正式协议</th></tr><tr><td>预测目标</td><td colspan="2">相同十类地物标签：AnnualCrop、Forest、HerbaceousVegetation、Highway、Industrial、Pasture、PermanentCrop、Residential、River、SeaLake。不是RGB/SAR二分类或配对匹配。</td></tr><tr><td>一次输入</td><td>一张RGB或一张SAR，三通道固定拼图；两个域分别评价</td><td>同地块RGB和SAR共同输入；RGB含共享冻结视觉特征条，SAR占第四tile</td></tr><tr><td>规模</td><td>TRAIN6000、VAL2000、原空间TEST2000张，两域均衡</td><td>TRAIN15998、VAL5465、TEST5429对，同地点两模态绑定</td></tr><tr><td>目的/指标</td><td>同一模型处理两个域；分别报告普通准确率、宏召回、路由功率</td><td>EuroSAT作为四任务终身学习第一任务；正式主指标宏平均召回</td></tr><tr><td>已核验</td><td colspan="2">F三个划分的全部像素/标签逐一对应G完整数据的同划分子集；保留原空间组，无临时筛样本。</td></tr></table><p>双方OEO均为原intensity_softsign数学形式。当前a/b/f每个主相位层含末层均OEO；历史g末层省略显式OEO回编码，放置差异保留，不能把旧g指标说成已统一重跑。原数据授权依现存MIT记录，不标为CC BY4.0。</p><p>所有运行保留profile、源commit/SHA、best/last、逐轮TRAIN/VAL、锁权重记录、逐样本预测与实际Top2功率/capture。D2NN没有末端Linear，因此电子读出容量并不匹配，不能称等电子预算结构因果消融。</p></html>'''
    completed_note='''<h2>完整收敛与目标检查</h2><p>F首轮LG训练/验证约42%，D2NN训练44%/验证41%；低性能不能简单归因于严重过拟合。B4联合候选低于父，保留78.72%；B6当前83.11%，仍未比固定D7高15pp。旧冻结CNN前端的79.50/74.90不能填入当前无前端版本，全部候选保留。</p><img src="convergence.svg" style="width:100%"><p>黑线VAL，蓝线干净TRAIN；红星仅为锁权重后原TEST开发结果。TRAIN菱形为选中权重的完整TRAIN重评。不同候选不拼为连续训练。</p>'''
    page=page.replace('</html>',completed_note+'</html>')
    convex_note='<h2>冻结光学后的有界单Linear拟合</h2><p>只优化原Linear，光学相位逐元素不变。TRAIN交叉熵+.001/2倍相对父头的平方范数，最多150个L-BFGS内部迭代；每5步导回float32 canonical模型，VAL选模并保留父。末梯度仍非零，不能宣称已经求得全局最优或将剩余差距证明为光学容量上限。</p><table><tr><th>任务</th><th>数值预算/选段</th><th>选中TRAIN / VAL / 原TEST开发</th><th>最终迭代TRAIN</th><th>最终梯度最大值</th></tr>'+''.join(convex_rows)+'</table><p>B4达到83.51%，比固定D5高16.22pp；B6为84.18%，比固定D7高12.10pp，尚未15pp。F为42.40%，改善有限，低分原因仍在定位。</p>'
    page=page.replace('</html>',convex_note+'</html>')
    gram_rows=[];report['TRAIN_Gram_coordinate_candidates']={}
    for name in ['f_lg_convex','b_lg6_convex']:
        result=read(task/'runs/simulation/figure_Gram_readout_s17_20261031/worker0'/name/'result.json');report['TRAIN_Gram_coordinate_candidates'][name]=result
        if result:
            vals=' / '.join(f"{100*result[k]['accuracy']:.2f}%" for k in ['train','val','original_test_dev'])
            gram_rows.append(f'<tr><td>{name}</td><td>{vals}</td><td>{result["final_training_gradient_max_abs"]:.3g}</td></tr>')
    note='<h2>相同目标、相同预算的TRAIN Gram数值坐标对照</h2><p>与上一轮使用相同原始父、相同原W锚定惩罚和150内部迭代上限；只用TRAIN求可逆坐标，导出时吸收到原单Linear。不存在推理白化层或新电子参数。</p><table><tr><th>任务</th><th>TRAIN / VAL / 原TEST开发</th><th>最终原W梯度最大值</th></tr>'+''.join(gram_rows)+'</table><p>B6为652/752=86.70%，相对固定D7高14.63pp，仍差3个正确达到15pp，不能四舍五入声称达标。F为905/2000=45.25%，原训练约56%而验证44.65%，仍需定位光学表征与泛化，不能将正则头最优等同于模型容量上限。</p>'
    page=page.replace('</html>',note+'</html>')
    (out/'results.html').write_text(page,encoding='utf-8');print([(n,r['status']['state']) for n,r in records.items()])


if __name__=='__main__':main()
