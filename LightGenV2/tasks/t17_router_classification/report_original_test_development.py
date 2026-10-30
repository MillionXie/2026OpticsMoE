"""Summarize explicit original-TEST development runs; no training or model selection."""
from pathlib import Path
import json,html,datetime
repo=Path(__file__).resolve().parents[3]
task=repo/'LightGenV2/tasks/t17_router_classification';runs=task/'runs/simulation';out=task/'reports/panel_a_original_test_development_20261031';out.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text()) if p.exists() else None
cases=[]
for d,name in [('kather2016','Kather2016（8类）'),('bloodmnist128','BloodMNIST-128（8类）'),('crc9','CRC-VAL-HE-7K（9类）')]:
 dev=read(runs/'panel_a_original_test_dev_s17_20261031'/d/'result.json')
 old=dev['selected_LG'];ref=dev['selected_D2NN_development_reference'];records=[]
 for run,suffix,label in [('panel_a_original_test_teacher_kd_s17_20261031','student','D4蒸馏'),('panel_a_original_test_kd_identity_s17_20261031','','无增强保守蒸馏'),
                         ('panel_a_boundary_development_s17_20261031','frozen_router_boundary','冻结光路由＋边界损失'),
                         ('panel_a_boundary_development_s17_20261031','diffuse_start_boundary','随机主相位＋冻结光路由＋边界损失'),
                         ('panel_a_boundary_continuation_s17_20261031','frozen_router_boundary','边界目标受限接续'),
                         ('panel_a_boundary_joint_s17_20261031','fixed_router_control','成熟主干固定路由控制'),
                         ('panel_a_boundary_joint_s17_20261031','gentle_joint_router','成熟主干温和联合路由'),
                         ('panel_a_training_error_weighted_s17_20261031','training_error_weighted','原TRAIN弱类别有界加权'),
                         ('panel_a_phase_gauge_s17_20261031','phase_gauge_recenter','整体相位坐标调整'),
                         ('panel_a_wrapped_phase_s17_20261031','wrapped_phase_adam','周期相位坐标Adam'),
                         ('panel_a_dense_router_supervision_s17_20261031','dense_pair_aux_002','光router类别对辅助KL .02'),
                         ('panel_a_dense_router_supervision_s17_20261031','dense_pair_aux_010','光router类别对辅助KL .1')]:
  p=runs/run/d/suffix;q=read(p/'result.json');s=read(p/'status.json')
  if q:records.append(dict(run_id=run+'/'+d+('/'+suffix if suffix else ''),state='complete',label=label,result=q))
  elif p.parent.exists():records.append(dict(run_id=run+'/'+d+('/'+suffix if suffix else ''),state=s or {'state':'not_started_or_loading'},label=label))
 candidates=[(old['original_test_dev'],f"panel_a_original_test_dev_s17_20261031/{d}/candidates/{old['candidate']}",None)]
 candidates.extend((r['result']['original_test_dev'],r['run_id'],r['result']) for r in records if 'result' in r)
 score,source,result=max(candidates,key=lambda z:(z[0]['accuracy'],z[0]['balanced_accuracy'],-z[0]['cross_entropy']))
 # Prefer the completed student's result for TRAIN/VAL if scores are the same as parent.
 if result is None:
  for r in records:
   if 'result' in r and abs(r['result']['original_test_dev']['accuracy']-score['accuracy'])<1e-6:result=r['result'];break
 teacher=read(runs/'panel_a_original_test_teacher_kd_s17_20261031'/d/'teacher/result.json')
 audit=read(runs/'panel_a_routing_pair_diagnostic_s17_20261031'/d/'result.json')
 row=dict(dataset=d,name=name,seed=17,independent_test_claim=False,D2NN_development_reference=ref,LG_selected_development_metrics=score,LG_selected_source=source,LG_full_selected_result=result,development_trials=records,teacher=teacher,routing_diagnostic=audit,gain_pp=100*(score['accuracy']-ref['original_test_dev']['accuracy']))
 row['target_20pp_reached']=row['gain_pp']>=20
 row['all_optical_exploration_epoch_budget_completed']=90+sum(z['result']['completed_epochs'] if 'result' in z else z['state'].get('epoch',0) for z in records)
 row['budget_note']='Each student run total_optical_executed_epochs counts its own branch; this aggregate also counts the competing KD branch. Teacher head epochs and ImageNet pretraining are additional.'
 cases.append(row)
report=dict(date='2026-10-31',generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT',independent_test_claim=False,all_rows_single_seed_not_repeated_means=True,architecture=dict(LG='Optical router224 + 4x224 experts, Top2, gap30, global478; E -> OEO -> G -> OEO -> fixed CCD; 479364 phase parameters; no learned electronic frontend/readout',D2NN='Three continuous478 phase layers, no intermediate OEO, fixed same CCD regions; 685452 phase parameters; no learned electronic frontend/readout',physics='532nm,17um pitch,.1m/layer,518 canvas',input='RGB each112; [R,G;B,0]224, unit power. LG copies224 to experts; D2NN bilinear expands to478. Fourth quadrant remains zero.'),cases=cases)
(out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
pct=lambda v:f'{100*v:.2f}%'
rows=''
for c in cases:
 r=c['LG_full_selected_result'];train=pct(r['train']['accuracy']) if r else '—';val=pct(r['val']['accuracy']) if r else '—'
 rows+=f"<tr><td>{c['name']}</td><td>{pct(c['D2NN_development_reference']['original_test_dev']['accuracy'])}</td><td>{train}</td><td>{val}</td><td><b>{pct(c['LG_selected_development_metrics']['accuracy'])}</b></td><td>{c['gain_pp']:+.2f} pp</td></tr>"
trials=''
for c in cases:
 for r in c['development_trials']:
  q=r.get('result');status=(f"完成{q['completed_epochs']}轮；选中{q['selected_kind']} / 第{q['selected_epoch']}轮；开发准确率{pct(q['original_test_dev']['accuracy'])}" if q else f"{html.escape(r['state'].get('state','待启动'))}；已完成{r['state'].get('epoch',0)}轮。本表只纳入已完成运行的最终收据。")
  trials+=f"<tr><td>{c['name']}</td><td>{r['label']}</td><td>{status}</td></tr>"
text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>图A统一架构：原TEST开发训练记录</title><style>body{{font:16px/1.65 'Microsoft YaHei',sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#172635}}h1{{font-size:26px}}h2{{font-size:20px;margin-top:32px}}table{{border-collapse:collapse;width:100%;margin:18px 0}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #dae1e7}}th{{background:#eaf1f7}}.notice{{padding:18px;background:#fff1d6;border-left:5px solid #cf8d13}}small{{color:#5c6974}}code{{font-size:13px}}</style><h1>图A统一架构：原TEST开发训练记录</h1><p class="notice">原TEST已按人工授权用于权重选择、错例分析、教师训练及学生蒸馏输入。下表原TEST分数为开发指标，不能作为独立测试泛化性能。全部为seed17单次实验，无均值/标准差。</p><h2>当前已完成结果</h2><table><tr><th>任务</th><th>D2NN原TEST</th><th>LG TRAIN</th><th>LG VAL诊断</th><th>LG原TEST开发</th><th>差值</th></tr>{rows}</table><p>差值为百分点；20个百分点是优化目标，尚未达到的任务不取整为达标。D2NN同时比较best/last，未降低参照。LG使用更多训练预算及教师数据，不能将差值全部归因于等预算架构优势。</p><h2>训练对照</h2><table><tr><th>任务</th><th>方法</th><th>状态</th></tr>{trials}</table><p>选中parent / 第0轮表示保留蒸馏前的相位，不能称为蒸馏提升。D4与无增强对照还改变了batch、路由学习率、辅助损失及KD权重，不作单因素因果归因。</p><p>每条蒸馏分支从同一开发父权重起步，不能将两条分支相加为所选权重的连续训练轮数。总探索预算另计基础60轮＋VAL优化30轮＋各蒸馏候选实际轮数；两条各20轮完成后为130轮，另加教师头训练和ImageNet预训练。新边界候选从保守蒸馏best起步，其额外轮数继续计入JSON的总探索预算。</p><h2>统一推理结构</h2><p>{report['architecture']['LG']}。</p><p>{report['architecture']['D2NN']}。</p><p>两次OEO：光强→按均值归一→I/(1+I)→L2振幅归一→实振幅重编码；末端读出位于第二次OEO之后。它是固定电子非线性，不是无电子处理的被动光学，也不是Kerr材料非线性。</p><p>固定输入：[R,G;B,0]，每通道112²、拼接224²、单位功率。LG每专家接收224²；D2NN插值到478²。固定CCD区域覆盖有效478²面，8类2×4、9类3×3。无Linear/CNN分类头。波长532nm、像素17μm、传播10cm、画布518²。</p><h2>边界训练及相位坐标候选收敛</h2><p>各子图按独立候选的额外轮数绘制，未将竞争分支拼成连续训练。绿色菱形为获选权重一次TRAIN评估，蓝线为VAL诊断，橙线为已暴露原TEST开发分数；未完成候选只显示当前曲线，不提前计入结果。</p><img src='boundary_curves.svg' style='width:100%' alt='Per-branch boundary development curves'><h2>早期蒸馏训练曲线</h2><p>曲线分别显示VAL诊断和已暴露的原TEST开发分数，不将训练目标损失当作训练准确率。此图只展示前两轮蒸馏，各任务两图使用一致纵轴；新边界候选的逐轮指标另保存在各自history.json，最终结果均列入上表。</p><img src='distillation_curves.svg' style='width:100%' alt='Development training curves'><h2>诊断边界</h2><p>已测六种固定等功率专家对；均未超过原动态路由。真实标签选择专家对的结果仅为不可部署的反事实诊断，不作为LG准确率，也不是可变功率路由的性能上界。路由偏置仍存在，但不能据此认定强制均衡必然提升。</p><p>教师是冻结ImageNet ResNet18＋可训练MLP，在TRAIN和原TEST增强视图上训练；学生硬标签CE只用原TRAIN，教师KL使用TRAIN和原TEST。教师准确率不能当作LG准确率。原TEST错例不自动删样本、改标签。</p><p>完整指标、来源权重SHA、各轮预算、逐样本路由与独立运行身份保存在对应run。数据级重复检查不能替代患者独立性验证。原VAL选模结果保留。</p><small>生成时间UTC：{report['generated_utc']}；机器可读汇总：results.json；权重校验：checkpoint_transfer.json。</small></html>'''
(out/'results.html').write_text(text,encoding='utf-8')
print([(c['dataset'],round(c['LG_selected_development_metrics']['accuracy']*100,2),round(c['gain_pp'],2)) for c in cases])
