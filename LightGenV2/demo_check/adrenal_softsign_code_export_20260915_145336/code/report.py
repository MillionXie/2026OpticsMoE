"""All four families, no outcome filtering; primary fixed threshold plus calibrated views."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from experiments import ROOT,EXP,formal_variants,run_dir,parameters
from metrics import from_csv
from calibration import choose_thresholds,full_metrics
from run_experiment import read,save,sha

METRICS=[('auroc','AUROC',1.,'AUROC',True),('accuracy','准确率',100.,'准确率（%）',True),
 ('balanced_accuracy','平衡准确率',100.,'平衡准确率（%）',True),('average_precision','平均精确率 AP',1.,'AP',True),
 ('macro_f1','宏平均 F1',1.,'Macro-F1',True),('positive_recall','病变召回率',100.,'召回率（%）',True),
 ('specificity','特异度',100.,'特异度（%）',True),('detector_plane_mse','探测面 MSE',1.,'MSE（训练目标，含 ×100）',False),
 ('mse','分类分数 MSE',1.,'MSE（类别分数与 one-hot 标签）',False)]
KEYS=[m[0] for m in METRICS];COLORS={'moe':'#167CB5','d2nn':'#DD7724'}
LABELS={'softplus2':'Softplus β=2','relu_tanh':'正半轴 Tanh','relu_softsign':'正半轴 Softsign'}

def csvwrite(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def aggregate(rows):
    return {k+s:float(fn([r[k] for r in rows])) for k in KEYS for s,fn in [('_mean',np.mean),('_sd',lambda x:np.std(x,ddof=1))]}

def make_report():
    out=ROOT/'reports';out.mkdir(exist_ok=True)
    selection=read(ROOT/'protocol/activation_selection.json');act=selection['selected'];formal=formal_variants(act)
    lock=read(ROOT/'protocol/final_lock.json');rows=[];ids_ref=None;orders={};hashes={}
    for seed in EXP['seeds']:
        for v in formal:
            dest=run_dir(v,seed)
            assert sha(dest/'best.pt')==lock['checkpoints'][(dest/'best.pt').relative_to(ROOT).as_posix()]
            saved=read(dest/'test_metrics.json');fixed,preds,y,p=from_csv(dest/'test_predictions.csv')
            for k in KEYS:
                if k!='detector_plane_mse':assert abs(saved[k]-fixed[k])<1e-10
            ids=[(r['sample_id'],r['label_true']) for r in preds]
            if ids_ref is None:ids_ref=ids
            else:assert ids==ids_ref
            assert len(ids)==298
            done=read(dest/'completed.json');assert done['epochs']==50 and done['updates']==7450
            assert all(done['changed_phase_planes'].values())
            if seed in orders:assert orders[seed]==done['order_sha256']
            else:orders[seed]=done['order_sha256']
            if v['oeo'] or v['architecture']=='moe':
                if v['architecture']=='moe':
                    assert saved['routing']['hard_counts']==[298]*9
                    assert np.all(np.array(saved['routing']['cycle_expert_nonzero_input_counts'])==298)
            _,_,vy,vp=from_csv(dest/'validation_predictions.csv')
            thresholds=read(dest/'thresholds.json');assert thresholds['policies']==choose_thresholds(vy,vp)['policies']
            calculated=read(dest/'test_threshold_metrics.json')
            for policy,threshold in thresholds['policies'].items():
                m=full_metrics(y,p,threshold['threshold'],saved['detector_plane_mse'])
                for k in KEYS:assert abs(m[k]-calculated[policy][k])<1e-12
                rows.append({'architecture':v['architecture'],'depth':v['depth'],'activation':v['activation'],
                  'oeo':v['oeo'],'seed':seed,'parameters':parameters(v),'policy':policy,
                  'threshold':threshold['threshold'],'selected_epoch':read(dest/'selection.json')['epoch'],
                  **{k:m[k] for k in KEYS},'confusion_matrix':json.dumps(m['confusion_matrix'])})
            for filename in ['test_predictions.csv','test_metrics.json','validation_predictions.csv','thresholds.json']:
                hashes[(dest/filename).relative_to(ROOT).as_posix()]=sha(dest/filename)
    assert len(rows)==180
    summaries=[]
    for policy in ['fixed_0.5','val_accuracy','val_balanced']:
        for v in formal:
            rr=[r for r in rows if r['architecture']==v['architecture'] and r['depth']==v['depth'] and r['activation']==v['activation'] and r['policy']==policy]
            assert len(rr)==5
            summaries.append({'architecture':v['architecture'],'depth':v['depth'],'activation':v['activation'],'oeo':v['oeo'],
                'policy':policy,'n':5,'parameters':parameters(v),**aggregate(rr)})
    lookup={(r['architecture'],r['depth'],r['oeo'],r['policy']):r for r in summaries}
    ordering=[]
    for d in [2,4,6]:
        for k in ['auroc','accuracy','balanced_accuracy']:
            for policy in ['fixed_0.5','val_accuracy','val_balanced']:
                values=[lookup[(a,d,on,policy)][k+'_mean'] for a,on in [('moe',True),('moe',False),('d2nn',True),('d2nn',False)]]
                ordering.append({'depth':d,'metric':k,'policy':policy,'moe_oeo':values[0],'moe':values[1],
                  'd2nn_oeo':values[2],'d2nn':values[3],'strict_mean_order':all(x>y for x,y in zip(values,values[1:]))})
    # Paired seed differences quantify variability rather than claim significance from mean ordering.
    paired=[]
    for d in [2,4,6]:
        for policy in ['fixed_0.5','val_accuracy','val_balanced']:
            for name,left,right in [('moe_oeo_minus_moe',('moe',True),('moe',False)),
                  ('moe_minus_d2nn_oeo',('moe',False),('d2nn',True)),('d2nn_oeo_minus_d2nn',('d2nn',True),('d2nn',False))]:
                diffs=[]
                for seed in EXP['seeds']:
                    def find(key):return next(r for r in rows if (r['architecture'],r['oeo'],r['depth'],r['seed'],r['policy'])==(*key,d,seed,policy))
                    a,b=find(left),find(right);diffs.append({k:a[k]-b[k] for k in KEYS})
                paired.append({'depth':d,'policy':policy,'comparison':name,**aggregate(diffs)})
    csvwrite(out/'per_seed.csv',rows);csvwrite(out/'mean_sd.csv',summaries);csvwrite(out/'ordering.csv',ordering)
    csvwrite(out/'paired_summary.csv',paired);csvwrite(out/'screening_validation.csv',selection['screen_summary'])
    font_manager.fontManager.addfont(str(ROOT/'assets/simhei.ttf'))
    plt.rcParams.update({'font.family':['SimHei','DejaVu Sans'],'axes.unicode_minus':False,'axes.spines.top':False,
      'axes.spines.right':False,'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
    policies={'fixed_0.5':'固定分类阈值 0.5','val_accuracy':'各模型仅用验证集选择准确率阈值','val_balanced':'各模型仅用验证集选择平衡准确率阈值'}
    def draw(ax,metric,policy):
        k,title,factor,ylabel,up=metric
        for on in [True,False]:
            for a in ['moe','d2nn']:
                points=[lookup[(a,d,on,policy)] for d in [2,4,6]]
                yy=np.array([r[k+'_mean'] for r in points])*factor;ee=np.array([r[k+'_sd'] for r in points])*factor
                ee[np.abs(ee)<1e-12]=0
                marker=('o' if a=='moe' else 's') if on else ('^' if a=='moe' else 'D')
                ax.errorbar([2,4,6],yy,yerr=ee,color=COLORS[a],marker=marker,linestyle='-' if on else '--',
                  markerfacecolor=COLORS[a] if on else 'white',capsize=4,linewidth=2,markersize=7)
        ax.set_xticks([2,4,6]);ax.set_xlabel('主光路相位层数');ax.set_ylabel(ylabel)
        ax.set_title(title+(' ↑' if up else ' ↓'),loc='left',fontweight='bold');ax.grid(axis='y',alpha=.22)
    legend=[Line2D([0],[0],color=COLORS[a],marker=('o' if a=='moe' else 's') if on else ('^' if a=='moe' else 'D'),
       markerfacecolor=COLORS[a] if on else 'white',linestyle='-' if on else '--',
       label=('MoE' if a=='moe' else 'D2NN')+' · '+(LABELS[act]+' OEO' if on else '无 OEO')) for on in [True,False] for a in ['moe','d2nn']]
    for policy,title in policies.items():
        fig,axs=plt.subplots(3,3,figsize=(15.8,13.6))
        for ax,metric in zip(axs.flat,METRICS):draw(ax,metric,policy)
        fig.suptitle('AdrenalMNIST 二分类：2 / 4 / 6 层非线性对比',fontsize=21,fontweight='bold',y=.985)
        fig.text(.5,.951,LABELS[act]+'；每点为 5 个配对种子均值，误差条为样本标准差。',ha='center')
        fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.933),ncol=2,frameon=False)
        fig.text(.5,.042,title+'；两种架构同规则。AUROC、AP 和连续 MSE 不受分类阈值影响。',ha='center',fontsize=10)
        fig.text(.5,.020,'全部为本次 RTX 5090 实验；50 轮、验证 AUROC 选模；激活先在 4 层验证筛选。',ha='center',fontsize=10)
        fig.subplots_adjust(left=.075,right=.975,bottom=.10,top=.845,hspace=.48,wspace=.34)
        for ext in ['png','pdf','svg']:fig.savefig(out/f'depth246_{policy}.{ext}',dpi=220)
        plt.close(fig)
    result={'selected_activation':act,'screening':selection,'summary':summaries,'per_seed':rows,
      'ordering':ordering,'paired_summary':paired,'provenance':{'data_sha256':EXP['data_sha256'],
       'formal_runs':60,'unique_training_runs':68,'screening_runs':12,'test_sample_n':298,
       'validation_thresholds_locked_before_test':True,'file_sha256':hashes}}
    save(out/'results.json',result)
    lines=['# 非线性筛选与 2/4/6 层正式测试','',f'入选共享激活：{LABELS[act]}。',
      '68 次独立训练：12 次四层筛选，正式比较60次，其中4次复用完全相同的筛选训练。每次50轮。',
      '筛选只用验证集；先按两种架构的平均AUROC筛入距最优0.01以内候选，再兼顾固定阈值准确率、平衡准确率。',
      '所有测试配置与分类阈值在测试前锁定。测试集此前已用于其他实验评估，不能称为从未使用的外部验证集。',
      '同一激活用于两种架构和全部深度，MSE、优化器、训练轮数、参数匹配、数据和种子保持一致。',
      '平均值满足排序不等于每个种子均满足，也不代表统计显著。固定阈值与验证阈值结果均完整保留。','',
      '|层数|MoE+OEO AUROC|MoE AUROC|D2NN+OEO AUROC|D2NN AUROC|严格均值排序|','|---|---|---|---|---|---|']
    for d in [2,4,6]:
        pts=[lookup[(a,d,on,'fixed_0.5')] for a,on in [('moe',True),('moe',False),('d2nn',True),('d2nn',False)]]
        good=next(r['strict_mean_order'] for r in ordering if r['depth']==d and r['metric']=='auroc' and r['policy']=='fixed_0.5')
        lines.append('|'+str(d)+'|'+'|'.join(f"{p['auroc_mean']:.4f} ± {p['auroc_sd']:.4f}" for p in pts)+'|'+('是' if good else '否')+'|')
    lines+=['','九项全部指标与配对差见 mean_sd.csv、per_seed.csv、paired_summary.csv。',
      '主图 depth246_fixed_0.5.png；兼顾准确率图 depth246_val_accuracy.png；病变识别参考图 depth246_val_balanced.png。',
      '请同时查看病变召回率、平衡准确率与混淆矩阵，识别全部预测为正常的情况。']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
