"""Verify the preplanned routing interventions and summarize final test evidence."""
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'.codex_plot_deps'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
OUT=ROOT/'completed';DATA=OUT/'archive/moe_root/results'
perf=json.loads((DATA/'PERFORMANCE.json').read_text(encoding='utf-8'))
moe=perf['models']['moe'];verified={};routing={}
files={'uniform':'uniform','isolated':'isolated','restore_A_initial_phases':'reset_A','restore_B_initial_phases':'reset_B'}
for domain in ('A','B'):
    with np.load(DATA/'moe'/f'{domain}_test.npz',allow_pickle=False) as z:
        labels=z['labels'].copy();indices=z['indices'].copy();baseline=z['predictions'].copy()
        power=z['route_weights'].astype(np.float64)**2
        assert np.allclose(power.sum(1),1,atol=1e-6)
        assert np.allclose(power.mean(0),moe['tests'][domain]['routes']['mean_power'],atol=1e-6)
        group_a=power[:,:2].sum(1)
        routing[domain]={'mean_power':power.mean(0).tolist(),'group_A_mean':float(group_a.mean()),'group_A_std':float(group_a.std()),'group_A_quantiles':np.quantile(group_a,[0,.05,.5,.95,1]).tolist(),'fraction_group_A_above_half':float((group_a>.5).mean()),'samples':len(labels)}
    for name,prefix in files.items():
        report=moe['diagnostics'][name][domain]
        with np.load(DATA/'moe'/f'diagnostic_{prefix}_{domain}.npz',allow_pickle=False) as z:
            assert np.array_equal(z['labels'],labels) and np.array_equal(z['indices'],indices)
            assert np.isfinite(z['logits']).all()
            pred=z['logits'].argmax(1);assert np.array_equal(pred,z['predictions'])
            cm=np.bincount(10*labels+pred,minlength=100).reshape(10,10)
            accuracy=float((labels==pred).mean())
            f1=float((2*np.diag(cm)/np.maximum(1,cm.sum(0)+cm.sum(1))).mean())
            assert accuracy==report['accuracy'] and f1==report['macro_f1'] and cm.tolist()==report['confusion_matrix']
            assert np.allclose((z['route_weights']**2).sum(1),1,atol=1e-6)
            verified.setdefault(name,{})[domain]={'accuracy':accuracy,'macro_f1':f1,'prediction_change_fraction':float((pred!=baseline).mean()),'accuracy_delta_pp':100*(accuracy-moe['tests'][domain]['accuracy'])}
result={'passed':True,'routing_test':routing,'interventions':verified,'test_based_tuning':False}
(OUT/'ROUTING_TEST_VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11})
fig,axes=plt.subplots(1,2,figsize=(12.5,4.7),gridspec_kw={'width_ratios':[1.25,1]})
order=['moe','AB','A_only','B_only'];names=['光路由 MoE A+B','D2NN A+B','D2NN A-only','D2NN B-only']
x=np.arange(4);width=.36
for k,(domain,color) in enumerate([('A','#2764aa'),('B','#db873b')]):
    values=[100*perf['models'][m]['tests'][domain]['accuracy'] for m in order]
    bars=axes[0].bar(x+(k-.5)*width,values,width,label=domain+(' 光学' if domain=='A' else ' SAR'),color=color)
    axes[0].bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
axes[0].set(xticks=x,xticklabels=names,ylim=(0,105),ylabel='测试准确率（%）',title='四模型固定检查点的双域测试')
axes[0].tick_params(axis='x',labelsize=9)
axes[0].legend(loc='upper right',frameon=False);axes[0].grid(axis='y',alpha=.18);axes[0].set_axisbelow(True)
arr=np.array([routing[d]['mean_power'] for d in ('A','B')])*100
im=axes[1].imshow(arr,vmin=0,vmax=100,cmap='Blues',aspect='auto')
for i in range(2):
    for j in range(4):axes[1].text(j,i,f'{arr[i,j]:.2f}%',ha='center',va='center',color='white' if arr[i,j]>60 else '#152334')
axes[1].set(xticks=range(4),xticklabels=['E0（A）','E1（A）','E2（B）','E3（B）'],yticks=[0,1],yticklabels=['A 测试','B 测试'],title='MoE 平均输入功率份额')
fig.colorbar(im,ax=axes[1],label='功率份额（%）',fraction=.05,pad=.04)
fig.suptitle(f"EuroSAT 光学/SAR 十分类 · seed 42 · A/B 各测试 {routing['A']['samples']:,} 张",fontsize=13)
fig.tight_layout();fig.savefig(OUT/'EuroSAT性能与路由.png',dpi=200);plt.close(fig)

names_diag={'uniform':'均匀路由','isolated':'已知域隔离（提供域标签）','restore_A_initial_phases':'恢复 A 专家初始相位','restore_B_initial_phases':'恢复 B 专家初始相位'}
lines=['## 最终测试上的路由与专家贡献诊断','','以下是预先安排的固定模型干预，不据此改动模型或选择检查点。所有诊断已从保存的逐样本 logits 独立复算，测试样本、标签和总分配功率均已核对。','','| 路由 / 相位设置 | A 准确率 | B 准确率 | 两域平均 |','|---|---:|---:|---:|',f"| 自动路由（正式结果） | {moe['tests']['A']['accuracy']*100:.2f}% | {moe['tests']['B']['accuracy']*100:.2f}% | {moe['mean_accuracy']*100:.2f}% |"]
for name,item in verified.items():
    a=item['A']['accuracy']*100;b=item['B']['accuracy']*100
    lines.append(f'| {names_diag[name]} | {a:.2f}% | {b:.2f}% | {(a+b)/2:.2f}% |')

lines += ['', 'A 组 E0/E1 在光学 A 测试集的平均功率为 '+f"{routing['A']['group_A_mean']*100:.2f}%"+'，在 SAR B 测试集为 '+f"{routing['B']['group_A_mean']*100:.2f}%"+'。', '', '自动路由相对均匀路由的双域平均准确率差为 '+f"{100*(moe['mean_accuracy']-sum(v['accuracy'] for v in verified['uniform'].values())/2):+.3f}"+' 个百分点。专家分工需要结合逐域相位恢复效果解释；功率偏置本身不构成性能贡献证据。', '', '原始数据共27,000对，按预设输入完整性规则成对排除108对SAR固定编码后全零的图像，保留26,892对。地理分组及配对绑定规则已于训练前封存。', '', '单seed42结果。归档包含源代码、固定检查点哈希、封存证明、日志、逐样本预测和图表；大型选定权重保留在服务器。']
report=OUT/'EuroSAT性能报告.md'
text=report.read_text(encoding='utf-8').split('\n\n## 最终测试上的路由与专家贡献诊断')[0]
report.write_text(text+'\n\n'+'\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
