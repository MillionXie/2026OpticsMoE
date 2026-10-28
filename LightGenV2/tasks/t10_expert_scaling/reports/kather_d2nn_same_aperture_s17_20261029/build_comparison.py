"""Paired seed17/60-epoch aperture control; preserve the historical mixed table."""
from pathlib import Path
import csv,hashlib,json

OUT=Path(__file__).resolve().parent
REPORTS=OUT.parent
BASE=REPORTS/'plotting/kather_expert_count_20260921/three_seed_test_per_seed.csv'
SUMMARY=OUT/'final_summary_20261029.json'
GROUPS={4:[1,2,3,4],16:[1,4,8,16],25:[1,6,12,25],49:[1,12,24,49]}
COLORS={4:'#4477AA',16:'#EE7733',25:'#228833',49:'#AA3377'}

def write(path,value):
    path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

def data():
    report=json.loads(SUMMARY.read_text(encoding='utf-8-sig'))
    assert report['status']=='complete' and report['validation']['status']=='passed'
    assert report['seed']==17 and report['training_contract']['epochs']==60
    baseline={r['experts']:r for r in report['rows']}
    raw=list(csv.DictReader(BASE.open(encoding='utf-8-sig')))
    rows=[];paired=[]
    for n,ks in GROUPS.items():
        b=baseline[n];g=b['phase_side'];assert b['parameters']==4*g*g
        for k in ks:
            r=next(v for v in raw if v['architecture']=='MoE' and int(v['experts'])==n and int(v['top_k'])==k and int(v['seed'])==17)
            correct=round(float(r['test_accuracy'])*752)
            assert abs(float(r['test_accuracy'])-correct/752)<1e-7
            rows.append(dict(N=n,k=k,ratio=k/n,LG_correct=correct,LG_accuracy_percent=100*correct/752,LG_checkpoint_sha256=r['checkpoint_sha256'],LG_original_run=r['run_name'],Dense_OEO_correct=b['metrics']['test']['correct'],Dense_OEO_accuracy_percent=b['metrics']['test']['accuracy_percent'],G=g,LG_parameters=b['LG_parameters_including_router'],Dense_OEO_parameters=b['parameters']))
        full=rows[-1]
        paired.append(dict(**full,difference_pp=full['LG_accuracy_percent']-full['Dense_OEO_accuracy_percent']))
    return dict(seed=17,epochs=60,TEST_per_model=752,statistics='Single seed; no SD or significance claim.',scope='Original60-epoch LG vs new same-global-aperture Dense-OEO. Post-hoc architecture control; unequal RGB source bandwidth remains.',color_encoding='Same global phase aperture G x G, not equal active pixels per layer or total parameters.',rows=rows,full_k_pairs=paired,sources=[dict(path=p.relative_to(REPORTS).as_posix(),LF_sha256=hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()) for p in [SUMMARY,BASE]])

def documents(d):
    lines=['# 同全局孔径的结构对照：seed17，60轮','',
        '新基线命名为**无路由稠密衍射基线（Dense-OEO）**。四层均为连续相位面，并保留逐层 OEO 非线性；它同时移除路由和分块专家结构，属于结构对照，不能称仅移除 Router 的单因素消融，也不宜简称无源线性 D2NN。','',
        '| N（LG容量） | 全局面 / Dense-OEO每层面 | LG总参数（含Router） | Dense-OEO总参数 | Dense-OEO TEST正确数 | 准确率（%） | 最佳轮次 |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    report=json.loads(SUMMARY.read_text(encoding='utf-8-sig'))
    for r in report['rows']:
        lines.append(f"| {r['experts']} | {r['phase_side']}×{r['phase_side']} | {r['LG_parameters_including_router']:,} | {r['parameters']:,} | {r['metrics']['test']['correct']}/752 | {r['metrics']['test']['accuracy_percent']:.2f} | {r['best_epoch']} |")
    lines+=['','**单seed17，不报告跨seed均值或标准差。** 原始60轮LG的同seed全激活结果：','',
        '| N | LG全激活正确数 | LG准确率（%） | Dense-OEO准确率（%） | 差值（pp） |','|---:|---:|---:|---:|---:|']
    for r in d['full_k_pairs']:lines.append(f"| {r['N']} | {r['LG_correct']}/752 | {r['LG_accuracy_percent']:.2f} | {r['Dense_OEO_accuracy_percent']:.2f} | {r['difference_pp']:+.2f} |")
    lines+=['','![同孔径seed17结构对照](same_aperture_seed17.png)','','## 中文图注（可直接使用）','',
        '**图：同全局孔径下 LightGen 与无路由稠密衍射基线的结构对照。**（a）LightGen 在不同专家容量 N 下，测试准确率随激活专家比例 k/N 的变化；连线仅引导视线。（b）相同全局孔径下，全激活 LightGen（实心圆）与 Dense-OEO（空心方形）的准确率。颜色表示配对的全局相位面尺寸 G×G：N=4、16、25、49 分别对应 G=478、986、1240、1748。Dense-OEO 无专家路由，因此不定义 k/N。两种结构均使用4个特征衍射层，并在每层后保留相同的光—电—光（OEO）归一化与非线性处理。LightGen 包含2个分块专家层、2个全局层及额外的224×224路由相位面，总可训练相位参数量为 2N·224²+2G²+224²；Dense-OEO 的4个连续相位面均为 G×G，总参数量为4G²。两者参数量依次为0.909/0.914、3.600/3.889、5.634/6.150、11.078/12.222百万（LightGen/Dense-OEO）。图中采用seed17、60轮训练及验证集macro-NLL最低的EMA模型，各在锁定后评估752张测试图像；不表示三种子均值或统计显著性。','',
        '**比较边界：**同色仅表示全局孔径尺寸相同，不表示每层可训练像素数或总参数相同。LG专家层存在30像素间隙；Dense-OEO使用更大的RGB编码采样网格，源信息带宽尚未严格匹配。本重跑是在已观察历史TEST后开展的结构开发对照，不是新的确认性试验。此图使用原始60轮LG，不混入90轮跟进或TEST参与梯度的full-k拟合结果。','',
        '## 训练、迁移与核验','',
        '四模型均完成60轮、每轮219次更新，末Adam步数13140；每模型已完成209760个TRAIN梯度输入，TEST梯度输入为0。N16从A100上的完整第28轮继承live/EMA/Adam及原best，迁至GPU0再完成32轮；RNG按原入口重设，不称逐位恢复。被中断的部分轮不计为完整轮，其额外梯度输入数和准确计算耗时未知。N4在A100自然完成；用户收回后本任务不再使用A100，测试改用GPU4/0/1/3。其他任务未操作。','',
        '实际训练来源为6d0ca6b（N4/N25/N49）及657e275（迁移管理与N16续训）；服务器host HEAD ae9e5c0仅是主目录身份。旧metadata/evaluator中设计模板的“provisional 6层”不覆盖实际arguments.layers=4及CPU实读4张相位面的事实。','',
        '53份小产物manifest、11份生产依赖SHA、CPU实读best/last/live/EMA/Adam、60轮历史及余弦LR、VAL最小NLL选模、四模型先锁时间、752张VAL/TEST的真实IDs/labels/概率/argmax/混淆矩阵/NLL，以及最终TRAIN混淆矩阵均已独立核验。TRAIN逐样本概率未导出；不声称有其逐样本概率复算。父进程及全部TRAIN/TEST子进程已退出，无本任务计算进程，PT留在服务器。','',
        '[正式报告](final_summary_20261029.json)；[作图精确数据](comparison_data.json)；[单seed明细表](paired_seed17.csv)；[生成源](build_comparison.py)。','',
        '[独立核验记录](final_independent_audit.json)与[资源收据](final_snapshot/resource_release.json)为本地manifest产物，不随源码Git提交。原始数据、所有best/last及旧三seed D2NN结果保留。','',
        '[此前最终混预算三seed表](../plotting/kather_latest_five_20261028/final_results.md)继续保留；其中*为90轮开发跟进，†为TEST参与教师/学生训练的暴露拟合，不能与本单seed60轮基线混称等预算独立泛化比较。','']
    (OUT/'README.md').write_bytes(('\n'.join(lines)).encode('utf-8'))
    with (OUT/'paired_seed17.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(d['rows'][0]));writer.writeheader();writer.writerows(d['rows'])

def render(d):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','pdf.fonttype':42})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(11.8,4.7),sharey=True)
    for i,(n,ks) in enumerate(GROUPS.items()):
        r=[v for v in d['rows'] if v['N']==n]; color=COLORS[n]
        ax.plot([v['ratio'] for v in r],[v['LG_accuracy_percent'] for v in r],'-o',color=color,lw=1.8,ms=5)
        pair=d['full_k_pairs'][i]
        bx.plot([i-.10,i+.10],[pair['LG_accuracy_percent'],pair['Dense_OEO_accuracy_percent']],color=color,lw=1.3,alpha=.55)
        bx.scatter(i-.10,pair['LG_accuracy_percent'],color=color,s=55,zorder=3)
        bx.scatter(i+.10,pair['Dense_OEO_accuracy_percent'],facecolor='white',edgecolor=color,marker='s',s=55,lw=1.7,zorder=3)
        bx.annotate(f"{pair['LG_accuracy_percent']:.2f}",(i-.10,pair['LG_accuracy_percent']),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
        bx.annotate(f"{pair['Dense_OEO_accuracy_percent']:.2f}",(i+.10,pair['Dense_OEO_accuracy_percent']),xytext=(0,-17),textcoords='offset points',ha='center',fontsize=9)
    ax.set_title('(a) LightGen: expert activation',loc='left',fontsize=12)
    bx.set_title('(b) Paired global-aperture control',loc='left',fontsize=12)
    ax.set_xlabel('Activated expert ratio k/N');ax.set_ylabel('Test accuracy (%)')
    ax.set_xlim(-.025,1.04);ax.set_xticks([0,.25,.5,.75,1])
    bx.set_xticks(range(4),[f"N={n}\n{d['full_k_pairs'][i]['G']}×{d['full_k_pairs'][i]['G']}" for i,n in enumerate(GROUPS)])
    bx.set_xlabel('Paired global phase aperture (pixels)');bx.set_xlim(-.5,3.5)
    for a in [ax,bx]:
        a.set_ylim(43,86);a.set_yticks([45,50,55,60,65,70,75,80,85]);a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.16)
    colorlegend=[Line2D([0],[0],color=COLORS[n],lw=2,label=f"N={n}, G={d['full_k_pairs'][i]['G']}") for i,n in enumerate(GROUPS)]
    fig.legend(handles=colorlegend,loc='lower left',bbox_to_anchor=(.055,.055),ncol=4,frameon=False,fontsize=9)
    bx.legend(handles=[Line2D([0],[0],marker='o',ls='',color='#444',label='LightGen (k=N)'),Line2D([0],[0],marker='s',ls='',mfc='white',mec='#444',label='Dense-OEO')],loc='upper left',frameon=False,fontsize=9)
    fig.text(.06,.017,'Seed17 | 60 epochs | 752 TEST images/model | Single-seed accuracy; no SD | Source bandwidth remains unmatched',fontsize=9,color='#555')
    fig.subplots_adjust(left=.065,right=.985,top=.91,bottom=.25,wspace=.12)
    for ext in ['png','svg','pdf']:fig.savefig(OUT/f'same_aperture_seed17.{ext}',dpi=220,facecolor='white')
    plt.close(fig)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');a=ap.parse_args()
    d=data()
    if not a.check:write(OUT/'comparison_data.json',d);documents(d);render(d)
    print(json.dumps({'seed':17,'pairs':d['full_k_pairs']},ensure_ascii=True))
