"""Render the explicitly mixed-budget, single-seed Dense-OEO follow-up."""
from pathlib import Path
import argparse,hashlib,json

OUT=Path(__file__).resolve().parent
SUMMARY=OUT/'final_summary_20261029.json'
COLORS={4:'#4477AA',16:'#EE7733',25:'#228833',49:'#AA3377'}

def data():
    r=json.loads(SUMMARY.read_text(encoding='utf-8'))
    assert r['status']=='complete' and r['validation']['status']=='passed' and r['seed']==17
    assert r['selected']['TEST']['correct']==462 and r['target']['met']
    assert [v['display_TEST_correct'] for v in r['rows']]==[457,462,464,465]
    assert [v['display_epochs'] for v in r['rows']]==[60,90,60,60]
    baseline=OUT.parent/'kather_d2nn_same_aperture_s17_20261029/final_summary_20261029.json'
    assert hashlib.sha256(baseline.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==r['baseline_report']['LF_sha256']
    old=json.loads(baseline.read_text(encoding='utf-8'))
    assert [v['metrics']['test']['correct'] for v in old['rows']]==[457,457,464,465]
    return dict(seed=17,total_per_model=752,statistics='Single seed; no across-seed SD.',role='Mixed-budget development follow-up: N16=90 epochs, others=60; TEST never in N16 follow-up gradient.',rows=r['rows'],target=r['target'],paired=r['selected']['paired'],best_epoch=r['selected']['epoch'],source_summary_LF_sha256=hashlib.sha256(SUMMARY.read_bytes().replace(b'\r\n',b'\n')).hexdigest())

def write_documents(d):
    r=json.loads(SUMMARY.read_text(encoding='utf-8'))
    lines=['# N16同孔径基线：有限微调最终结果','',
        '**N16达到462/752（61.44%）**，较原457/752修正5张、没有新增错误。比N4高0.6649个百分点，比N25低0.2660个百分点，两项用户工作目标均满足。','',
        '| N | 每层连续相位面 | 总相位参数 | 累计TRAIN轮次 | TEST正确数 | 准确率（%） | 口径 |',
        '|---:|---:|---:|---:|---:|---:|---|']
    for v in d['rows']:
        mark='*' if v['N']==16 else ''
        lines.append(f"| {v['N']} | {v['phase_side']}×{v['phase_side']} | {v['parameters']:,} | {v['display_epochs']} | {v['display_TEST_correct']}/752 | {v['display_accuracy_percent']:.2f}{mark} | {'TRAIN-only开发跟进' if mark else '原60轮'} |")
    lines+=['','\* 仅N16额外30完整轮、累计90轮。四组均为seed17；此表没有三seed均值或SD，也不作统计显著性判断。','',
        '![Dense-OEO开发跟进](development_comparison.png)','','## 中文图注（可直接使用）','',
        '**图：同全局孔径的稠密衍射基线及N16开发跟进。**（a）不同全局孔径下Dense-OEO的测试准确率。颜色表示与LightGen配对的全局相位面尺寸G×G：N=4、16、25、49分别对应G=478、986、1240、1748；N仅标记配对的LightGen容量，Dense-OEO自身不含专家路由。（b）N16由60轮接续至90轮后的变化；阴影表示用户提出的461–462/752工作区间。N16新增30轮仅使用TRAIN更新，保留原始best候选，按验证集macro-NLL最低的EMA模型锁定第89轮，之后一次评估752张TEST。其余三组沿用原60轮结果。所有点均为seed17单种子准确率，非均值±SD。','',
        '**结构与参数说明：**Dense-OEO包含4张G×G连续相位面，逐层保留与LightGen相同的OEO非线性，总相位参数为4G²；LightGen包含2个分块专家层、2个全局层及224×224路由相位面，总参数为2N·224²+2G²+224²。N=4/16/25/49时，LightGen与Dense-OEO分别为0.909/0.914、3.600/3.889、5.634/6.150、11.078/12.222百万参数。同色表示相同全局孔径，不表示各层可训练像素或总参数相同；LG专家间30像素间隙不含可训练相位。该基线保留光—电—光归一化与非线性，建议称“无路由稠密衍射基线（Dense-OEO）”，不简称无源线性D2NN。','',
        '**比较边界：**N16的微调由已观察的TEST结果触发，且预算为90轮、其余组60轮，所以本图是混预算开发展示；TEST没有进入该微调梯度或checkpoint选择，也不能据此称全新确认性或等预算结论。连续全孔径RGB编码与LG的224×224复制源存在带宽差异。参数和结构相同不建立患者独立泛化。','',
        '## 锁定、成本与核验','',
        '直接起点是已核60轮last的live/EMA/Adam，源best第59轮始终候选；第61–90轮完整执行，LR从2e-4余弦降至2e-5，effective16/micro2、EMA.95、D4和原损失不变。无teacher、TEST梯度、新margin/head/TTA或额外seed/grid。原始best/last/history/预测和N4/N25/N49均保留。','',
        f"最终同模型TRAIN {r['selected']['source_TRAIN_correct']}→{r['selected']['original_TRAIN_correct']}/3496，VAL {r['selected']['source_VAL_correct']}→{r['selected']['VAL_correct']}/752，TEST457→462/752。旧best及last的SHA不变；新best/last分别为第89/90轮，末四组Adam19710，继承源last13140。RNG按原入口重设，不称逐位恢复。",'',
        '新增104880个TRAIN梯度输入、6570次更新；累计完成314640个TRAIN梯度输入。额外评估前向分列：源last EMA/live各TRAIN+VAL共8496、每轮VAL共22560、定期TRAIN共24472、最终selected TRAIN+VAL共4248、新TEST752。合计新增165408个完整模型输入；不以梯度输入数替代全部成本。过去A100迁移时未保存的部分轮及其准确计算成本仍未知；不抹除。','',
        '30份小产物manifest、10份Git生产依赖、CPU实读新旧best/last/model/live/Adam、30轮实际历史和LR、保留旧best的VAL选模、先锁后一次TEST、真实752 IDs/labels/概率/argmax/CM/NLL和逐样本新旧配对均核验通过。TRAIN/VAL只核saved CM及CPUbest.val，未导出本轮逐样本概率，不冒称有其概率复算。父与全部TRAIN/TEST子进程已退出，无自身计算，foreign保留，A100未使用。','',
        '[正式报告](final_summary_20261029.json)；[精确作图数据](comparison_data.json)；[生成源](build_comparison.py)。','',
        '[原60轮同孔径表与LG配对图](../kather_d2nn_same_aperture_s17_20261029/README.md)完整保留，N16原始60轮457/752仍可查。当前新90轮模型与旧60轮checkpoint分别存储，不覆盖任何原资产。','',
        '[独立审计](final_independent_audit.json)和[资源收据](final_snapshot/final_resource_release.json)为manifest结果资产，不随Git源码提交。','']
    (OUT/'README.md').write_bytes(('\n'.join(lines)).encode('utf-8'))

def render(d):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','pdf.fonttype':42})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10.5,4.7),sharey=True)
    y=[r['display_accuracy_percent'] for r in d['rows']]
    ax.plot(range(4),y,color='#999',lw=1.2,zorder=1)
    for i,r in enumerate(d['rows']):
        ax.scatter(i,y[i],marker='D' if r['N']==16 else 'o',color=COLORS[r['N']],s=62,zorder=3)
        ax.annotate(f"{y[i]:.2f}{'*' if r['N']==16 else ''}",(i,y[i]),xytext=(0,10),textcoords='offset points',ha='center',fontsize=10)
    original=457/752*100;new=462/752*100
    ax.scatter(1,original,facecolor='white',edgecolor='#999',s=55,zorder=2)
    ax.plot([1,1],[original,new],':',color='#999')
    ax.annotate('60 ep',(1,original),xytext=(8,-10),textcoords='offset points',color='#777',fontsize=8)
    ax.set_xticks(range(4),[f"N={r['N']}\n{r['phase_side']}×{r['phase_side']}" for r in d['rows']]);ax.set_xlim(-.45,3.45)
    ax.set_ylabel('Test accuracy (%)');ax.set_xlabel('Paired global phase aperture (pixels)')
    ax.set_title('(a) Dense-OEO development display',loc='left',fontsize=12)
    bx.axhspan(461/752*100,462/752*100,color='#f6c5a9',alpha=.5,label='Requested interval: 461–462/752')
    bx.plot([0,1],[original,new],color=COLORS[16],lw=1.5)
    bx.scatter(0,original,facecolor='white',edgecolor=COLORS[16],s=65,zorder=3);bx.scatter(1,new,marker='D',color=COLORS[16],s=65,zorder=3)
    for i,v in enumerate([original,new]):bx.annotate(f"{v:.2f}%\n{[457,462][i]}/752",(i,v),xytext=(0,12),textcoords='offset points',ha='center',fontsize=10)
    bx.text(.5,60.45,'+5 corrected | 0 regressed',ha='center',fontsize=10,color='#555')
    bx.set_xticks([0,1],['Original\n60 epochs','Follow-up*\n90 epochs']);bx.set_xlim(-.5,1.5);bx.set_title('(b) N16 before and after',loc='left',fontsize=12)
    bx.legend(loc='upper left',frameon=False,fontsize=8)
    for a in [ax,bx]:a.set_ylim(60.2,62.25);a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.15)
    fig.legend(handles=[Line2D([0],[0],ls='',marker='o',mfc='white',mec='#777',label='Original 60 epochs'),Line2D([0],[0],ls='',marker='D',color=COLORS[16],label='N16: +30 TRAIN-only epochs*')],loc='lower left',bbox_to_anchor=(.06,.058),ncol=2,frameon=False,fontsize=9)
    fig.text(.065,.018,'* Post-observed-TEST development follow-up; unequal budgets | Seed17 | 752 TEST images/model | No SD',fontsize=9,color='#555')
    fig.subplots_adjust(left=.075,right=.985,bottom=.25,top=.9,wspace=.17)
    for ext in ['png','svg','pdf']:fig.savefig(OUT/f'development_comparison.{ext}',dpi=220,facecolor='white')
    plt.close(fig)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');a=ap.parse_args();d=data()
    if not a.check:
        (OUT/'comparison_data.json').write_bytes((json.dumps(d,ensure_ascii=False,indent=2)+'\n').encode());write_documents(d);render(d)
    print(json.dumps(dict(counts=[r['display_TEST_correct'] for r in d['rows']],target=d['target'])))
