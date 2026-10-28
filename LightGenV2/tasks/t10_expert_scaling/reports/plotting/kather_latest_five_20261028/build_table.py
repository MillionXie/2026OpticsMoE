"""Recompute the five latest rows from integer correct counts and render a table.

This is an explicitly mixed-protocol display, not an equal-budget comparison.
No screenshot pixels or manually rounded per-seed accuracies enter statistics.
"""
from pathlib import Path
import csv
import hashlib
import json
import statistics

ROOT=Path(__file__).resolve().parents[6]
REPORTS=ROOT/'LightGenV2/tasks/t10_expert_scaling/reports'
OUT=Path(__file__).resolve().parent
BASE=REPORTS/'plotting/kather_expert_count_20260921/three_seed_test_per_seed.csv'
TAIL=REPORTS/'expert_tail_followup_20261026/final_summary_20261027.json'
FULL120=REPORTS/'kather_n49k49_continue90to120_20261027/final_summary_20261027.json'
FIT=REPORTS/'kather_n49_seed_balanced_followup_20261028/final_summary_20261028.json'
SEEDS=[17,27,37]
GROUPS={4:[1,2,3,4],16:[1,4,8,16],25:[1,6,12,25],49:[1,12,24,49]}
UPDATES={(4,4),(16,16),(25,25),(49,24),(49,49)}


def stats(counts):
    assert len(counts)==3 and all(isinstance(v,int) and 0<=v<=752 for v in counts)
    values=[100*v/752 for v in counts]
    mean=statistics.mean(values);sd=statistics.stdev(values)
    return dict(correct_counts=counts,total_per_seed=752,seeds=SEEDS,mean_percent=mean,sample_sd_percent=sd,display=f'{mean:.2f} ± {sd:.2f}')


def data():
    raw=list(csv.DictReader(BASE.open(encoding='utf-8-sig')));tail=json.loads(TAIL.read_text());fit=json.loads(FIT.read_text());full120=json.loads(FULL120.read_text())
    groups={}
    for row in raw:
        key=(row['architecture'],int(row['experts']),int(row['top_k']));groups.setdefault(key,{})[int(row['seed'])]=row
    original={}
    for key,items in groups.items():
        assert sorted(items)==SEEDS;counts=[]
        for seed in SEEDS:
            accuracy=float(items[seed]['test_accuracy']);count=round(accuracy*752);assert abs(accuracy-count/752)<1e-7;counts.append(count)
        original[key]=stats(counts)
    # Independently check every value transcribed from the supplied screenshot.
    expected={4:['62.50 ± 0.46','69.90 ± 0.73','71.01 ± 0.69','69.73 ± 0.28'],16:['48.36 ± 1.01','70.39 ± 2.04','70.04 ± 1.56','73.89 ± 0.63'],25:['63.79 ± 0.50','80.19 ± 0.23','80.23 ± 0.55','76.46 ± 1.38'],49:['61.13 ± 1.24','80.45 ± 0.35','80.41 ± 0.38','76.64 ± 0.31']}
    d2nn_expected={4:'59.66 ± 0.08',16:'62.50 ± 0.23',25:'61.17 ± 0.87',49:'62.54 ± 0.43'}
    for n,ks in GROUPS.items():
        assert [original['MoE',n,k]['display'] for k in ks]==expected[n]
        assert original['D2NN',n,n]['display']==d2nn_expected[n]
    tail_groups={}
    for n,k in sorted(UPDATES):
        rows=sorted([v for v in tail['rows'] if (v['experts'],v['top_k'])==(n,k)],key=lambda v:v['seed']);assert [v['seed'] for v in rows]==SEEDS
        assert all(v['test_samples']==752 and v['total_epochs']==90 and abs(v['test_accuracy']-v['test_correct']/752)<1e-7 for v in rows)
        tail_groups[n,k]=stats([v['test_correct'] for v in rows])
    students={v['seed']:v for v in fit['students']};counts=[fit['seed17_finalization']['test_fit_correct'],students[27]['test_fit_correct'],students[37]['test_fit_correct']]
    latest_fit=stats(counts);assert counts==[613,608,601]
    assert abs(latest_fit['mean_percent']-fit['descriptive_three_seed_fit']['mean_percent'])<1e-10
    assert abs(latest_fit['sample_sd_percent']-fit['descriptive_three_seed_fit']['sample_standard_deviation_percent'])<1e-10
    rows=[]
    for n,ks in GROUPS.items():
        for k in ks:
            value=latest_fit if (n,k)==(49,49) else tail_groups[n,k] if (n,k) in UPDATES else original['MoE',n,k]
            marker='†' if (n,k)==(49,49) else '*' if (n,k) in UPDATES else ''
            rows.append(dict(experts=n,top_k=k,original_60=original['MoE',n,k],latest=value,marker=marker,updated=(n,k) in UPDATES,role='TEST-trained exposed development fit; unequal discovery budgets' if marker=='†' else '90-epoch post-TEST-observation development follow-up; TEST not in gradient' if marker=='*' else 'Original60-epoch result'))
        candidates=[v for v in rows if v['experts']==n];best=max(v['latest']['mean_percent'] for v in candidates)
        for v in candidates:v['bold_group_max']=v['latest']['mean_percent']==best
    assert sum(v['updated'] for v in rows)==5
    assert [(v['experts'],v['top_k']) for v in rows if v['bold_group_max']]==[(4,4),(16,16),(25,12),(49,24)]
    return dict(statistics='Each seed752 samples; mean±sample SD across seeds17/27/37, ddof=1; computed from integer correct counts before rounding.',display_role='Latest mixed-budget/protocol display. It is not an equal-budget or held-out-generalization comparison.',updated_five=[dict(experts=v['experts'],top_k=v['top_k'],**v['latest'],marker=v['marker'],role=v['role']) for v in rows if v['updated']],rows=rows,d2nn=[dict(experts=n,**original['D2NN',n,n]) for n in GROUPS],full49_history=dict(original60=original['MoE',49,49],tail90=tail_groups[49,49],continued120=stats([v['test_correct'] for v in sorted(full120['rows'],key=lambda v:v['seed'])]),latest_TEST_trained_fit=latest_fit),source_files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [BASE,TAIL,FULL120,FIT]],checks=dict(all20_screenshot_cells_match_original_sources=True,exactly_five_MoE_rows_updated=True,D2NN_unchanged=True,group_maxima_recomputed=True,no_VAL_or_teacher_accuracy_used=True,no_rounding_before_aggregation=True))


def render(d):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties,findfont
    from matplotlib.patches import Rectangle
    fontdir=Path('C:/Windows/Fonts')
    def font(name,fallback,size,weight='normal'):
        p=fontdir/name
        return FontProperties(fname=str(p),size=size,weight=weight) if p.exists() else FontProperties(fname=findfont(fallback),size=size,weight=weight)
    cn=font('msyh.ttc','Noto Sans CJK SC',13)
    header=font('msyh.ttc','Noto Sans CJK SC',23,'bold')
    num=font('times.ttf','DejaVu Serif',24)
    bold=font('timesbd.ttf','DejaVu Serif',24,'bold')
    plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(11.1,13.3),dpi=180);fig.subplots_adjust(0,0,1,1);ax.set_xlim(0,1110);ax.set_ylim(1330,0);ax.axis('off');fig.patch.set_facecolor('white')
    ax.text(24,29,'专家数与 Top-k：最新五组结果汇总',fontproperties=font('msyh.ttc','Noto Sans CJK SC',18,'bold'),va='center',color='#154A73')
    ax.text(24,62,'不同训练预算与指标口径，供汇总展示',fontproperties=cn,va='center',color='#555555')
    x=[24,254,484,794,1086];top=90;hh=92;rh=57
    headers=['专家数','Top-k','MoE 准确率','D2NN准确率']
    for j in range(4):
        ax.add_patch(Rectangle((x[j],top),x[j+1]-x[j],hh,facecolor='#0879BD',edgecolor='white',linewidth=1.5))
        ax.text((x[j]+x[j+1])/2,top+hh/2,headers[j],fontproperties=header,color='white',ha='center',va='center')
    for i,row in enumerate(d['rows']):
        y=top+hh+i*rh;bg='#DDEBF7' if i//4%2==0 else '#E8ECEF'
        for j in range(3):
            fill='#FFF0D5' if j==2 and row['marker']=='†' else bg
            ax.add_patch(Rectangle((x[j],y),x[j+1]-x[j],rh,facecolor=fill,edgecolor='white',linewidth=1.4))
        for j,text in enumerate([str(row['experts']),str(row['top_k']),row['latest']['display']+row['marker']]):
            ax.text((x[j]+x[j+1])/2,y+rh/2,text,fontproperties=bold if j==2 and row['bold_group_max'] else num,color='#7B4912' if j==2 and row['marker']=='†' else 'black',ha='center',va='center')
    for index,v in enumerate(d['d2nn']):
        y=top+hh+index*4*rh;bg='#DDEBF7' if index%2==0 else '#E8ECEF'
        ax.add_patch(Rectangle((x[3],y),x[4]-x[3],4*rh,facecolor=bg,edgecolor='white',linewidth=1.4))
        ax.text((x[3]+x[4])/2,y+2*rh,v['display'],fontproperties=num,ha='center',va='center')
    foot=top+hh+16*rh+30
    notes=['单位：%；三个seed（17/27/37）的均值 ± 样本标准差（ddof=1）。','* 四行更新为累计90轮的开发跟进：4/4、16/16、25/25、49/24。','† 49/49的TEST已进入教师和学生训练，属于拟合结果，不是独立泛化结果。','   该行三seed正确数为613/608/601；起点与累计开发预算不同。','未标记行及D2NN沿用原60轮；本表不用于等预算优劣判断。','加粗为每组列示均值最大项；原数据和全部历史结果均保留。']
    for j,note in enumerate(notes):ax.text(24,foot+j*27,note,fontproperties=cn,va='center',color='#6F471D' if j in [2,3] else '#444444')
    fig.savefig(OUT/'latest_table.png',dpi=180,facecolor='white')
    fig.savefig(OUT/'latest_table.svg',facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    d=data();(OUT/'verified_data.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');render(d)
    print(json.dumps(dict(updated_five=[{k:v[k] for k in ['experts','top_k','correct_counts','display','marker']} for v in d['updated_five']],checks=d['checks']),ensure_ascii=True))
