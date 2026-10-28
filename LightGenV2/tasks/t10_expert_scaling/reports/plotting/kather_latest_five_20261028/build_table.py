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
N25_PILOT=REPORTS/'kather_n25_balanced_distill_20261028/final_summary_20261028.json'
N25_FOLLOWUP=REPORTS/'kather_n25_seed_balanced_followup_20261028/final_summary_20261028.json'
SEEDS=[17,27,37]
GROUPS={4:[1,2,3,4],16:[1,4,8,16],25:[1,6,12,25],49:[1,12,24,49]}
UPDATES={(4,4),(16,16),(25,25),(49,24),(49,49)}


def stats(counts):
    assert len(counts)==3 and all(isinstance(v,int) and 0<=v<=752 for v in counts)
    values=[100*v/752 for v in counts]
    mean=statistics.mean(values);sd=statistics.stdev(values)
    return dict(correct_counts=counts,total_per_seed=752,seeds=SEEDS,mean_percent=mean,sample_sd_percent=sd,display=f'{mean:.2f} ± {sd:.2f}')


def mark_group_maxima(rows):
    # Equal denominators/seed counts make integer totals an exact comparison,
    # including ties; rounded displays and floating-point means are not selectors.
    for n in GROUPS:
        candidates=[row for row in rows if row['experts']==n]
        maximum=max(sum(row['latest']['correct_counts']) for row in candidates)
        for row in candidates:row['bold_group_max']=sum(row['latest']['correct_counts'])==maximum


def data():
    raw=list(csv.DictReader(BASE.open(encoding='utf-8-sig')))
    def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
    tail=read(TAIL);fit=read(FIT);full120=read(FULL120)
    pilot25=read(N25_PILOT);followup25=read(N25_FOLLOWUP)
    assert pilot25['status']=='complete' and pilot25['seed']==17
    assert pilot25['experts']==pilot25['top_k']==25
    assert followup25['status']=='complete', 'N25 follow-up report is not final; keep the previous table'
    assert len(followup25['students'])==2
    students25={row['seed']:row['selected'] for row in followup25['students']}
    assert sorted(students25)==[27,37]
    selected25={17:pilot25['selected'],**students25}
    for selected in selected25.values():
        assert selected['TEST_total']==752 and type(selected['TEST_fit_correct']) is int
        assert type(selected['original_TRAIN_correct']) is int and 0<=selected['original_TRAIN_correct']<=3496
        assert type(selected['VAL_correct']) is int and 0<=selected['VAL_correct']<=752
    latest25=stats([selected25[seed]['TEST_fit_correct'] for seed in SEEDS])
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
            value=latest25 if (n,k)==(25,25) else latest_fit if (n,k)==(49,49) else tail_groups[n,k] if (n,k) in UPDATES else original['MoE',n,k]
            marker='†' if (n,k) in [(25,25),(49,49)] else '*' if (n,k) in UPDATES else ''
            role=('TEST-trained exposed development fit; 90-epoch baseline plus fixed16x1024 adaptation and shared extra teacher' if (n,k)==(25,25)
                else 'TEST-trained exposed development fit; unequal discovery budgets' if (n,k)==(49,49)
                else '90-epoch post-TEST-observation development follow-up; TEST not in gradient' if marker=='*' else 'Original60-epoch result')
            rows.append(dict(experts=n,top_k=k,original_60=original['MoE',n,k],latest=value,marker=marker,updated=(n,k) in UPDATES,role=role))
    mark_group_maxima(rows)
    assert sum(v['updated'] for v in rows)==5
    assert [(v['experts'],v['top_k']) for v in rows if v['bold_group_max'] and v['experts']!=25]==[(4,4),(16,16),(49,24)]
    # All fifteen other MoE values remain on their previous exact integer counts.
    for row in rows:
        key=(row['experts'],row['top_k'])
        if key==(25,25):continue
        previous=latest_fit if key==(49,49) else tail_groups[key] if key in UPDATES else original[('MoE',)+key]
        assert row['latest']==previous
    return dict(statistics='Each seed752 samples; mean±sample SD across seeds17/27/37, ddof=1; computed from integer correct counts before rounding.',display_role='Latest mixed-budget/protocol display. It is not an equal-budget or held-out-generalization comparison.',updated_five=[dict(experts=v['experts'],top_k=v['top_k'],**v['latest'],marker=v['marker'],role=v['role']) for v in rows if v['updated']],rows=rows,d2nn=[dict(experts=n,**original['D2NN',n,n]) for n in GROUPS],full25_history=dict(original60=original['MoE',25,25],tail90=tail_groups[25,25],latest_TEST_trained_fit=latest25),full49_history=dict(original60=original['MoE',49,49],tail90=tail_groups[49,49],continued120=stats([v['test_correct'] for v in sorted(full120['rows'],key=lambda v:v['seed'])]),latest_TEST_trained_fit=latest_fit),source_checksum_scope='UTF-8 source text with CRLF normalized to LF; matches Git blob identity across Windows/Linux.',source_files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()) for p in [BASE,TAIL,FULL120,FIT,N25_PILOT,N25_FOLLOWUP]],checks=dict(all20_screenshot_cells_match_original_sources=True,exactly_five_MoE_rows_updated=True,only_N25_latest_value_replaced=True,N25_all_three_completed_reports_used=True,other15_MoE_values_unchanged=True,D2NN_unchanged=True,full49_history_unchanged=True,group_maxima_recomputed=True,group_maxima_recomputed_from_integer_totals=True,no_VAL_or_teacher_accuracy_used=True,no_rounding_before_aggregation=True))


def update_readme(d):
    """Fill only the two explicit generated blocks after final reports validate."""
    path=OUT/'README.md';text=path.read_text(encoding='utf-8')
    def replace_block(name,body):
        nonlocal text
        begin=f'<!-- BEGIN {name} -->';end=f'<!-- END {name} -->'
        assert text.count(begin)==text.count(end)==1
        before,rest=text.split(begin);_,after=rest.split(end)
        text=before+begin+'\n'+body+'\n'+end+after
    lines=['| 专家数 | Top-k | Seed17正确数 | Seed27正确数 | Seed37正确数 | 最新均值±样本SD（%） | 口径 |',
        '|---|---|---:|---:|---:|---:|---|']
    for row in d['updated_five']:
        a,b,c=row['correct_counts'];role='TEST参与教师和学生训练后的暴露拟合' if row['marker']=='†' else '累计90轮开发跟进'
        lines.append(f"| {row['experts']} | {row['top_k']} | {a} | {b} | {c} | {row['display']}{row['marker']} | {role} |")
    replace_block('VERIFIED FIVE','\n'.join(lines))
    maxima='、'.join(f"{row['experts']}/{row['top_k']}" for row in d['rows'] if row['bold_group_max'])
    replace_block('GROUP MAXIMA',f'按三个seed整数正确数总和比较组内列示均值，加粗位置为{maxima}；总数相同则并列加粗。')
    path.write_text(text,encoding='utf-8')


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
    notes=['单位：%；三个seed（17/27/37）的均值 ± 样本标准差（ddof=1）。','* 三行累计90轮开发跟进：4/4、16/16、49/24；未标记行及D2NN沿用原60轮。','† 25/25、49/49的TEST已进入教师和学生训练，属于暴露拟合，不是独立泛化。','25/25每seed另加16×1024输入短段；49/49保留不同起点与累计发现预算。','† 两行复用含ImageNet知识的教师；额外教师成本单列，不作等预算优劣判断。','加粗按组内整数正确数总和判定最大列示均值；原数据和全部历史结果均保留。']
    for j,note in enumerate(notes):ax.text(24,foot+j*27,note,fontproperties=cn,va='center',color='#6F471D' if j in [2,3] else '#444444')
    fig.savefig(OUT/'latest_table.png',dpi=180,facecolor='white')
    fig.savefig(OUT/'latest_table.svg',facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    d=data();(OUT/'verified_data.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');render(d);update_readme(d)
    print(json.dumps(dict(updated_five=[{k:v[k] for k in ['experts','top_k','correct_counts','display','marker']} for v in d['updated_five']],checks=d['checks']),ensure_ascii=True))
