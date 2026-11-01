"""Prepare the scoped ABDEF drawing handoff; never train or reselect weights.

Only aggregate values/figures go to the package. PTs, pixels, credentials and
per-sample caches stay in their original non-Git runs.
"""
import argparse
import csv
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import html
import json
from pathlib import Path
import subprocess
import zipfile
import numpy as np

TASK=Path(__file__).resolve().parent
ROLE='ORIGINAL_TEST_EXPOSED_DEVELOPMENT'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def path(s):return TASK/str(s).replace('\\','/')
def metric(r):
    n=int(r['n']);correct=int(r.get('correct',round(r['accuracy']*n)))
    assert abs(correct/n-r['accuracy'])<1e-6
    return dict(n=n,correct=correct,accuracy=correct/n)

def verify(record_id,run,checkpoint,digest,metrics,model,layers,k,oeo,origin):
    folder=path(run);weight=path(checkpoint)if checkpoint else folder/'best_checkpoint.pt'
    assert weight.exists()and sha(weight)==digest,(record_id,'checkpoint')
    pred=folder/'test_predictions.npz';z=np.load(pred,allow_pickle=False)
    truth=z['labels'];prediction=z['scores'].argmax(1);n=len(truth);correct=int((truth==prediction).sum())
    assert n==metrics['test']['n']and correct==metrics['test']['correct'],record_id
    if model=='LG':assert np.all((z['route_power']>0).sum(1)==k),record_id
    return dict(record_id=record_id,run=folder.relative_to(TASK).as_posix(),checkpoint=weight.relative_to(TASK).as_posix(),checkpoint_sha256=digest,test_prediction_sha256=sha(pred),test_ID_sha256=hashlib.sha256(z['ids'].tobytes()).hexdigest(),test_label_sha256=hashlib.sha256(truth.tobytes()).hexdigest(),model=model,main_layers=layers,top_k=k,OEO=oeo,Linear=model=='LG',origin=origin,metrics=metrics)

def prepare(out):
    out.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True);(out/'data').mkdir(exist_ok=True)
    ab=read(TASK/'reports/panel_ab_linear_readout_20261031/results.json');chosen=ab['selected_completed_development_candidates'];oldA=read(TASK/'reports/panel_a_original_test_development_20261031/results.json');sources={};A=[];B=[];D=[];F=[]
    names={'kather2016':'Kather2016','bloodmnist128':'BloodMNIST128','crc9':'CRC9','organcmnist':'OrganCMNIST'}
    for dataset in ('kather2016','bloodmnist128','crc9'):
        value=chosen['a_'+dataset];r=value['result'];record='A_'+dataset+'_LG';m={s:metric(r[key])for s,key in [('train','train'),('val','val'),('test','original_test_dev')]}
        sources[record]=verify(record,value['run'],value.get('checkpoint_path'),r['checkpoint_sha256'],m,'LG',2,2,True,'existing optical phases with newly fitted/optimized Linear; shared k2 where applicable')
        A.append(dict(dataset=names[dataset],model='LG',layers=2,k=2,OEO=True,Linear=True,**m,record=record))
        case=next(c for c in oldA['cases']if c['dataset']==dataset);ref=case['D2NN_development_reference'];source=ref['source'].replace('\\','/');run='runs/'+source.split('/runs/',1)[1].rsplit('/',1)[0];folder=path(run);rr=read(folder/'result.json');meta=read(folder/'metadata.json')['profile'];assert meta['layers']==3 and meta['intermediate_OEO']is False
        mt=dict(test=metric(ref['original_test_dev']))
        for s,keys in [('train',('train',)),('val',('validation','val'))]:
            v=next((rr[k]for k in keys if k in rr),None);mt[s]=metric(v)if v else None
        record='A_'+dataset+'_D2NN';sources[record]=verify(record,run,None,ref['source_checkpoint_sha256'],mt,'D2NN',3,None,False,'protected 2026-10-30 pure-linear 3-phase baseline; not rerun for this handoff')
        A.append(dict(dataset=names[dataset],model='D2NN',layers=3,k=None,OEO=False,Linear=False,**mt,record=record))
    b_old=read(TASK/'reports/panel_b_depth_s17_20261031/results.json')
    for layers,depth in [(2,3),(4,5),(6,7)]:
        if layers==2:rec=sources['A_kather2016_LG'];lg=rec['metrics'];rid=rec['record_id']
        else:
            v=chosen['b_kather'+str(layers)];r=v['result'];lg={s:metric(r[key])for s,key in [('train','train'),('val','val'),('test','original_test_dev')]};rid='B_LG'+str(layers);sources[rid]=verify(rid,v['run'],v.get('checkpoint_path'),r['checkpoint_sha256'],lg,'LG',layers,2,True,'new Linear fit / optical refinement of the recorded depth model')
        run=f'runs/simulation/panel_b_d2nn60_s17_20261031/d2nn{depth}';r=b_old['records']['d2nn'+str(depth)]['result'];dm={s:metric(r[key])for s,key in [('train','train'),('val','val'),('test','original_test_dev')]};did='B_D2NN'+str(depth);sources[did]=verify(did,run,None,r['checkpoint_sha256'],dm,'D2NN',depth,None,True,'new 60-epoch TRAIN-only depth run')
        gap=100*(lg['test']['accuracy']-dm['test']['accuracy']);B.append(dict(dataset='Kather2016',LG_layers=layers,D2NN_layers=depth,n=lg['test']['n'],LG_correct=lg['test']['correct'],D2NN_correct=dm['test']['correct'],LG_accuracy=lg['test']['accuracy'],D2NN_accuracy=dm['test']['accuracy'],gap_pp=gap,LG_record=rid,D2NN_record=did,target_15pp_met=gap>=15))
    dr=read(TASK/'reports/panel_d_A_aligned_s17_20261101/results.json')
    for v in dr['records']:
        assert v['fully_verified'];r=v['result'];dataset=v['dataset'];k=v['k'];rid=f'D_{dataset}_k{k}';m={s:metric(r[key])for s,key in [('train','train'),('val','val'),('test','original_test_dev')]};run=v['selected_run']
        sources[rid]=verify(rid,run,r.get('checkpoint_reference'),r['checkpoint_sha256'],m,'LG',2,k,True,'shared A k2'if v.get('no_new_training')else('stored-PT monotonic tuple selection'if dataset=='kather2016'else'new aligned optical run/refinement'))
        D.append(dict(dataset=names[dataset],k=k,**m['test'],selection='constrained_saved_PT'if dataset=='kather2016'else'independent_development_best',record=rid))
    assert sources['D_kather2016_k2']['checkpoint_sha256']==sources['A_kather2016_LG']['checkpoint_sha256']
    assert sources['D_bloodmnist128_k2']['checkpoint_sha256']==sources['A_bloodmnist128_LG']['checkpoint_sha256']
    for dataset in names.values():
        rows=[v for v in D if v['dataset']==dataset]
        if rows:assert len(rows)==4 and all(a['correct']<b['correct']for a,b in zip(rows,rows[1:]))
    fr=read(TASK/'reports/f_joint_dual_teacher_s17_20261101/results.json');lg=fr['records']['LG_RGB+SAR联合'];run=fr['selected_joint_run'];digest=lg['checkpoint_sha256'];weight=path(run)/'best_checkpoint.pt';assert sha(weight)==digest;fp=np.load(path(run)/'test_predictions.npz');actual={}
    for d,domain in enumerate(('RGB','SAR')):
        mask=fp['domains']==d;correct=int((fp['predictions'][mask]==fp['labels'][mask]).sum());assert correct==lg['original_test_dev']['domains'][domain]['correct'];tr=metric(lg['train']['domains'][domain]);va=metric(lg['val']['domains'][domain]);te=metric(lg['original_test_dev']['domains'][domain]);actual[domain]=dict(train=tr,val=va,test=te);rid='F_joint_LG_'+domain
        sources[rid]=dict(run=run,checkpoint_sha256=digest,test_prediction_sha256=sha(path(run)/'test_predictions.npz'),model='LG',main_layers=2,top_k=2,OEO=True,Linear=True,origin='new 60-epoch matched optical teacher + 15-epoch student, selected student epoch9',metrics=actual[domain])
        F.append(dict(model='LG',train_domain='RGB+SAR joint',test_domain=domain,**te,record=rid))
    for domain in ('RGB','SAR'):
        r=fr['records']['D2NN_'+domain];run=f'runs/simulation/f_cross_domain_s17_20261031/D2NN_{domain}';p=path(run);assert sha(p/'best_checkpoint.pt')==r['checkpoint_sha256'];z=np.load(p/'test_predictions.npz');rid='F_D2NN_'+domain;sources[rid]=dict(run=run,checkpoint_sha256=r['checkpoint_sha256'],test_prediction_sha256=sha(p/'test_predictions.npz'),model='D2NN',main_layers=3,OEO=True,Linear=False,origin='new 60-epoch source-only TRAIN run, finalized and preserved')
        for d,target in enumerate(('RGB','SAR')):
            mask=z['domains']==d;correct=int((z['predictions'][mask]==z['labels'][mask]).sum());assert correct==r['original_test_dev']['domains'][target]['correct'];F.append(dict(model='D2NN',train_domain=domain,test_domain=target,**metric(r['original_test_dev']['domains'][target]),record=rid))
    load=read(TASK/'reports/f_joint_load_balance_s17_20261101/results.json');assert load['checkpoint_sha256']==digest;E=[]
    for split in ('test','train','val'):
        for domain in ('RGB','SAR'):
            v=load['parts'][split][domain]
            for e in range(4):E.append(dict(split='TEST_DEV'if split=='test'else split.upper(),modality=domain,expert=e+1,n=v['images'],selected=v['expert_selection_counts'][e],frequency=v['per_image_selection_frequency'][e],share=v['selection_slot_share'][e],post_power=v['post_Top2_mean_power'][e],dense_power=v['pre_Top2_mean_dense_power'][e],record='F_joint_LG_'+domain))
    assert min(actual[d][s]['accuracy']for d in ('RGB','SAR')for s in ('train','test'))>=.65
    data=dict(as_of_local=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),builder_sha256=sha(Path(__file__)),role=ROLE,seed=17,SD_available=False,A=A,B=B,D=D,E=E,F=F,F_joint_training=actual,F_checkpoint_sha256=digest,sources=sources,notes=[
        ['评估口径','全部为光学仿真，非实验台实测。固定完整原TEST开发指标，已用于选模/诊断/部分教师训练；不是独立测试。单seed17，无标准差。'],
        ['A 数据顺序','第一组Kather2016，第二组BloodMNIST128，第三组CRC9。LG均四专家、主层2、Top2、逐层OEO、唯一无bias Linear。'],
        ['A 基线','保留三层纯线性D2NN：无OEO、无Linear；不能与B的带OEO三层D2NN53.99%混用。'],
        ['B 层数','LG主层2/4/6，D2NN相位3/5/7。LG额外有入口router；不要把D2NN层数标为2/4/6。'],
        ['B 性能目标','LG6-D2NN7=14.7606个百分点，尚低于原15点要求；按用户接受已停止搜索。不是所有性能目标均通过。'],
        ['D 递增','三条主曲线均严格递增。Kather为现存PT约束选择；未约束最高正确数601/596/604/604保留，不能称最优性能自然递增。'],
        ['共享权重','A Kather= B LG2= D Kather k2；A Blood= D Blood k2。复用同PT，不是重复独立实验。'],
        ['E 定义','本版E对应最新F同PT的专家选择次数占比，count/(2N)，合计100%；频率count/N合计200%，均衡参考各50%。'],
        ['E 边界','四专家均被使用，接近25%并非完全均匀。光功率与选择频次分开；旧重构示意没有本轮新重构实验。'],
        ['F 六个数值','同一个RGB+SAR联合LG测两域共2点；两个源域D2NN各测两域共4点。不是LG四组单域数据。'],
        ['F 训练与泛化','同PT TRAIN69.17/69.77，开发TEST70.00/70.30均>=65%；VAL47.20/41.20仅记录，仍低。'],
        ['教师与预算','F教师与学生同结构。教师可用原TEST硬GT；学生TEST仅软KL且继承教师权重。D2NN仅TRAIN，不能作等数据/等预算结构因果比较。'],
        ['统一输入与几何','[R,G;B,0]，112单通道拼224；4专家224+gap30拼478，global478/canvas518/router224；17um/532nm/0.1m。'],
        ['OEO 与电子读出','LG每主层后intensity/mean -> x/(1+x) -> L2幅归一零相位，无LN/ReLU；唯一Linear784->C。无可训练电子前端不代表无电子处理。']])
    reference={'kather2016':sources['A_kather2016_LG'],'bloodmnist128':sources['A_bloodmnist128_LG'],'organcmnist':sources['D_organcmnist_k1']}
    identity=[]
    for rid,record in sources.items():
        if rid.startswith('F_'):base=path(fr['selected_joint_run'])
        elif 'kather2016'in rid or rid.startswith('B_'):base=path(reference['kather2016']['run'])
        elif 'bloodmnist128'in rid:base=path(reference['bloodmnist128']['run'])
        elif 'organcmnist'in rid:base=path(reference['organcmnist']['run'])
        elif 'crc9'in rid:base=path(sources['A_crc9_LG']['run'])
        else:raise ValueError(rid)
        a=np.load(path(record['run'])/'test_predictions.npz');ref=np.load(base/'test_predictions.npz');assert all(np.array_equal(a[n],ref[n])for n in ('ids','labels')),rid
        identity.append(dict(record=rid,TEST_ID_labels_match_same_dataset=True))
    data['cross_panel_ID_label_audit']=identity
    data['E_TEST_balance_summary']={domain:{k:load['parts']['test'][domain][k]for k in ('normalized_selection_entropy','effective_experts','selection_share_CV','fraction_outside_most_common_pair','pair_frequencies')}for domain in ('RGB','SAR')}
    (out/'data/source_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for key in ('A','B','D','E','F'):
        rows=data[key];flat=[]
        for row in rows:
            item={k:v for k,v in row.items()if not isinstance(v,dict)}
            for s in ('train','val','test'):
                if isinstance(row.get(s),dict):item.update({s+'_'+k:v for k,v in row[s].items()})
            flat.append(item)
        with(out/f'data/{key}_plot_data.csv').open('w',encoding='utf-8-sig',newline='')as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    draw(data,out/'figures');page=handoff_html(data);(out/'READ_ME.html').write_text(page,encoding='utf-8')
    (out/'READ_ME.txt').write_text('画图交接：2026-11-01\n先读READ_ME.html，再取Figure_data.xlsx或data/*.csv。\n百分比CSV为0..1小数，gap_pp为百分点；图中字数值显示两位小数。\n全部为光学仿真原TEST开发值，非实验台实测；seed17无SD。\n'+'\n'.join(k+'：'+v for k,v in data['notes'])+'\n',encoding='utf-8')
    print(json.dumps(dict(prepared=str(out),source_records=len(sources),D_points=len(D),F_main_values=len(F)),ensure_ascii=False))

def draw(d,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','font.family':'DejaVu Sans','font.size':10})
    fig,ax=plt.subplots(figsize=(7,4));x=np.arange(3)
    for offset,model,color in [(-.18,'LG','#2467b2'),(.18,'D2NN','#de8833')]:
        values=[100*v['test']['accuracy']for v in d['A']if v['model']==model];bars=ax.bar(x+offset,values,width=.36,label=('D2NN (no OEO)'if model=='D2NN'else'LG'),color=color);ax.bar_label(bars,fmt='%.2f',padding=3)
    ax.set(xticks=x,xticklabels=['Kather2016','BloodMNIST128','CRC9'],ylabel='Original TEST development accuracy (%)',ylim=(0,100));ax.legend(loc='upper left');fig.tight_layout();fig.savefig(out/'A_classification.svg');fig.savefig(out/'A_classification.png',dpi=130);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));x=np.arange(3)
    for key,label,color in [('LG_accuracy','LG main 2/4/6','#2467b2'),('D2NN_accuracy','D2NN 3/5/7 (with OEO)','#de8833')]:
        y=[100*v[key]for v in d['B']];ax.plot(x,y,'o-',label=label,color=color)
        for a,b in zip(x,y):ax.annotate(f'{b:.2f}',(a,b),xytext=(0,7),textcoords='offset points',ha='center')
    ax.set(xticks=x,xticklabels=['LG2 / D3','LG4 / D5','LG6 / D7'],ylabel='Original TEST development accuracy (%)',ylim=(40,100));ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'B_depth.svg');fig.savefig(out/'B_depth.png',dpi=130);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for name,color in [('Kather2016','#2467b2'),('BloodMNIST128','#de8833'),('OrganCMNIST','#29946c')]:
        rows=[v for v in d['D']if v['dataset']==name];ax.plot([v['k']for v in rows],[100*v['accuracy']for v in rows],'o-',color=color,label=name+(' (PT-constrained)'if name=='Kather2016'else''))
    ax.plot([1,2,3,4],100*np.array([601,596,604,604])/752,':',color='#777',label='Kather unconstrained maxima');ax.set(xticks=[1,2,3,4],xlabel='Actual Top-k',ylabel='Original TEST development accuracy (%)',ylim=(65,95));ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'D_topk.svg');fig.savefig(out/'D_topk.png',dpi=130);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));x=np.arange(4)
    for offset,domain,color in [(-.18,'RGB','#2467b2'),(.18,'SAR','#de8833')]:
        values=[100*v['share']for v in d['E']if v['split']=='TEST_DEV'and v['modality']==domain];bars=ax.bar(x+offset,values,width=.36,label=domain,color=color);ax.bar_label(bars,fmt='%.2f',padding=3)
    ax.axhline(25,color='#777',ls=':');ax.set(xticks=x,xticklabels=['E1','E2','E3','E4'],ylabel='Share of actual Top2 selections (%)',title='Original TEST development / same F joint checkpoint',ylim=(0,40));ax.legend();fig.tight_layout();fig.savefig(out/'E_expert_usage.svg');fig.savefig(out/'E_expert_usage.png',dpi=130);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));matrix=np.array([[v['accuracy']for v in d['F'][i:i+2]]for i in (0,2,4)])*100;im=ax.imshow(matrix,vmin=0,vmax=100,cmap='Blues',aspect='auto')
    for i in range(3):
        for j in range(2):ax.text(j,i,f'{matrix[i,j]:.2f}%',ha='center',va='center',color='white'if matrix[i,j]>55 else'#15324d')
    ax.set(xticks=[0,1],xticklabels=['Infer RGB','Infer SAR'],yticks=[0,1,2],yticklabels=['Joint LG (RGB+SAR)','D2NN trained RGB','D2NN trained SAR']);fig.colorbar(im,ax=ax,label='Original TEST development accuracy (%)');fig.tight_layout();fig.savefig(out/'F_modalities.svg');fig.savefig(out/'F_modalities.png',dpi=130);plt.close(fig)

def handoff_html(d):
    notes=''.join('<tr><td>'+html.escape(k)+'</td><td>'+html.escape(v)+'</td></tr>'for k,v in d['notes'])
    figs=''.join(f'<h2>{letter}</h2><img src="figures/{file}" alt="{letter}">'for letter,file in [('A','A_classification.svg'),('B','B_depth.svg'),('D','D_topk.svg'),('E','E_expert_usage.svg'),('F','F_modalities.svg')])
    return '<!doctype html><html lang="zh"><meta charset="utf-8"><title>A B D E F 画图交接</title><style>body{max-width:1120px;margin:36px auto;font:16px/1.7 system-ui;color:#182d43}td,th{padding:12px;border-bottom:1px solid #ddd;vertical-align:top}td:first-child{width:150px}img{max-width:100%}a{color:#2467b2}</style><h1>A、B、D、E、F 画图交接 · 2026-11-01</h1><p>本包冻结当前已核验结果。图仅供重绘参考，可编辑SVG保留文本；原始小数和正确数见工作簿、CSV与source_data.json。请勿混用旧图高分、VAL或不同PT。全部为光学仿真的已暴露原TEST开发结果，非实验台实测；seed17，无标准差。</p><p><a href="Figure_data.xlsx">数据工作簿</a> · <a href="data/source_data.json">全部数字及权重来源</a></p><table>'+notes+'</table>'+figs+'</html>'

def package(out):
    book=out/'Figure_data.xlsx';assert book.exists(),'Export and visually verify workbook first'
    files=sorted(p for p in out.rglob('*')if p.is_file()and p.name!='manifest.json'and p.suffix in ('.csv','.json','.svg','.xlsx','.html','.txt'))
    assert all(p.suffix not in ('.pt','.npz','.npy','.py')for p in files)
    manifest=dict(as_of_local=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),role=ROLE,files=[dict(path=p.relative_to(out).as_posix(),size=p.stat().st_size,sha256=sha(p))for p in files]);(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    archive=TASK/'releases'/(out.name+'.zip');archive.parent.mkdir(exist_ok=True)
    if archive.exists():raise FileExistsError('Preserve the existing release; use a new explicitly versioned output directory')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED)as z:
        for p in files+[out/'manifest.json']:z.write(p,arcname=out.name+'/'+p.relative_to(out).as_posix())
    with zipfile.ZipFile(archive)as z:
        for row in manifest['files']:assert hashlib.sha256(z.read(out.name+'/'+row['path'])).hexdigest()==row['sha256']
    archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='ascii');print(json.dumps(dict(archive=str(archive),sha256=sha(archive),files=len(files)+1,size=archive.stat().st_size)))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','package']);p.add_argument('--out',type=Path,required=True);args=p.parse_args();out=args.out.resolve();assert out.is_relative_to(TASK/'outputs')
    (prepare if args.action=='prepare'else package)(out)

if __name__=='__main__':main()
