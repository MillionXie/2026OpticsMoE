"""CPU-only export of saved observations. No model forward, training or hardware.

Missing observations stay missing. Historical table values, packaged replays,
and numerically matching but unproven screenshot versions remain distinguished.
"""
import csv, hashlib, json, math,io,pickle,zipfile
from collections import Counter,OrderedDict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr
from scipy.io import loadmat

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
T=ROOT/'LightGenV2/tasks'
TASKS=[]
SR=OUT/'source_records'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def csvread(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def flat(d,group='overall'):
    out={}
    for k,v in d.items():
        if isinstance(v,dict):out.update(flat(v,group+'/'+k))
        elif isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v):out[(group,k)]=v
    return out
def new(key,name,primary,expected,notes):
    d=dict(task=key,name=name,primary=primary,expected=dict(zip(['Sim','Exp'],expected)),notes=notes,
           stages={},headers=[],samples=[],sources=[],checks=[])
    TASKS.append(d);return d
def source(d,p):
    p=Path(p);record={'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}
    if record not in d['sources']:d['sources'].append(record)
    return record['path']
def stage(d,label,metrics,checkpoint,status,notes,files):
    for p in files:source(d,p)
    d['stages'][label]=dict(metrics=[[g,k,v] for (g,k),v in metrics.items()],checkpoint_sha256=checkpoint,status=status,notes=notes)
def check(d,label,name,value):d['checks'].append(dict(stage=label,check=name,detail=value))
def mean(xs):
    xs=list(xs);return sum(xs)/len(xs)

def saved_tensors(path):
    """Restricted NumPy read of tensor-only ZIP data; never executes pickle code."""
    kinds={'FloatStorage':'<f4','HalfStorage':'<f2','LongStorage':'<i8'}
    class Storage:
        def __init__(self,name):self.name=name
    def tensor(storage,offset,size,stride,*ignored):
        a=storage[int(offset):]
        return np.lib.stride_tricks.as_strided(a,shape=tuple(size),strides=tuple(int(s)*a.itemsize for s in stride)).copy()
    with zipfile.ZipFile(path) as z:
        prefix=next(n[:-8] for n in z.namelist() if n.endswith('data.pkl'))
        class Reader(pickle.Unpickler):
            def find_class(self,module,name):
                if module=='torch' and name in kinds:return Storage(name)
                if module=='torch._utils' and name in ('_rebuild_tensor_v2','_rebuild_tensor'):return tensor
                if module=='collections' and name=='OrderedDict':return OrderedDict
                raise ValueError('Unexpected pickle global: '+module+'.'+name)
            def persistent_load(self,pid):
                kind,typ,key,location,count=pid;assert kind=='storage' and isinstance(typ,Storage)
                return np.frombuffer(z.read(prefix+'data/'+key),dtype=kinds[typ.name],count=count)
        return Reader(io.BytesIO(z.read(prefix+'data.pkl'))).load()

def full_rankings(queries,candidates,query_labels,candidate_labels):
    q=np.asarray(queries,dtype=np.float32);c=np.asarray(candidates,dtype=np.float32)
    q=q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),np.float32(1e-12));c=c/np.maximum(np.linalg.norm(c,axis=1,keepdims=True),np.float32(1e-12))
    order=np.argsort(-(q@c.T),axis=1,kind='stable');rel=np.asarray(candidate_labels)[order]==np.asarray(query_labels)[:,None]
    counts=rel.sum(1);assert (counts>0).all()
    first=rel.argmax(1)+1;positions=np.arange(1,rel.shape[1]+1,dtype=np.float64)
    ap=(np.cumsum(rel,axis=1)/positions[None]*rel).sum(1)/counts
    used=min(10,rel.shape[1]);ap10=(np.cumsum(rel[:,:used],axis=1)/positions[None,:used]*rel[:,:used]).sum(1)/np.minimum(counts,used)
    return order,first,ap,ap10,[(np.flatnonzero(r)+1).tolist() for r in rel]
def quality(pairs):
    y=np.asarray([r[1] for r in pairs],dtype=np.float64);p=np.asarray([r[2] for r in pairs],dtype=np.float64)
    return dict(count=len(y),srcc=float(spearmanr(y,p).statistic),krcc=float(kendalltau(y,p).statistic),plcc=float(pearsonr(y,p).statistic),rmse=float(np.sqrt(np.mean((y-p)**2))),mae=float(np.mean(abs(y-p))))
def quality_samples(d,label,rows):
    d['headers']=['stage','sample_id','MOS_target','predicted_score','signed_error','absolute_error','squared_error','partition']
    for sid,y,p,part in rows:d['samples'].append([label,sid,y,p,p-y,abs(p-y),(p-y)**2,part])
    assert len(rows)==len(set(r[0] for r in rows))
    check(d,label,'saved_prediction_recalculation',quality(rows))
def release_rows(release):
    rows=[]
    for f in release['fields']:
        for sid,y,p,valid in zip(f['sample_ids'],f['targets'],f['simulation_prediction'],f['valid']):
            if valid:rows.append((sid,y,p,'TEST'))
    return rows

# 1. Temporal: same PT and the same 558 video identifiers.
b=T/'t06_video_quality_assessment/runs/hardware'
d=new('lgvq_temporal','LGVQ 时间质量','srcc',[.8044,.7977],'同一权重；558 TEST视频。SRCC/KRCC/PLCC仅为整体指标，不是逐视频指标。')
rp=b/'shs_20260914/temporal/release.json';ep=b/'temporal_full_20260914/results.json';r=read(rp);e=read(ep)
sr=release_rows(r);er=[(x['video'],x['target'],x['prediction'],'TEST') for x in e['rows']]
assert len(sr)==len(er)==558 and {x[0]:x[1] for x in sr}=={x[0]:x[1] for x in er}
for label,m,rows,p in [('Sim',r['simulation_metrics'],sr,rp),('Exp',e['metrics'],er,ep)]:
    stage(d,label,flat(m),r['checkpoint_sha256'],'verified_full_test','仿真发布包与实验报告以sample_id及MOS一一核对。',[p])
    quality_samples(d,label,rows)
    assert abs(quality(rows)['srcc']-m['srcc'])<1e-12
check(d,'pair','same_sample_ids_targets',True)

# 2. Spatial: historical simulation and a packaged numerical replay differ.
d=new('lgvq_spatial','LGVQ 空间质量','srcc',[.6710,.6381],'实验完整558含112条参与适配的旧TEST及446条选择集；不是独立泛化结果。原仿真记录.6710079，发布包重放.6710969；两者分别保留。')
rp=b/'shs_20260914/spatial/release.json';ep=b/'spatial_train2250_20260914/readout_partial_test20/weighted_target5/train2250_plus_test112_holdout446/results.json';r=read(rp);e=read(ep)
sm=flat(r['simulation_metrics']);sm[('historical_training','srcc')]=r['reference_srcc']
stage(d,'Sim',sm,r['checkpoint_sha256'],'same_pt_packaged_replay','逐视频来自发布包，故对应SRCC=.6710968960；截图.6710对应历史训练记录，不能把重放逐样本冒称历史原预测。',[rp])
quality_samples(d,'Sim',release_rows(r))
er=[(x['video'],x['target'],x['after'],x['partition']) for x in e['rows'] if x['partition']!='original_train']
assert len(er)==558 and Counter(x[3] for x in er)=={'test_holdout':446,'test_adaptation':112}
em=flat(e['after']['original_test_all_mixed']);em[('overall','krcc')]=quality(er)['krcc']
for group,key in [('heldout_selection','test_holdout_selection'),('adaptation_seen','test_adaptation')]:em.update(flat(e['after'][key],group))
stage(d,'Exp',em,e['best_checkpoint_sha256'],'verified_mixed_test','112旧TEST参与适配，446旧TEST用于选择；保留partition；不可称558独立TEST。',[ep])
quality_samples(d,'Exp',er)
assert {x[0] for x in er}=={x[0] for x in release_rows(r)}
assert abs(quality(er)['srcc']-e['after']['original_test_all_mixed']['srcc'])<1e-12

# 3. Image-to-text: distinguish the historical training aggregate and the
# complete same-weight A100 replay; never invent the historical 1934 ranks.
d=new('abo_image_to_text','ABO 图搜文','hit_at_1',[.8058,.8029],'用户确认Exp更正为1927/2400=.8029167（显示.8029），原.8019167保留。Sim已补2400条同PT A100回放：1931/2400=.8045833；历史训练.8058333另列，不能假称回放明细属于历史1934次命中。')
d['original_screenshot_exp']=.8019
p=ROOT/'.codex_tmp/abo_import/review/inference/reference/final_report.json';r=read(p)
rp=T/'t08_abo_image_text_retrieval/reports/figure_analysis_20261013/raw/ours_same_weight_replay_metrics.json';pr=SR/'server/868ba39b47_retrieval_predictions.csv';replay=read(rp)
assert replay['checkpoint_sha256']==r['checkpoint_sha256']
m={('overall',k.replace('recall_at_','hit_at_')):v for k,v in replay['metrics'].items()};m.update({('overall','queries'):2400,('overall','candidates'):100})
m.update({('historical_training',k.replace('recall_at_','hit_at_')):v for k,v in r['student'].items()})
stage(d,'Sim',m,r['checkpoint_sha256'],'same_pt_packaged_replay','补齐同PT A100回放2400条原保存排名；主列.8045833对应这批明细，历史训练.8058333另列，不混作同一轮。TEST选模，开发指标。',[p,rp,pr])
d['headers']=['stage','query_id','query_SKU','top1_candidate_id','top1_candidate_SKU','first_positive_rank','H@1','H@5','H@10','reciprocal_rank','average_precision','positive_count']
ep=SR/'lab/e877fe612a_metrics.json';npz=SR/'lab/10bbaa9b09_embeddings.npz';obsolete=SR/'lab/6c5986c59a_UPSTREAM_RECAPTURE_REQUIRED.json';e=read(ep);a=np.load(npz,allow_pickle=False)
assert e['checkpoint_sha256']==r['checkpoint_sha256']
testp=T/'t08_abo_image_text_retrieval/reports/text_to_image_physical_statistics_20261015/raw/test.csv';titlep=testp.with_name('titles.csv');test={x['sample_id']:x for x in csvread(testp)};titles={int(x['label']):x for x in csvread(titlep)}
source(d,testp);source(d,titlep)
for x in csvread(pr):
    sid=x['sample_id'];rank=int(x['true_rank']);label=int(x['predicted_label'])
    assert int(test[sid]['label'])==int(x['true_label']) and test[sid]['product_id']==x['product_id']
    d['samples'].append(['Sim',sid,x['product_id'],str(label),titles[label]['product_id'],rank,int(rank<=1),int(rank<=5),int(rank<=10),1/rank,1/rank,1])
sim_mrr=mean(x[9] for x in d['samples']);m[('overall','map')]=sim_mrr;assert abs(sim_mrr-m['overall','mrr'])<1e-12
d['stages']['Sim']['metrics']=[[g,k,v] for (g,k),v in m.items()]
q=a['queries']/np.linalg.norm(a['queries'],axis=1,keepdims=True);t=a['titles']/np.linalg.norm(a['titles'],axis=1,keepdims=True)
order=np.argsort(-(q@t.T),axis=1);ranks=np.argmax(order==a['labels'][:,None],axis=1)+1
assert len(e['predictions'])==len(ranks)==2400
for i,x in enumerate(e['predictions']):
    sid=x['sample_id'].removeprefix('image_');rank=int(ranks[i]);assert rank==x['true_rank'] and int(order[i,0])==x['predicted_label'] and int(a['labels'][i])==int(test[sid]['label'])==x['true_label']
    d['samples'].append(['Exp',sid,test[sid]['product_id'],str(x['predicted_label']),titles[x['predicted_label']]['product_id'],rank,int(rank<=1),int(rank<=5),int(rank<=10),1/rank,1/rank,1])
em={('overall','queries'):2400,('overall','candidates'):100}
for col,key in [(6,'hit_at_1'),(7,'hit_at_5'),(8,'hit_at_10'),(9,'mrr'),(10,'map')]:em[('overall',key)]=mean(x[col] for x in d['samples'] if x[0]=='Exp')
em.update({('overall','mean_rank'):float(ranks.mean()),('overall','median_rank'):float(np.median(ranks))})
em.update({('original_reported_summary',k.replace('recall_at_','hit_at_')):v for k,v in e['metrics'].items()})
assert sum(x[6] for x in d['samples'] if x[0]=='Exp')==1927
stage(d,'Exp',em,e['checkpoint_sha256'],'verified_corrected_saved_summary','用户确认按保存2400条及embedding复算将.8019更正为.8029；原汇总另组保留。原报告production_qualified=false、旧会话old_metrics_not_current=true仍披露，不称本次已重验光路。',[ep,npz,obsolete])
check(d,'Exp','embedding_ranks_equal_saved_predictions_all_2400',True);check(d,'Exp','original_summary_vs_actual_hit1',{'reported':e['metrics']['recall_at_1'],'recomputed':1927/2400,'correct':1927,'total':2400})
assert {x[1]:x[2] for x in d['samples'] if x[0]=='Sim'}=={x[1]:x[2] for x in d['samples'] if x[0]=='Exp'}
check(d,'pair','same_2400_query_ids_SKUs_and_checkpoint',True)

# Identity takes precedence over completeness: neither a later same-PT replay
# nor a superseded capture can replace the specifically requested final result.
d['notes']='指定Sim恢复原最终报告1934/2400=.8058333（.8058），epoch25 EMA；.8045833回放仅作参考。Exp的.8029167来自已标记过时的旧会话，不能确认是最终版，主列不采用；两批2400条参考明细完整保留并独立标识。'
d['expected']['Exp']=.8019
d['superseded_session_corrected_value']=1927/2400
sm={(g,k):v for g,k,v in d['stages']['Sim']['metrics']}
formal={('overall','queries'):2400,('overall','candidates'):100}
formal.update({('overall',k.replace('recall_at_','hit_at_')):v for k,v in r['student'].items()})
formal['overall','map']=r['student']['mrr']  # One positive per query: AP=RR.
formal.update({('same_pt_replay_reference',k):v for (g,k),v in sm.items() if g=='overall'})
d['stages']['Sim'].update(metrics=[[g,k,v] for (g,k),v in formal.items()],status='original_final_report',notes='指定原最终报告：epoch25 EMA，1934/2400=.8058333333，PT a2aa9a93。原1934命中这一轮的逐样本排名尚未恢复；1931命中的A100回放不替代指定版本。',formal_export_eligible=True)
em={(('superseded_session_reference' if g=='overall' else g),k):v for g,k,v in d['stages']['Exp']['metrics']}
d['stages']['Exp'].update(metrics=[[g,k,v] for (g,k),v in em.items()],status='final_identity_unconfirmed',notes='仅找回已被替代旧会话：原会话明确old_metrics_not_current=true，vision_expert重采后下游四层作废。旧数据复算1927/2400=.8029167只证明该旧文件内部计数，不证明最新/最终实验身份。最终Exp主指标待核，不以旧会话补齐。',formal_export_eligible=False)
d['stages']['Exp']['reference_checkpoint_sha256']=d['stages']['Exp']['checkpoint_sha256']
d['stages']['Exp']['checkpoint_sha256']=None  # The final Exp checkpoint is not established.
d['reference_sample_bindings']={
    'Sim':{'row_label':'Sim回放参考（非指定版）','metric_group':'same_pt_replay_reference','rows':2400},
    'Exp':{'row_label':'Exp旧会话参考（非最终）','metric_group':'superseded_session_reference','rows':2400}}
for row in d['samples']:row[0]=d['reference_sample_bindings'][row[0]]['row_label']
for name in ['FINAL_VERIFICATION.json']:
    source(d,p.parent/name)
for name in ['i2t_final_identity_full_20261026.inventory.json','4246fd452c_00_READ_ME.md','641f7946ff_02_summary.json']:
    source(d,SR/'lab'/name)
check(d,'Sim','original_final_report_identity',{'best_epoch':25,'weight_variant':'ema','correct':1934,'total':2400,'reference_replay_not_substituted':True})
check(d,'Exp','final_identity_not_established',{'old_metrics_not_current':True,'new_final_report_located':False,'superseded_reference_only':True})

# 4. Image-to-image: screenshot is not a same-checkpoint pair.
d=new('abo_image_to_image','ABO 图搜图','hit_at_1',[.8225,.8113],'从原2400条全量特征补齐两侧完整排名/AP，并与800查询Top-1逐条一致。Sim .8225不含RGB旁路；Exp rank72 .81125，其同PT仿真.83375，当前两列不是同权重对照。')
ip=T/'t07_abo_image_retrieval/runs/simulation/a100_input_inclusive_20261024_01/ours_repeat01/identity.json'
vp=ROOT/'LightGenV2/reports/20260928_nine_task_a100_table/version_audit_20261023/version_and_cost_audit_20261023.json'
v=next(x for x in read(vp)['rows'] if x['key']=='abo_image_to_image')
sp=SR/'server/7cdbe41d76_final_report.json';spr=SR/'server/34ee7a8f2d_predictions.csv';s=read(sp)
assert s['best_sha256']==v['sim_checkpoint_sha256']
sm=flat(s['normal']['test']);sm.update({('overall','queries'):800,('overall','candidates'):1600})
stage(d,'Sim',sm,s['best_sha256'],'verified_full_queries','800条原Top-1与保存特征完整排序逐条一致；已补MRR/完整MAP，原MAP@10单独保留，不混淆截断与完整AP。',[sp,spr,ip])
ep=ROOT/'handoffs/abo_latestfresh35_lab_20260930/rank72_physical_full_report_20261001.json';e=read(ep)
em={('overall',k.replace('recall_at_','hit_at_')):val for k,val in e['metrics'].items()};em.update({('overall','queries'):800,('overall','candidates'):1600})
pp=T/'t07_abo_image_retrieval/reports/physical_qwen_statistics_20261014/raw/dataset_manifest.csv'
parent={r['sample_id']:r for r in csvread(pp)};source(d,pp)
d['headers']=['stage','query_id','query_SKU','top1_candidate_id','top1_candidate_SKU','first_positive_rank','H@1','H@5','H@10','reciprocal_rank','average_precision','positive_count','category','positive_ranks','average_precision_at_10','record_origin']
simrows=csvread(spr);assert len(simrows)==800
query_ids={x['sample_id'] for x in simrows};rankings={}
for label,fp in [('Sim',SR/'server/345188cc50_features.pt'),('Exp',SR/'lab/39328b0e5a_physical_features.pt')]:
    source(d,fp);saved=saved_tensors(fp);ids=saved['ids'];assert set(ids)==set(parent) and len(ids)==2400
    assert saved['vectors'].shape==(2400,64) and np.isfinite(saved['vectors']).all()
    if label=='Sim':assert saved['manifest_sha256']==s['manifest_sha256']
    gallery=[i for i,sid in enumerate(ids) if sid not in query_ids];queries=[i for i,sid in enumerate(ids) if sid in query_ids]
    assert len(gallery)==1600 and len(queries)==800
    # The parent CSV's split column is older. Use the actual 800 enrolled
    # query IDs and their 1600-image complement, not that old split column.
    order,first,aps,ap10,positives=full_rankings(saved['vectors'][queries],saved['vectors'][gallery],[parent[ids[i]]['product_id'] for i in queries],[parent[ids[i]]['product_id'] for i in gallery])
    assert all(len(p)==8 for p in positives)
    rankings[label]={ids[q]:dict(top=ids[gallery[o[0]]],first=int(f),ap=float(ap),ap10=float(a10),positives=json.dumps(pos)) for q,o,f,ap,a10,pos in zip(queries,order,first,aps,ap10,positives)}
    target=sm if label=='Sim' else em
    target['overall','mrr']=float(np.mean(1/first));target['overall','map']=float(aps.mean());target['overall','map_at_10']=float(ap10.mean())
    for cutoff in (1,5,10):
        hit_counts=np.array([sum(p<=cutoff for p in positions) for positions in positives])
        target['overall','precision_at_'+str(cutoff)]=float(np.mean(hit_counts/cutoff))
        target['overall','positive_recall_at_'+str(cutoff)]=float(np.mean(hit_counts/8))
        target['overall','r_at_'+str(cutoff)]=float(np.mean(hit_counts>0))
    target['overall','query_count']=800;target['overall','candidate_count']=1600;target['overall','relevant_candidates_per_query']=8
    ideal=sum(1/math.log2(p+1) for p in range(1,9))
    target['overall','ndcg_at_10']=mean(sum(1/math.log2(p+1) for p in positions if p<=10)/ideal for positions in positives)
    if label=='Sim':assert abs(target['overall','map_at_10']-s['normal']['test']['map_at_10'])<1e-12
    else:assert abs(target['overall','mrr']-e['metrics']['mrr'])<1e-12
for x in simrows:
    sid=x['sample_id'];assert parent[sid]['product_id']==x['product_id'] and parent[x['top1_sample_id']]['product_id']==x['top1_product_id']
    full=rankings['Sim'][sid];rank=full['first'];assert full['top']==x['top1_sample_id'] and int(rank==1)==int(x['hit_at_1'])
    d['samples'].append(['Sim',sid,x['product_id'],x['top1_sample_id'],x['top1_product_id'],rank,int(rank<=1),int(rank<=5),int(rank<=10),1/rank,full['ap'],8,parent[sid]['class_name'],full['positives'],full['ap10'],'原保存仿真特征重排'])
assert sum(x[6] for x in d['samples'])==658
groups={}
for x in e['predictions']:
    rank=x['rank_of_first_relevant'];sid=x['sample_id'];category=parent[sid]['class_name'];groups.setdefault(category,[]).append(int(rank<=1))
    assert x['product_id']==parent[sid]['product_id'] and x['top1_product_id']==parent[x['top1_sample_id']]['product_id']
    full=rankings['Exp'][sid];assert full['top']==x['top1_sample_id'] and rank==full['first']
    d['samples'].append(['Exp',sid,x['product_id'],x['top1_sample_id'],x['top1_product_id'],rank,int(rank<=1),int(rank<=5),int(rank<=10),1/rank,full['ap'],8,category,full['positives'],full['ap10'],'原实测特征重排'])
category_order=[('vase','Vases'),('home mirror','Mirrors'),('bed','Beds'),('pillow','Pillows'),('stool seating','Stools'),('light fixture','Light fixtures'),('wall art','Wall art'),('sofa','Sofas'),('rug','Rugs'),('chair','Chairs')]
assert set(groups)=={c for c,_ in category_order}
for c,name in category_order:
    hits=groups[c];em.update({('category/'+name,'queries'):len(hits),('category/'+name,'hit_at_1'):mean(hits)})
for label,metrics in [('Sim',sm),('Exp',em)]:
    for c,name in category_order:
        group=[x for x in d['samples'] if x[0]==label and x[12]==c];metrics['category/'+name,'queries']=len(group)
        for column,key in [(6,'hit_at_1'),(7,'hit_at_5'),(8,'hit_at_10'),(9,'mrr'),(10,'map'),(14,'map_at_10')]:metrics['category/'+name,key]=mean(x[column] for x in group)
d['stages']['Sim']['metrics']=[[g,k,val] for (g,k),val in sm.items()]
stage(d,'Exp',em,e['checkpoint_sha256'],'verified_full_queries','649/800=.81125；原物理特征2400×64完整排序，800条首正例及Top-1全部吻合，已补完整MAP和每查询8个正例位置。',[ep])
assert abs(mean(x[6] for x in d['samples'] if x[0]=='Exp')-.81125)<1e-12

# 5. Text-to-image: simulation and adapted measured-feature readout.
d=new('abo_text_to_image','ABO 文搜图','hit_at_1',[.8600,.8500],'100个标题、2400候选、24正例；Sim从原保存FP16 embedding补全AP，100条首正例排名不变，AP均值相对原FP32报告差4.3e-6，分别保留。Exp为原实测特征上固定读出头CPU回放。')
b=ROOT/'handoffs/abo_text_to_image_10cm_alpha040_20260925';sp=b/'final_report.json';srp=b/'text_to_image_predictions.csv';r=read(sp)
replay=T/'t08_abo_image_text_retrieval/reports/text_to_image_physical_statistics_20261015/raw/ours_cpu_replay.json';e=read(replay)
d['headers']=['stage','query_id','query_SKU','top1_candidate_id','top1_candidate_SKU','first_positive_rank','H@1','H@5','H@10','reciprocal_rank','average_precision','positive_count','title','positive_ranks','record_origin']
fp=SR/'server/cadb214283_student_embeddings.pt';source(d,fp);saved=saved_tensors(fp);testrows=csvread(testp)
assert saved['titles'].dtype==saved['query'].dtype==np.float16
order,first,aps,ap10,positives=full_rankings(saved['titles'],saved['query'],range(100),[int(x['label']) for x in testrows]);assert all(len(p)==24 for p in positives)
for x in csvread(srp):
    label=int(x['title_label']);rank=int(x['first_positive_rank']);top=json.loads(x['top10_sample_ids'])[0];assert rank==int(first[label]) and testrows[order[label,0]]['sample_id']==top
    d['samples'].append(['Sim',str(label),x['product_id'],top,test[top]['product_id'],rank,int(rank<=1),int(rank<=5),int(rank<=10),1/rank,float(aps[label]),24,x['title'],json.dumps(positives[label]),'保存FP16嵌入重排'])
for x in e['predictions']:
    rank=x['first_positive_rank'];d['samples'].append(['Exp',str(x['title_label']),x['product_id'],x['top10_sample_ids'][0],x['top10_product_ids'][0],rank,int(rank<=1),int(rank<=5),int(rank<=10),x['reciprocal_rank'],x['average_precision'],24,x['title'],json.dumps(x['positive_ranks']),'固定读出头CPU回放'])
for label,m,checkpoint,files,note in [('Sim',r['student'],e['body_sha256'],[sp,srp],'Sim首正例/Top-1均与原100查询相同；AP从同次保存FP16 embedding计算，主MAP与这些逐条AP一致，另列原FP32 MAP，不冒称恢复了FP32尾部全部排名。'),('Exp',e['metrics'],e['readout_sha256'],[replay,b/'finetune10/finetune_report.json'],'冻结实测特征CPU重放；读出头TRAIN600/VAL200选择epoch9；未重新验证CCD到特征。')]:
    m={**m,'queries':100,'candidates':2400,'query_count':100,'gallery_count':2400,'relevant_images_per_query':24}
    metrics=flat(m)
    if label=='Sim':
        metrics.update(flat(m,'historical_FP32_summary'));metrics['overall','map']=float(aps.mean())
    stage_rows=[row for row in d['samples'] if row[0]==label]
    for cutoff in (1,5,10):
        hits=[sum(p<=cutoff for p in json.loads(row[13])) for row in stage_rows]
        value=mean(n/24 for n in hits)
        metrics['overall','recall_at_'+str(cutoff)]=value
        metrics['overall','positive_recall_at_'+str(cutoff)]=value
        metrics['overall','precision_at_'+str(cutoff)]=mean(n/cutoff for n in hits)
    stage(d,label,metrics,checkpoint,'verified_saved_predictions',note,files)
    rows=[x for x in d['samples'] if x[0]==label];assert len(rows)==100 and abs(mean(x[6] for x in rows)-m['hit_at_1'])<1e-12
assert {r[1]:r[2] for r in d['samples'] if r[0]=='Sim'}=={r[1]:r[2] for r in d['samples'] if r[0]=='Exp'}

# 6. Pose: grouped PCK from native counts; derive per-image Exp metrics from GT.
d=new('lsp','LSP 关键点','pck_at_0.2_torso',[.7348,.7316],'已查到用户在LSP对话2026-10-23确认采用Exp last30=.7316。Sim为legacy staged e50=.7348；Exp底座另经有界振幅重训至e15，再以TRAIN100适配原读出头30轮，非只在旧Sim上换设备。')
b=T/'t02_keypoint_detection/runs/hardware';sp=b/'lsp_staged7348_20261008/units_continuous_audit.json';ep=b/'lsp_bounded_e15_train100_20261009/head_adapt30/report.json';s=read(sp);e=read(ep)
joint_groups={'Overall (14 joints)':None,'Shoulder':['right_shoulder','left_shoulder'],'Elbow':['right_elbow','left_elbow'],'Wrist':['right_wrist','left_wrist'],'Hip':['right_hip','left_hip'],'Knee':['right_knee','left_knee'],'Ankle':['right_ankle','left_ankle']}
for label,m,checkpoint,p,status in [('Sim',s['metrics'],s['checkpoint_sha256'],sp,'verified_full_test'),('Exp',e['strict_cpu']['last'],e['last_sha256'],ep,'verified_user_selected_adapted')]:
    metrics=flat(m)
    for group,joints in joint_groups.items():
        if joints is None:val=m['pck_at_0.2_torso'];count=m['pck_evaluated_joints']
        else:
            count=sum(m['per_joint_counts'][j] for j in joints);val=sum(m['per_joint_pck_at_0.2_torso'][j]*m['per_joint_counts'][j] for j in joints)/count
        metrics[('PCK group/'+group,'pck_at_0.2_torso')]=val;metrics[('PCK group/'+group,'evaluated_joints')]=count
    stage(d,label,metrics,checkpoint,status,'PCK两侧关节按有效计数加权；Overall含14关节。Sim原14000关节预测已找回；Exp有用户明确选版确认。两版本训练历史不同。',[p])
saved=b/'lsp_bounded_e15_train100_20261009/head_adapt30/last_samples.json';source(d,saved)
annotation=T/'t02_keypoint_detection/runs/smoke/lsp_bounded_asset_transfer_20261008/extracted/2026OpticsMoE/data/lsp_pose/lsp_dataset/joints.mat';source(d,annotation)
a=np.asarray(loadmat(annotation)['joints'],dtype=np.float32).transpose(2,1,0)[:,:,0:2]-np.float32(1)
joint_names=list(e['strict_cpu']['last']['per_joint_counts']);d['headers']=['stage','sample_id','PCK@0.2_torso','PCKh@0.5_head','mean_pixel_error_px','normalized_mean_error_torso','visible_joints','torso_scale_px','head_scale_px']+[j+'_error_px' for j in joint_names]+[j+'_PCK_hit' for j in joint_names]
for record in read(saved):
    idx=int(record['sample_id'].split('_')[1])-1;xy=a[idx].copy();center=(xy.min(0)+xy.max(0))*.5
    side=max(float(np.ptp(xy[:,0])),float(np.ptp(xy[:,1])),32.)*1.25
    left=math.floor(float(center[0])-side*.5);top=math.floor(float(center[1])-side*.5)
    sx=np.float32(224/(math.ceil(float(center[0])+side*.5)-left));sy=np.float32(224/(math.ceil(float(center[1])+side*.5)-top))
    xy[:,0]=(xy[:,0]-left)*sx;xy[:,1]=(xy[:,1]-top)*sy
    visible=np.isfinite(xy).all(1)&(xy[:,0]>=0)&(xy[:,0]<224)&(xy[:,1]>=0)&(xy[:,1]<224)
    torso=np.float32((float(np.linalg.norm(xy[8]-xy[3]))+float(np.linalg.norm(xy[9]-xy[2])))/2);head=np.float32(2*float(np.linalg.norm(xy[12]-xy[13])))
    pred=np.asarray(record['coordinates'][0],dtype=np.float32);error=np.linalg.norm(pred-xy,axis=1);hit=error<=np.float32(.2)*torso;hhit=error<=np.float32(.5)*head
    d['samples'].append(['Exp',record['sample_id'],float(hit[visible].mean()),float(hhit[visible].mean()),float(error[visible].mean()),float((error[visible]/torso).mean()),int(visible.sum()),float(torso),float(head)]+[float(v) if ok else None for v,ok in zip(error,visible)]+[int(v) if ok else None for v,ok in zip(hit,visible)])
for col,key in [(2,'pck_at_0.2_torso'),(3,'pckh_at_0.5_head'),(4,'mean_pixel_error'),(5,'normalized_mean_error_torso')]:
    value=sum(row[col]*row[6] for row in d['samples'])/sum(row[6] for row in d['samples'])
    assert abs(value-e['strict_cpu']['last'][key])<3e-6,(key,value,e['strict_cpu']['last'][key]);check(d,'Exp','offline_from_saved_coordinates/'+key,value)
sim_prediction=SR/'server/1a22f68dbe_selected_checkpoint_test_predictions.csv';sim_report=SR/'server/eae0bd914f_selected_checkpoint_test_evaluation.json';source(d,sim_prediction);source(d,sim_report)
observations={}
for row in csvread(sim_prediction):observations.setdefault(row['sample_id'],[]).append(row)
assert len(observations)==1000 and read(sim_report)['checkpoint_sha256']==s['checkpoint_sha256']
for sid,joints in observations.items():
    joints.sort(key=lambda x:int(x['joint_index']));assert [r['joint_name'] for r in joints]==joint_names
    xy=np.asarray([[float(r['true_x']),float(r['true_y'])] for r in joints],dtype=np.float32);error=np.asarray([float(r['pixel_error']) for r in joints],dtype=np.float32);visible=np.asarray([r['visible'].lower()=='true' for r in joints])
    torso=np.float32((float(np.linalg.norm(xy[8]-xy[3]))+float(np.linalg.norm(xy[9]-xy[2])))/2);head=np.float32(2*float(np.linalg.norm(xy[12]-xy[13])));hit=error<=np.float32(.2)*torso;hhit=error<=np.float32(.5)*head
    d['samples'].append(['Sim',sid,float(hit[visible].mean()),float(hhit[visible].mean()),float(error[visible].mean()),float((error[visible]/torso).mean()),int(visible.sum()),float(torso),float(head)]+[float(v) if ok else None for v,ok in zip(error,visible)]+[int(v) if ok else None for v,ok in zip(hit,visible)])
for col,key in [(2,'pck_at_0.2_torso'),(3,'pckh_at_0.5_head'),(4,'mean_pixel_error'),(5,'normalized_mean_error_torso')]:
    rows=[r for r in d['samples'] if r[0]=='Sim'];value=sum(row[col]*row[6] for row in rows)/sum(row[6] for row in rows)
    assert abs(value-s['metrics'][key])<3e-6,(key,value,s['metrics'][key]);check(d,'Sim','saved_joint_csv/'+key,value)

# 7. Saliency: retain original CC; fill other metrics from frozen CCD replay.
d=new('salicon','SALICON 显著性','cc',[.8625,.8598],'用户确认Exp采用原CC=.8597739692，即.8598。每侧5000图：原逐图CC保留，MAE/KLD/NSS/SIM/AUC由原冻结模型、原输入缓存及原三层CCD的CPU重放补齐；另存重放CC用于核对，不以均值填充。')
d['original_screenshot_exp']=.8599
b=T/'t03_saliency/runs';sp=b/'simulation/moe_alpha40_sam_batch8_crosssample_20260913_seed42/candidate_selected_evaluation/selected_checkpoint_test_evaluation.json';ep=b/'hardware/salicon_08625_shs_20260914/full01_retry01/results.json';s=read(sp);e=read(ep)
d['headers']=['stage','sample_id','MAE','KLD','CC','NSS','SIM','AUC_Judd','CC_CPU_replay','record_origin']
replayed=SR/'derived/salicon_native_cpu_full5000.json';derived=read(replayed)
assert derived['complete'] and derived['aggregate_validation_passed'] and derived['checkpoint_sha256']==s['verified_checkpoint_sha256']
assert derived['completed_samples']==5000 and not derived['training'] and not derived['hardware_called'] and not derived['remote_writes']
derived_rows={(r['stage'],r['sample_id']):r for r in derived['rows']};assert len(derived_rows)==10000
source(d,replayed)
for label,m,p,status in [('Sim',s['metrics'],sp,'verified_full_test'),('Exp',e['metrics'],ep,'verified_full_test')]:
    metrics=flat(m);metrics.update(flat(derived['metrics'][label],'CPU_replay_validation'))
    stage(d,label,metrics,s['verified_checkpoint_sha256'],status,'同PT、同5000输入和逐帧SHA；原逐图CC保留，其他逐图指标为原始缓存/CCD的冻结CPU补算。重放整体另组保留，允许设备浮点末位差异。',[p,replayed])
    for row in e['rows']:
        r=derived_rows[label,row['sample_id']];original_cc=row['simulation_cc' if label=='Sim' else 'cc_float64'];assert abs(r['cc_float64']-original_cc)<3e-4
        d['samples'].append([label,row['sample_id'],r['mae'],r['kld'],original_cc,r['nss'],r['sim'],r['auc_judd'],r['cc_float64'],'原CC；其余指标CPU重放'])
    cc=mean(x[4] for x in d['samples'] if x[0]==label);check(d,label,'mean_saved_float64_CC',cc)
    assert abs(cc-m['cc'])<1e-6
    for col,key in [(2,'mae'),(3,'kld'),(5,'nss'),(6,'sim'),(7,'auc_judd')]:
        value=mean(x[col] for x in d['samples'] if x[0]==label);assert abs(value-derived['metrics'][label][key])<1e-6
        assert abs(value-m[key])<3e-5,(label,key,'CPU replay differs from historical aggregate',value,m[key])
        check(d,label,'mean_CPU_replayed/'+key,value)

# 8. OpenMoji: recover predictions and all eight metrics from original CCDs.
d=new('openmoji','OpenMoji 语义交互','changed_cell_accuracy',[.8765,.8655],'同一e45 PT/1000 TEST清单。Exp逐例最终网格、编辑掩膜及8指标已用原6000张CCD离线CPU重放补齐；重放整体/四操作与原报告核对，不训练、不重拍、不接设备。')
b=T/'t04_semantic_interaction';sp=b/'runs/simulation/layered_dc30_ccdsmall_selected_e45_s73_20260925/audit.json';pp=sp.parent/'normal_predictions.jsonl';ep=b/'runs/hardware/lab08655_qwen08120_comparison_20261015/ours_lab_full_report.json';s=read(sp);e=read(ep)
replayed=SR/'derived/openmoji_native_cpu_full1000.json';derived=read(replayed)
assert derived['complete'] and derived['aggregate_validation_passed'] and derived['checkpoint_sha256']==e['checkpoint_sha256']
assert derived['completed_samples']==1000 and not derived['training'] and not derived['hardware_called'] and not derived['remote_writes']
source(d,replayed)
for label,metrics,p,status in [('Sim',s['normal'],sp,'verified_full_test'),('Exp',e['metrics'],ep,'verified_frozen_CCD_replay')]:
    fm={}
    for group in ['overall','add','replace','move','remove']:fm.update(flat(metrics[group],group))
    stage(d,label,fm,e['checkpoint_sha256'],status,'四操作各250；Accuracy指changed-cell accuracy。Exp逐例从原CCD/同PT补算，8项整体和分操作均与原报告核对。',[p])
rows=[json.loads(x) for x in pp.read_text(encoding='utf-8').splitlines() if x.strip()];source(d,pp)
keys=list(s['normal']['overall']);keys=[k for k in keys if k!='samples'];d['headers']=['stage','sample_id','task','instruction']+keys+['record_status']
d['samples']=[['Sim',r['sample_id'],r['task'],r['instruction']]+[r[k] for k in keys]+['原逐例指标'] for r in rows]
assert len(rows)==1000 and abs(mean(r['changed_cell_accuracy'] for r in rows)-.8765)<1e-12
manifest=SR/'lab/295296b2f7_test.jsonl';contract=SR/'lab/71cc244b17_run_contract.json';exp_original=SR/'lab/652794bb4f_report.json';c=read(contract)
assert sha(manifest)==c['test_manifest_sha256']=='93aa528c61fb7da116dc0bb7e749f7f3e525d1140050982e33604e3e72d22fca' and c['checkpoint_sha256']==e['checkpoint_sha256']
for p in [manifest,contract,exp_original]:source(d,p)
testrows=[json.loads(x) for x in manifest.read_text(encoding='utf-8').splitlines()];assert {r['sample_id']:(r['task'],r['instruction']) for r in testrows}=={r['sample_id']:(r['task'],r['instruction']) for r in rows}
byid={r['sample_id']:r for r in derived['rows']};assert len(byid)==1000
for r in testrows:
    replay_row=byid[r['sample_id']];assert (replay_row['task'],replay_row['instruction'])==(r['task'],r['instruction'])
    d['samples'].append(['Exp',r['sample_id'],r['task'],r['instruction']]+[replay_row[k] for k in keys]+['原CCD固定权重CPU重放'])
check(d,'pair','same_TEST_manifest_SHA_and_sample_ids_instructions',True)

# 9. Conditional editing: paired IDs/targets, but adapted decoder PT differs.
d=new('abo_conditional_edit','ABO 条件图像编辑','psnr_db',[34.28,31.55],'2304 TEST同一sample/source/mode，Sim与Exp是不同PT；实验适配解码器。原生256像素float[0,1]逐图PSNR后取均值；不是8bit PNG指标。')
b=ROOT/'handoffs/t12_four_group_summary_20260927';d['headers']=['stage','sample_id','mode','source_id','category','prompt','MSE_0_1','MAE_0_1','PSNR_dB','SSIM']
ids={}
for label,folder in [('Sim','small_sim'),('Exp','small_exp')]:
    rp=b/folder/'report.json';pp=b/folder/'sample_metrics.csv';r=read(rp);rows=csvread(pp);ids[label]={x['sample_id']:(x['source_id'],x['mode']) for x in rows}
    fm=flat({k:r[k] for k in ['samples','mse_0_1','mae_0_1','psnr_db','ssim']})
    for mode in ['background','joint','object']:
        rr=[x for x in rows if x['mode']==mode];fm[('mode/'+mode,'samples')]=len(rr)
        for metric in ['mse_0_1','mae_0_1','psnr_db','ssim']:fm[('mode/'+mode,metric)]=mean(float(x[metric]) for x in rr)
    for x in rows:d['samples'].append([label,x['sample_id'],x['mode'],x['source_id'],x['category'],x['prompt']]+[float(x[k]) for k in ['mse_0_1','mae_0_1','psnr_db','ssim']])
    stage(d,label,fm,r['checkpoint_sha256'],'verified_saved_metrics','原报告、原逐样本CSV；不同解码器权重。',[rp,pp])
    assert len(rows)==2304 and abs(mean(float(x['psnr_db']) for x in rows)-r['psnr_db'])<1e-10
assert ids['Sim']==ids['Exp'];check(d,'pair','same_sample_source_mode',True)

# Explicit absence, not inference or zero imputation.
REQUIRED={
'lgvq_temporal':['count','srcc','krcc','plcc','rmse','mae'],'lgvq_spatial':['count','srcc','krcc','plcc','rmse','mae'],
'abo_image_to_text':['queries','candidates','hit_at_1','hit_at_5','hit_at_10','mrr','map'],
'abo_image_to_image':['queries','candidates','hit_at_1','hit_at_5','hit_at_10','mrr','map'],
'abo_text_to_image':['queries','candidates','hit_at_1','hit_at_5','hit_at_10','mrr','map'],
'lsp':['pck_at_0.2_torso','pckh_at_0.5_head','mean_pixel_error','normalized_mean_error_torso'],
'salicon':['samples','kld','cc','sim','nss','auc_judd','mae'],
'openmoji':['samples','cell_accuracy','changed_cell_accuracy','foreground_category_accuracy','preserved_cell_accuracy','edit_grid_iou','object_f1','scene_exact_match','task_accuracy'],
'abo_conditional_edit':['samples','mse_0_1','mae_0_1','psnr_db','ssim']}
for i,d in enumerate(TASKS,1):
    d['filename']=f'{i:02d}_{d["name"].replace(" ","")}_LG_Sim_Exp.xlsx'
    d['required_metrics']=REQUIRED[d['task']]
    for label,s in d['stages'].items():
        values={(g,k):v for g,k,v in s['metrics']};s['primary_value']=values.get(('overall',d['primary']))
        s['sample_rows']=sum(row[0]==label for row in d['samples'])
        s['missing_overall_metrics']=[k for k in d['required_metrics'] if ('overall',k) not in values]
        expected=d['expected'][label];s['screenshot_numeric_rounding_matches']=(s['primary_value'] is not None and round(s['primary_value'],2 if d['primary']=='psnr_db' else 4)==expected)
        s['screenshot_lineage_confirmed']=s['status'] in ['original_final_report','verified_full_test','verified_mixed_test','verified_saved_predictions','verified_saved_metrics','verified_user_selected_adapted','verified_saved_top1','aggregate_with_verified_sample_identity','verified_full_queries','verified_corrected_saved_summary','verified_frozen_CCD_replay']
        primary_header={'changed_cell_accuracy':'changed_cell_accuracy','pck_at_0.2_torso':'PCK@0.2_torso','cc':'CC','psnr_db':'PSNR_dB','srcc':'predicted_score','hit_at_1':'H@1'}[d['primary']]
        s['sample_rows_with_primary_observation']=sum(row[0]==label and row[d['headers'].index(primary_header)] is not None for row in d['samples'])
    assert all(len(row)==len(d['headers']) for row in d['samples'])
    assert len({(row[0],row[1]) for row in d['samples']})==len(d['samples'])
payload=dict(date='2026-10-26',collector_runs_model_inference=False,contains_new_frozen_CPU_CCD_replays=True,no_training=True,no_hardware_access=True,
    source_root=str(ROOT),reference='LightGenV2/tasks/t19_resnet20_baselines/runs/outputs/t19_final_plot_data_20261024',
    blanks='Missing/unavailable, not zero. Screenshot target fields are not measured data.',tasks=TASKS)
(OUT/'plot_data.json').write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
with (OUT/'version_audit.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['task','stage','screenshot_target','archived_primary','status','samples_exported','checkpoint_sha256','notes'])
    for d in TASKS:
        for label,s in d['stages'].items():w.writerow([d['task'],label,d['expected'][label],s['primary_value'],s['status'],s['sample_rows'],s['checkpoint_sha256'],d['notes']+' '+s['notes']])
print(json.dumps({'workbooks':len(TASKS),'sample_records':sum(len(d['samples']) for d in TASKS),'tasks':[{'task':d['task'],'rows':len(d['samples']),'status':{l:s['status'] for l,s in d['stages'].items()}} for d in TASKS]},ensure_ascii=True))
