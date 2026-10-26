"""Independent ZIP/XML readback of all authored cells, without spreadsheet writers."""
import hashlib,json,math,zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
OUT=Path(__file__).resolve().parent
NS={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
data=json.loads((OUT/'plot_data.json').read_text(encoding='utf-8'))
audit=json.loads((OUT/'verification/workbook_export_audit.json').read_text(encoding='utf-8'))
def col(s):
    out=0
    for c in s:
        if not c.isalpha():break
        out=out*26+ord(c)-64
    return out-1
def cells(z,name):
    shared=[]
    if 'xl/sharedStrings.xml' in z.namelist():
        for si in ET.fromstring(z.read('xl/sharedStrings.xml')):
            shared.append(''.join(t.text or '' for t in si.iter('{'+NS['m']+'}t')))
    result={}
    for row in ET.fromstring(z.read(name)).findall('m:sheetData/m:row',NS):
        for c in row:
            typ=c.attrib.get('t');v=c.find('m:v',NS)
            if typ=='e':raise AssertionError(('Cell error',c.attrib))
            if typ=='s':value=shared[int(v.text)]
            elif typ=='str':value=v.text if v is not None else None
            elif typ=='inlineStr':value=''.join(t.text or '' for t in c.iter('{'+NS['m']+'}t'))
            elif v is not None and v.text is not None:value=float(v.text)
            else:value=None
            result[(int(row.attrib['r']),col(c.attrib['r']))]=value
    return result
def same(a,b):
    if b is None:return a is None
    if isinstance(b,(float,int)):return isinstance(a,(float,int)) and math.isclose(a,b,rel_tol=3e-15,abs_tol=1e-13)
    return a==b

def pearson(x,y):
    mx=sum(x)/len(x);my=sum(y)/len(y)
    dx=[v-mx for v in x];dy=[v-my for v in y]
    return sum(a*b for a,b in zip(dx,dy))/math.sqrt(sum(a*a for a in dx)*sum(b*b for b in dy))

def average_ranks(values):
    order=sorted(range(len(values)),key=values.__getitem__);ranks=[0.0]*len(values);i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]:j+=1
        for k in range(i,j):ranks[order[k]]=(i+j+1)/2
        i=j
    return ranks

def kendall_b(x,y):
    numerator=non_tied_x=non_tied_y=0
    for i in range(len(x)):
        for j in range(i):
            dx=(x[i]>x[j])-(x[i]<x[j]);dy=(y[i]>y[j])-(y[i]<y[j])
            numerator+=dx*dy;non_tied_x+=dx!=0;non_tied_y+=dy!=0
    return numerator/math.sqrt(non_tied_x*non_tied_y)
result=[];metric_checks=[]
for d,entry in zip(data['tasks'],audit['files']):
    path=OUT/d['filename'];assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
    with zipfile.ZipFile(path) as z:
        sheets=ET.fromstring(z.read('xl/workbook.xml')).find('m:sheets',NS)
        assert [s.attrib['name'] for s in sheets]==['整体指标','逐测试样本']
        a=cells(z,'xl/worksheets/sheet1.xml');b=cells(z,'xl/worksheets/sheet2.xml')
        for i,row in enumerate(entry['summary_values'],entry['summary_start_row']):
            for j,value in enumerate(row):assert same(a.get((i,j)),value),(d['task'],i,j,a.get((i,j)),value)
        exported_headers=d['plot_view']['headers']
        for j,h in enumerate(exported_headers):assert b[1,j]==h
        exported_rows=d.get('plot_view',{}).get('samples',d['samples'])
        for i,row in enumerate(exported_rows,2):
            for j,value in enumerate(row):assert same(b.get((i,j)),value),(d['task'],i,j,b.get((i,j)),value)
        if d.get('plot_view'):
            assert entry['source_sample_rows']==len(exported_rows)
            assert entry['archived_reference_rows']==len(d['samples'])-len(exported_rows)==d['plot_view']['excluded_reference_rows']
            assert entry['view']=='selected_plot_only' and d['plot_view']['selected_results_only']
            summary=entry['summary_values'];primary=next(row for row in summary if row[0]=='Overall' and row[1]==d['primary'])
            assert primary[2:4]==[d['stages'][s]['primary_value'] for s in ['Sim','Exp']]
            assert all(row[0] not in ['历史训练记录','原FP32汇总','固定权重CPU回放','同PT回放参考（非指定版）','旧会话参考（非最终）'] for row in summary)
            assert all(r[0] in ['Sim','Exp'] for r in exported_rows)
            assert all(h not in exported_headers for h in ['record_origin','record_status','CC_CPU_replay'])
            assert not any(isinstance(v,str) and ('Sim回放参考' in v or 'Exp旧会话参考' in v) for v in b.values())
        result.append({'task':d['task'],'file':d['filename'],'rows':len(exported_rows),'archived_rows_checked':len(d['samples']),'columns':len(exported_headers),'all_summary_cells_equal':True,'all_raw_cells_equal':True,'sha256':entry['sha256']})
    h={k:i for i,k in enumerate(d['headers'])}
    def check(stage,scope,key,value,expected,tolerance=1e-7):
        assert abs(value-expected)<=tolerance,(d['task'],stage,scope,key,value,expected)
        metric_checks.append({'task':d['task'],'stage':stage,'scope':scope,'metric':key,'recalculated':value,'archived':expected,'absolute_difference':abs(value-expected)})
    for stage,info in d['stages'].items():
        rows=[r for r in d['samples'] if r[0]==stage];metrics={(g,k):v for g,k,v in info['metrics']}
        if stage in d.get('reference_sample_bindings',{}):
            binding=d['reference_sample_bindings'][stage]
            assert not rows and info['sample_rows']==0,'Reference rows must not be attributed to the requested original run'
            rows=[r for r in d['samples'] if r[0]==binding['row_label']]
            assert len(rows)==binding['rows']
            metrics={('overall',k):v for (g,k),v in metrics.items() if g==binding['metric_group']}
            if d['task']=='abo_image_to_text':
                assert stage=='Sim' and len(rows)==2400
                original={(g,k):v for g,k,v in info['metrics']}
                assert original['overall','hit_at_1']==1934/2400
                assert metrics['overall','hit_at_1']==1931/2400
                assert info['formal_export_eligible'] and info['status']=='original_final_report'
            else:
                assert d['task']=='lgvq_spatial' and stage=='Sim' and len(rows)==558
                assert info['primary_value']==.6710079009479295 and metrics['overall','srcc']==.6710968960031272
        if d['task']=='abo_image_to_text' and stage=='Exp':
            assert info['checkpoint_sha256']==d['stages']['Sim']['checkpoint_sha256']
            assert info['primary_value']==1927/2400 and len(rows)==2400
            assert info['status']=='user_selected_verified_capture' and d['selection']['selected_by_user']
        if d['task']=='abo_image_to_text' and stage=='Sim':
            assert info['status']=='verified_frozen_CPU_replay' and len(rows)==2400
            assert info['historical_top1_reproduced'] and not info['historical_per_sample_recovered']
            assert sum(r[h['H@1']] for r in rows)==1934
            assert info['sample_provenance']=='frozen_original_model_CPU_inference_20261026'
        if not rows:continue
        if d['task'].startswith('lgvq'):
            targets=[r[h['MOS_target']] for r in rows];predictions=[r[h['predicted_score']] for r in rows]
            for key,value in [('srcc',pearson(average_ranks(targets),average_ranks(predictions))),('krcc',kendall_b(targets,predictions)),('plcc',pearson(targets,predictions))]:
                check(stage,'overall',key,value,metrics['overall',key],1e-12)
            for key,value in [('mae',sum(r[h['absolute_error']] for r in rows)/len(rows)),('rmse',math.sqrt(sum(r[h['squared_error']] for r in rows)/len(rows)))]:check(stage,'overall',key,value,metrics['overall',key],2e-5)
        elif d['task'].startswith('abo_') and d['task']!='abo_conditional_edit':
            for r in rows:
                rank=r[h['first_positive_rank']]
                assert rank>=1 and r[h['H@1']]==int(rank<=1) and r[h['H@5']]==int(rank<=5) and r[h['H@10']]==int(rank<=10)
                assert math.isclose(r[h['reciprocal_rank']],1/rank,abs_tol=1e-13)
                assert (r[h['query_SKU']]==r[h['top1_candidate_SKU']])==bool(r[h['H@1']])
                if 'positive_ranks' in h:
                    positive=json.loads(r[h['positive_ranks']]);assert positive==sorted(set(positive)) and len(positive)==r[h['positive_count']] and positive[0]==rank
                    ap=sum((j+1)/p for j,p in enumerate(positive))/len(positive)
                    assert math.isclose(ap,r[h['average_precision']],abs_tol=1e-13)
                    if 'average_precision_at_10' in h:
                        ap10=sum((j+1)/p for j,p in enumerate(positive) if p<=10)/min(len(positive),10)
                        assert math.isclose(ap10,r[h['average_precision_at_10']],abs_tol=1e-13)
            for column,key in [('H@1','hit_at_1'),('H@5','hit_at_5'),('H@10','hit_at_10'),('reciprocal_rank','mrr'),('average_precision','map')]:
                values=[r[h[column]] for r in rows]
                if ('overall',key) in metrics and all(v is not None for v in values):check(stage,'overall',key,sum(values)/len(values),metrics['overall',key])
        elif d['task']=='openmoji':
            if all(r[h['changed_cell_accuracy']] is None for r in rows):continue
            for (scope,key),expected in metrics.items():
                if key=='samples':continue
                group=rows if scope=='overall' else [r for r in rows if r[h['task']]==scope]
                check(stage,scope,key,sum(r[h[key]] for r in group)/len(group),expected)
        elif d['task']=='abo_conditional_edit':
            for (scope,key),expected in metrics.items():
                column={'mse_0_1':'MSE_0_1','mae_0_1':'MAE_0_1','psnr_db':'PSNR_dB','ssim':'SSIM'}.get(key)
                if not column:continue
                group=rows if scope=='overall' else [r for r in rows if r[h['mode']]==scope.split('/')[1]]
                check(stage,scope,key,sum(r[h[column]] for r in group)/len(group),expected,1e-10)
        elif d['task']=='salicon':
            check(stage,'overall','cc',sum(r[h['CC']] for r in rows)/len(rows),metrics['overall','cc'],1e-6)
            for column,key in [('MAE','mae'),('KLD','kld'),('NSS','nss'),('SIM','sim'),('AUC_Judd','auc_judd')]:
                assert all(r[h[column]] is not None for r in rows)
                check(stage,'CPU_replay_validation',key,sum(r[h[column]] for r in rows)/len(rows),metrics['CPU_replay_validation',key],1e-6)
        elif d['task']=='lsp':
            for column,key in [('PCK@0.2_torso','pck_at_0.2_torso'),('PCKh@0.5_head','pckh_at_0.5_head'),('mean_pixel_error_px','mean_pixel_error'),('normalized_mean_error_torso','normalized_mean_error_torso')]:check(stage,'overall',key,sum(r[h[column]]*r[h['visible_joints']] for r in rows)/sum(r[h['visible_joints']] for r in rows),metrics['overall',key],3e-6)
            for (scope,key),expected in metrics.items():
                if scope=='overall/per_joint_pck_at_0.2_torso':check(stage,scope,key,sum(r[h[key+'_PCK_hit']] for r in rows)/len(rows),expected)
coverage=[];unresolved=[]
for d in data['tasks']:
    for stage,info in d['stages'].items():
        if stage in d.get('reference_sample_bindings',{}):
            assert info['sample_rows_with_primary_observation']==info['sample_rows']==0
            assert not info['missing_overall_metrics']
            reference_count=d['reference_sample_bindings'][stage]['rows']
            unresolved.append({'task':d['task'],'stage':stage,'reason':'Original selected-run per-sample observations not recovered; aggregate retained','reference_rows_retained_outside_workbook':reference_count})
            coverage.append({'task':d['task'],'stage':stage,'sample_rows':0,'primary_rows':0,'missing_requested_overall_metrics':[],'reference_only_rows':reference_count,'final_version_complete':False})
            continue
        assert info['sample_rows_with_primary_observation']==info['sample_rows']>0,(d['task'],stage,'Missing primary observation')
        assert not info['missing_overall_metrics'],(d['task'],stage,info['missing_overall_metrics'])
        coverage.append({'task':d['task'],'stage':stage,'sample_rows':info['sample_rows'],'primary_rows':info['sample_rows_with_primary_observation'],'missing_requested_overall_metrics':[],'provenance':info.get('sample_provenance',info['status'])})
(OUT/'verification/saved_xml_verification.json').write_text(json.dumps({'workbooks':len(result),'all_passed':True,'all_requested_versions_complete':not unresolved,'unresolved_original_versions':unresolved,'sample_records':sum(r['rows'] for r in result),'archive_records_checked':sum(r['archived_rows_checked'] for r in result),'files':result,'metric_recalculations':metric_checks,'coverage':coverage},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'workbooks':len(result),'all_passed':True,'all_requested_versions_complete':not unresolved,'unresolved_original_versions':unresolved,'sample_records':sum(r['rows'] for r in result),'archive_records_checked':sum(r['archived_rows_checked'] for r in result)}))
