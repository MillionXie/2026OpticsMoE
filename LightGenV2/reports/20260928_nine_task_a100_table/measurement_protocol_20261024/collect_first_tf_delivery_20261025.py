"""File-only collection of the authorized first-optics/first-block comparison."""
import csv,json,hashlib,sys
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
BASE=HERE/'outputs/01a061e2-6f88-7913-87f5-b632000e4c83'
OUT=BASE/'first_tf_complete_20261025_01'
SEALED=BASE/'complete_20261025_01/collected_data_and_audit.json'
from first_tf_supplement_20261025 import integrate
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def collect(allow_pending=False):
    assert sha(SEALED)=='f0cd430b99aae15c69ddbeeeee1aa2d2b1334a6e804d7a91097a79c6659d2ffe'
    d=json.loads(SEALED.read_text(encoding='utf8'));catalog={x['file'].replace('\\','/'):x for x in d['raw_files']};used={};points=[];details=[]
    def read(folder,name,csvfile=False,new=False):
        rel=(Path(folder)/name).as_posix();p=REPO/rel;digest=sha(p)
        if not new:
            expected=catalog[rel];assert digest==expected['sha256'] and p.stat().st_size==expected['bytes'],rel
        used[rel]=dict(file=rel,bytes=p.stat().st_size,sha256=digest)
        if csvfile:
            with p.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))
        return json.loads(p.read_text(encoding='utf8'))
    starts={'01a':'parallel_optics.width_to_field','01b':'parallel_optics.width_to_field',
        '02':'vision.amplitude_linear_norm_softplus','03':'vision.encode_linear_norm_softplus',
        '04':'language.amplitude_linear_norm_softplus','05':'vision.amplitude_linear_norm_softplus',
        '06':'vision.amplitude_linear_norm_softplus','07':'language.amplitude_linear_norm_softplus',
        '08':'language.amplitude_encoding'}
    cached={}
    def point(run,key):
        folder=run['folder'].replace('\\','/');cachekey=(folder,key)
        if cachekey in cached:return dict(cached[cachekey])
        ours=run['system']=='ours';ident=read(folder,'identity.json');prefix=run['key'].split('|')[0]
        if ours and prefix=='40GB':prefix='01a'
        r=dict(key=key,system=run['system'],repeat=run['repeat'],folder=folder,valid=200,
            event_s=0.,wall_s=0.,optical_s=0.,cpu_output_s=0.,software_s=0.,new_supplement=False)
        if ours:
            fields=(REPO/folder/'fields.csv').exists();calls=read(folder,'fields.csv' if fields else 'calls.csv' if (REPO/folder/'calls.csv').exists() else 'samples.csv',True)
            stages=read(folder,'stages.csv',True);groups=defaultdict(lambda:dict(wall_s=0.,event_s=0.,calls=0))
            r['fields']=len(calls);r['executed_slots']=208 if prefix=='01a' else 200
            for i,c in enumerate(calls):
                rows=[x for x in stages if int(x['sample_index'])==i]
                candidates=[float(x['start_monotonic']) for x in rows if x['stage']==starts[prefix]];assert candidates,(prefix,i)
                beginning=min(candidates);end=float(c['cpu_result_return'])
                after=[x for x in rows if float(x['start_monotonic'])>=beginning]
                assert all(beginning<=float(x['start_monotonic'])<float(x['end_monotonic'])<=end for x in after)
                r['software_s']+=end-beginning
                for x in after:
                    wall=float(x['synchronized_wall_ms'])/1000;event=float(x['cuda_event_ms'])/1000 if x['cuda_event_ms'] else 0.
                    charged=x['charged']=='True';g=groups[(x['stage'],x['family'],charged)];g['wall_s']+=wall;g['event_s']+=event;g['calls']+=1
                    if charged:
                        r['wall_s']+=wall;r['event_s']+=event
                        if not x['cuda_event_ms']:r['cpu_output_s']+=wall
            passes=3 if prefix in ('04','05','06') else 6
            r['optical_s']=len(calls)*passes*1.0447/1000;r['electronic_s']=r['wall_s']
            r['wall_s']+=r['optical_s'];r['event_s']+=r['optical_s']
            r['system_j']=104.230*r['electronic_s']+121.746*r['optical_s'];r['board_j']=None
            r['start_description']=starts[prefix];r['energy_method']='选定电子同步时间×104.230W＋固定光执行时间×121.746W'
            for (name,family,charged),values in groups.items():details.append(dict(key=key,repeat=run['repeat'],stage=name,family=family,charged=charged,**values))
        elif prefix not in ('01a','01b') and len(key.split('|'))==2:
            # Only these seven points have new first-block observation authority.
            version='04' if prefix=='08' and run['repeat'] in (2,3) else '02' if prefix in ('07','08') else '01'
            newfolder=str(Path(folder).parent.parent/('a100_first_tf_core_'+prefix+'_20261025_'+version)/f'qwen_repeat{run["repeat"]:02d}').replace('\\','/')
            if not (REPO/newfolder/'first_tf_summary.json').exists():
                if not allow_pending:raise FileNotFoundError(newfolder+'/first_tf_summary.json')
                return dict(r,pending=True)
            summary=read(newfolder,'first_tf_summary.json',new=True);assert summary['status']=='complete'
            local_audit=read(newfolder,'local_first_tf_independent_audit01.json',new=True);assert local_audit['status']=='passed'
            receipt=read(newfolder,'backup_receipt_complete01.json',new=True)
            assert any(x['path']=='first_tf_summary.json' and x['sha256']==sha(REPO/newfolder/'first_tf_summary.json') for x in receipt['verified_files'])
            manifest=read(newfolder,'first_tf_files_manifest.json',new=True)
            for x in manifest:
                p=REPO/newfolder/x['file'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],str(p)
                rel=p.relative_to(REPO).as_posix();used[rel]=dict(file=rel,bytes=x['bytes'],sha256=x['sha256'])
            newcalls=read(newfolder,'calls.csv' if (REPO/newfolder/'calls.csv').exists() else 'samples.csv',True,new=True)
            oldcalls=read(folder,'calls.csv' if (REPO/folder/'calls.csv').exists() else 'samples.csv',True)
            assert len(newcalls)==len(oldcalls)==200
            for a,b in zip(newcalls,oldcalls):
                for col in ('sample_id','input_ids','attention_mask','valid_input_tokens','native_inputs_json'):
                    if col in a and col in b:assert a[col]==b[col],(key,col)
            core=read(newfolder,'first_tf_calls.csv',True,new=True);power=read(newfolder,'telemetry.csv',True,new=True)
            energy=integrate([(float(x['first_block_start']),float(x['cpu_result_return'])) for x in core],power)
            assert abs(energy['system_j']-summary['system_j'])<1e-6
            r.update(folder=newfolder,new_supplement=True,cpu_output_s=None,wall_s=energy['wall_s'],event_s=sum(float(x['diagnostic_core_cuda_ms']) for x in core)/1000,
                software_s=energy['wall_s'],system_j=energy['system_j'],board_j=energy['board_j'],max_support_age_s=energy['max_support_age_s'],
                start_description='原生首个'+('文字' if prefix in ('04','08') else '视觉')+'Transformer块',energy_method='每次首TF至CPU任务结果区间的338.2W主机代理＋同窗NVML积分')
        else:
            calls=read(folder,'calls.csv',True);power=read(folder,'telemetry.csv',True);assert len(calls)==200
            intervals=[(float(x['first_block_start']),float(x['call_return'])) for x in calls]
            energy=integrate(intervals,power);r.update(wall_s=energy['wall_s'],software_s=energy['wall_s'],system_j=energy['system_j'],
                board_j=energy['board_j'],max_support_age_s=energy['max_support_age_s'],
                event_s=sum(float(x['diagnostic_first_block_cuda_ms']) for x in calls)/1000,
                cpu_output_s=sum(float(x['call_return'])-float(x['gpu_output']) for x in calls),
                start_description='原生首个卷积块' if run['system']=='yolo' else '原生首个Transformer块',
                energy_method='每次首块至CPU评分结果区间的338.2W主机代理＋同窗NVML积分')
        cached[cachekey]=dict(r);return r
    mainpoints=[point(r,r['key']) for r in d['mainRuns']]
    temppoints=[point(r,r['key']) for r in d['tempRuns']]
    supplement_suites=[]
    for version in ('01','02','03','04'):
        folder='LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/a100_first_tf_seven_20261025_'+version
        receipt=REPO/folder/'backup_receipt_suite01.json'
        if not receipt.exists():
            if not allow_pending:raise FileNotFoundError(receipt)
            continue
        received=read(folder,'backup_receipt_suite01.json',new=True)
        for x in received['verified_files']:
            p=REPO/folder/x['path'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
            rel=p.relative_to(REPO).as_posix();used[rel]=dict(file=rel,bytes=x['bytes'],sha256=x['sha256'])
        state=read(folder,'suite_status.json',new=True)
        assert state['mps_released'] is True
        assert state['status']==('complete' if version=='04' else 'stopped')
        assert len(state['completed'])=={'01':15,'02':4,'03':0,'04':2}[version]
        supplement_suites.append(dict(folder=folder,status=state['status'],completed=len(state['completed']),source_commit=state['source_commit']))
        if (REPO/folder/'external_controller.log').exists():
            p=REPO/folder/'external_controller.log';rel=p.relative_to(REPO).as_posix();used[rel]=dict(file=rel,bytes=p.stat().st_size,sha256=sha(p))
    rejected='LightGenV2/tasks/t04_semantic_interaction/runs/simulation/a100_first_tf_core_07_20261025_01/qwen_repeat01'
    if (REPO/rejected/'backup_receipt_telemetry_failure01.json').exists():
        received=read(rejected,'backup_receipt_telemetry_failure01.json',new=True)
        for x in received['verified_files']:
            p=REPO/rejected/x['path'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
            rel=p.relative_to(REPO).as_posix();used[rel]=dict(file=rel,bytes=x['bytes'],sha256=x['sha256'],excluded_reason='功率读数N/A，172条不完整轮，不进入指标')
    for version in ('02','03'):
        interrupted='LightGenV2/tasks/t12_text_to_image/runs/simulation/a100_first_tf_core_08_20261025_'+version+'/qwen_repeat02'
        if (REPO/interrupted/'backup_receipt_MPS_interruption01.json').exists():
            received=read(interrupted,'backup_receipt_MPS_interruption01.json',new=True)
            for x in received['verified_files']:
                p=REPO/interrupted/x['path'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
                rel=p.relative_to(REPO).as_posix();used[rel]=dict(file=rel,bytes=x['bytes'],sha256=x['sha256'],excluded_reason='MPS控制通信中断，不完整轮，不进入指标')
    def records(old,raw):
        result=[]
        for oldrec in old:
            ps=[p for p in raw if p['key']==oldrec['key']];assert len(ps)==3
            record={k:v for k,v in oldrec.items() if k in ('key','id','task','system','metric','accuracy','full_test_n','ours_lab','tokens','cap','power_limit')}
            record['old_full_graph_op']=oldrec.get('op');record['core_op']=None;record['pending']=any(p.get('pending') for p in ps)
            if not record['pending']:
                for k in ('wall_s','event_s','optical_s','system_j','cpu_output_s','software_s'):record[k]=sum(p[k] for p in ps) if all(p[k] is not None for p in ps) else None
                record.update(ms=record['wall_s']/600*1000,j=record['system_j']/600,event_ms=record['event_s']/600*1000)
            result.append(record)
        return result
    full_index={k:dict(v,verification='原完整交付封存清单复用') for k,v in catalog.items()}
    for k,v in used.items():full_index[k]=dict(v,verification='本版逐SHA及大小重核')
    return dict(scope='Qwen first native Transformer (YOLO first CNN) to CPU result; LG first optical module selected-stage hardware projection to CPU result',
        mainRecords=records(d['mainRecords'],mainpoints),tempRecords=records(d['tempRecords'],temppoints),
        mainRuns=mainpoints,tempRuns=temppoints,lgStages=details,raw_files=list(full_index.values()),sealed_full_input_source_sha256=sha(SEALED),
        supplement_suites=supplement_suites,rejected_partial_run=rejected,
        input_preparation_and_inter_request_gaps_excluded=True,downstream_original_operations_preserved=True,
        primary_time='synchronized wall, matching energy',cuda_time='diagnostic event to result transfer, not added to synchronized wall',
        physical_optical_OP=None)
if __name__=='__main__':
    data=collect('--allow-pending' in sys.argv)
    if '--write' in sys.argv:
        OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'delivery_manifest.json').exists()
        (OUT/'core_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(main_points=len(data['mainRuns']),temp_points=len(data['tempRuns']),pending=[x['key'] for x in data['mainRecords'] if x['pending']],sources=len(data['raw_files']))))
