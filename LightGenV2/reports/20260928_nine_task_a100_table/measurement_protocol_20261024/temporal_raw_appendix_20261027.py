"""Source-preserving timing appendix and unaggregated global-control evidence."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import temporal_static_values_20261027 as base
from lgvq_async_trace_contract_20261026 import critical_path,OPTICAL_PASS_MS

OUT=Path(__file__).resolve().parent/'outputs/01a061e2-6f88-7913-87f5-b632000e4c83'
SOURCE=OUT/'Temporal_LightGen_appendix_20261027_09'
SOURCE_SHA='cf882208766e774437698e489c88114d512f58aed67e2317d974b171d7cc4796'
CONTROL_NAMES={'initial_entry':'整轮进入间隙','field_entry':'场间主机间隙','slot_sync':'槽位复用等待',
    'capture_sync':'捕获相关同步','score_sync':'评分就绪同步','final_sync':'最终同步返回','capture_gap':'捕获相邻间隙'}
OPTICAL_TEXT='单周期：SLM刷新0.714285714 ms + CCD曝光0.300000000 ms + CCD读出0.030700000 ms。'
SUMMARY_HEADERS=['模块','处理步骤','原事件/固定周期累计/ms','并行后计算累计/ms','并行后控制累计/ms','并行后合计/ms','等效时间/ms\n每16视频64帧']
ROUND_HEADERS=['轮次','模块','处理步骤','原事件/固定周期累计/ms','并行后计算累计/ms','并行后控制累计/ms','等效时间/ms\n每16视频64帧']
GLOBAL_HEADERS=['调度类别','原始记录数','原始区间累计/ms','依赖分离后节点累计/ms','并行后计入/ms','等效时间/ms\n每16视频64帧']
RAW_HEADERS=['轮次','原场次\n从0计','调度类别','原始动作','原始开始/ms','原始结束/ms','原始区间时长/ms',
    '依赖分离后节点/ms','是否计入附录','计入时间/ms','核算开始/ms','核算结束/ms','原始节点ID','依赖节点ID']


def global_owner(node,records):
    field=node['field'];key=node['id'];prefix=f'f{field}.'
    suffix=key[len(prefix):] if key.startswith(prefix) else key
    if suffix.endswith('.begin') or key=='origin':return None
    if suffix=='token.hostgap':return 'initial_entry' if field==0 else 'field_entry'
    if suffix.startswith('slotwait'):return 'slot_sync'
    if suffix.startswith('finalsync'):return 'final_sync'
    if re.fullmatch(r'sync\d+(?:\.hostgap)?',suffix):
        index=int(re.search(r'\d+',suffix).group())
        return 'score_sync' if records[field]['synchronization_observations'][index]['kind']=='score_ready' else 'capture_sync'
    if re.fullmatch(r'capture\d+\.hostgap',suffix):return 'capture_gap'
    return None


def original_interval(node,by_id):
    """Use original host observations, never a projected schedule as raw data."""
    witness=node['witness']
    if 'original_call' in witness:return witness['original_call']
    left,right=witness['boundaries']
    a=0. if left=='origin' else by_id[left]['witness']['original_call'][1]
    b=by_id[right]['witness']['original_call'][0]
    return [a,b]


def build_evidence(data):
    parts=[dict(p,module=r[0],part=r[1]) for p,r in zip(data['appendix_table']['phase_accounting'],data['appendix_table']['rows'])]
    rows=[];round_rows=[];raw_rows=[];sources=[]
    by_round=[];global_rounds=[];global_sums={key:[0,0.,0.,0.] for key in CONTROL_NAMES}
    for round_number,source in enumerate(data['LG_source_receipts'],1):
        folder=Path(source['leaf']);raw=base.read(folder/'async_raw_timing.json');trace=base.read(folder/'async_causal_trace.json')
        ledger=base.read(folder/'local_deployed_phase_ledger_v3_01.json');nodes=trace['nodes'];by_id={n['id']:n for n in nodes}
        mapping=base.event_phases(raw);evidence=base.phase_evidence(folder,raw,ledger);fine=evidence['fine'];fields=raw['fields']
        durations={n['id']:(OPTICAL_PASS_MS if n['kind']=='optical_simulation' else
            0. if n['kind'] in {'detector_ROI','electronic_residual','capture_excluded','H2D'} else n['duration_ms']) for n in nodes}
        for key in ledger['software_only_CPU_submission_node_ids']+ledger['excluded_electronic_graph_construction_node_ids']:durations[key]=0.
        schedule=critical_path(nodes,durations);chain=set(ledger['fixed_optics']['critical_path_node_ids'])
        base.compare(schedule['wall_ms'],evidence['wall_ms'],'source_schedule')
        round_parts=[]
        for p in parts:
            work=math.fsum(fine['work'].get(k,0.) for k in p['keys'])
            compute=math.fsum(fine['compute'].get(k,0.) for k in p['keys'])
            control=math.fsum(fine['overhead'].get(k,0.) for k in p['control_keys'])
            round_parts.append((work,compute,control))
            round_rows.append([round_number,p['module'],p['part'],work,compute,control,(compute+control)/fields])
        base.compare(math.fsum(c+h for _,c,h in round_parts),schedule['wall_ms'],'round_parts_close')
        by_round.append(round_parts)
        records={r['field']:r for r in raw['records']};local={key:[0,0.,0.,0.] for key in CONTROL_NAMES}
        original_rows=[]
        for node in nodes:
            if node['kind'] not in ('dispatch','sync'):continue
            owner=global_owner(node,records)
            if owner is None:continue
            a,b=original_interval(node,by_id);key=node['id'];d=durations[key]
            if b<a or d<0:raise ValueError('Invalid raw control duration')
            base.compare(node['duration_ms'],d,'global_duration_not_excluded')
            kept=d if key in chain else 0.
            # original interval contains real blocking; d retains only return
            # overhead after the dependency, exactly as the accepted DAG does.
            if d>(b-a)+1e-6:raise ValueError('Retained control longer than original call')
            original_rows.append([round_number,node['field'],CONTROL_NAMES[owner],
                '动作间隙' if key.endswith('.hostgap') else '同步调用',a,b,b-a,d,'是' if key in chain else '否',kept,
                schedule['starts_ms'][key],schedule['ends_ms'][key],key,', '.join(node['parents'])])
            q=local[owner];q[0]+=1;q[1]+=b-a;q[2]+=d;q[3]+=kept
        original_rows.sort(key=lambda r:(r[4],r[5],r[12]));raw_rows.extend(original_rows)
        for owner,q in local.items():
            base.compare(q[3],fine['overhead'].get(owner,0.),'global_original_records_close_'+owner)
            global_rounds.append([round_number,CONTROL_NAMES[owner],q[0],q[1],q[2],q[3],q[3]/fields])
            for j,v in enumerate(q):global_sums[owner][j]+=v
        sources.append(dict(round=round_number,run=folder.parent.name,fields=fields,
            raw_sha256=base.sha(folder/'async_raw_timing.json'),trace_sha256=base.sha(folder/'async_causal_trace.json'),
            ledger_sha256=base.sha(folder/'local_deployed_phase_ledger_v3_01.json'),wall_ms=schedule['wall_ms']))
    fields=sum(r['fields'] for r in sources)
    for i,p in enumerate(parts):
        work,compute,control=[math.fsum(r[i][j] for r in by_round) for j in range(3)]
        value=(compute+control)/fields;base.compare(value,p['equivalent_ms16'],'pooled_part_'+str(i))
        rows.append([p['module'],p['part'],work,compute,control,compute+control,value])
    global_rows=[[CONTROL_NAMES[k],*v,v[3]/fields] for k,v in global_sums.items()]
    original_count=sum(v[0] for v in global_sums.values());raw_count=len(raw_rows)
    if original_count!=raw_count or len({(r[0],r[12]) for r in raw_rows})!=raw_count:raise ValueError('Raw control rows missing or duplicated')
    base.compare(math.fsum(r[-1] for r in global_rows),data['appendix_table']['remaining_window_control_ms16'],'global_aggregate_matches_appendix')
    total=math.fsum(r[-1] for r in rows);base.compare(total,data['rows'][0][5],'all_parts_match_main')
    return dict(summary_headers=SUMMARY_HEADERS,summary_rows=rows,round_headers=ROUND_HEADERS,round_rows=round_rows,
        global_headers=GLOBAL_HEADERS,global_rows=global_rows,global_round_headers=['轮次']+GLOBAL_HEADERS,
        global_round_rows=global_rounds,raw_headers=RAW_HEADERS,raw_rows=raw_rows,sources=sources,fields=fields,
        original_control_records=raw_count,total_ms16=data['rows'][0][5],global_ms16=math.fsum(r[-1] for r in global_rows),
        optical_components=[['SLM刷新',1000/1400],['CCD曝光',.3],['CCD读出',.0307]],
        method_notes=[
          '每轮128场，每场16视频64帧。三轮共384场。所有数值按全精度核算后固定保存，不依赖其他工作表的公式。',
          '原事件/固定周期累计是各次记录时长之和，尚未处理并行覆盖，不可直接当作端到端时间。光学列使用固定器件周期，电子列来自原A100事件。',
          '并行后合计 = 并行后计算累计 + 并行后控制累计；每16视频等效时间 = 并行后合计 ÷ 384。各行与原附录逐项对应。',
          '原事件分块可合并展示。原始全程调度逐条列在调度原始明细，每轮场次从0计到127。未合并原始记录，也未用平均值替代起止时间。',
          '同步原始区间包含等待GPU完成的阻塞。等待关系由原依赖表达，节点时长只保留就绪后的返回开销，不能把阻塞再加一次。',
          '调度分类核对额外汇总三轮原始区间、依赖分离后的节点和并行后计入量。只有最后一列等效时间与附录的全程调度行相加核对。',
          '光传播的提交、间隙和排队仍按批准分组计入对应OEO电子行。全程调度只含原七类全局控制，不重复这些部件内控制。'])


def prepare():
    prior=base.read(SOURCE/'static_values.json')
    verified=base.prepare(Path(prior['source_json']).parent)
    base.compare(prior['rows'],verified['rows'],'canonical_original_cohort')
    book=SOURCE/base.BOOK_NAME
    if base.sha(book)!=SOURCE_SHA:raise ValueError('Actual edited workbook changed; inspect it again')
    data=dict(prior);data.update(base_workbook=str(book),base_workbook_sha256=SOURCE_SHA,
        base_main_cells=base.read_workbook_cells(book,1),base_appendix_cells=base.read_workbook_cells(book,2),
        base_word_document=str(SOURCE/'Temporal_LightGen_附录计时与统计方法_20261027.docx'),
        base_word_sha256='1619a5495c7b45e9a26a5c1a21bd92ba8dfadb1bd4864d641c563b030aff3aa9',edit_mode='raw_appendix')
    if base.sha(data['base_word_document'])!=data['base_word_sha256']:raise ValueError('Source Word changed')
    cells=data['base_appendix_cells'];p=dict(prior['appendix_table']);p['rows']=[]
    for n,expected in enumerate(prior['appendix_table']['rows'],4):
        base.compare(cells.get(f'C{n}'),expected[2],'edited_numeric_'+str(n))
        p['rows'].append([cells.get(f'A{n}'),cells.get(f'B{n}'),expected[2],OPTICAL_TEXT if n in [5,7,9,13,15,17] else cells.get(f'D{n}')])
    p['notes']=[cells.get('A'+str(n)) for n in range(24,34)];data['appendix_table']=p
    data['raw_appendix']=build_evidence(data)
    return data


def audit(folder,visual_reviewed):
    folder=Path(folder);data=base.read(folder/'static_values.json');base.compare(data,prepare(),'independent_full_raw_replay')
    book=folder/base.BOOK_NAME;raw=data['raw_appendix'];ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(book) as z:
        names=[n.attrib['name'] for n in ET.fromstring(z.read('xl/workbook.xml')).find('m:sheets',ns)]
        if names!=['统一结果','LightGen附录计时','原始分块核对','调度分类核对','调度原始明细']:raise ValueError('Unexpected sheets')
        if any(ET.fromstring(z.read(n)).findall('.//m:f',ns) for n in z.namelist() if re.fullmatch(r'xl/worksheets/sheet\d+.xml',n)):raise ValueError('Fixed-value workbook required')
        if any(n.startswith('xl/externalLinks/') for n in z.namelist()):raise ValueError('External reference')
    saved_main=base.read_workbook_cells(book,1)
    for address in set(saved_main)|set(data['base_main_cells']):
        base.compare(saved_main.get(address),data['base_main_cells'].get(address),'main_preserved_'+address)
    cells=base.read_workbook_cells(book,2)
    for address,value in data['base_appendix_cells'].items():
        if address not in {'D5','D7','D9','D13','D15','D17'}:base.compare(cells.get(address),value,'preserved_'+address)
    for n in [5,7,9,13,15,17]:base.compare(cells.get(f'D{n}'),OPTICAL_TEXT,'optical_three_components')
    def matrix(sheet,start,expected):
        cells=base.read_workbook_cells(book,sheet)
        for i,row in enumerate(expected,start):
            for j,value in enumerate(row):base.compare(cells.get(chr(65+j)+str(i)),value,f'raw_sheet{sheet}_{i}_{j}')
    matrix(3,5,[raw['summary_headers']]+raw['summary_rows']);matrix(3,27,[raw['round_headers']]+raw['round_rows'])
    matrix(3,23,[['合计',None,None,*[math.fsum(r[j] for r in raw['summary_rows']) for j in range(3,7)]]])
    matrix(3,91,[['组成部分','单次时长/ms']]+raw['optical_components']+[['单周期合计',OPTICAL_PASS_MS]])
    matrix(4,5,[raw['global_headers']]+raw['global_rows']);matrix(4,17,[raw['global_round_headers']]+raw['global_round_rows'])
    matrix(4,13,[['合计',*[math.fsum(r[j] for r in raw['global_rows']) for j in range(1,6)]]])
    matrix(5,4,[raw['raw_headers']]+raw['raw_rows'])
    if not visual_reviewed:raise ValueError('Review changed views and all new sheets first')
    doc=base.read(folder/'document_audit.json')
    if doc['status']!='passed' or doc['source_values_sha256']!=base.sha(folder/'static_values.json'):raise ValueError('Matching Word audit required')
    if base.sha(folder/'Temporal_LightGen_附录计时与统计方法_20261027.docx')!=doc['document_sha256']:raise ValueError('Word changed after audit')
    if base.sha(folder/'preview.png')!='93dcaa7bbc6bd204cb97dd6380f5195e1d6485ab741f71c8ce57643d6fc81dbb':raise ValueError('Main view changed')
    result=dict(status='passed',all_39_original_points_replayed=True,original_control_records=raw['original_control_records'],
        original_control_records_unaggregated=True,original_sync_blocking_intervals_preserved=True,
        per_round_and_pooled_control_reconciled=True,original_main_and_user_text_preserved=True,
        optical_descriptions_have_three_fixed_components=True,all_appendix_times_unchanged=True,
        total_ms16=raw['total_ms16'],global_ms16=raw['global_ms16'],sheets=5,formulas=0,external_links=0,
        workbook_sha256=base.sha(book),matching_Word_audited=True,visual_reviewed=True)
    with (folder/'saved_values_audit.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=True,indent=2)
    files=[book,folder/'Temporal_LightGen_附录计时与统计方法_20261027.docx',folder/'static_values.json',folder/'saved_values_audit.json',folder/'document_audit.json']
    with (folder/'delivery_manifest.json').open('x',encoding='utf8') as f:json.dump(dict(status='sealed',files=[dict(path=p.name,bytes=p.stat().st_size,sha256=base.sha(p)) for p in files]),f,ensure_ascii=True,indent=2)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','audit']);ap.add_argument('path',type=Path)
    ap.add_argument('--visual-reviewed',action='store_true');a=ap.parse_args()
    if a.mode=='prepare':
        data=prepare()
        with a.path.open('x',encoding='utf8') as f:json.dump(data,f,ensure_ascii=True,indent=2)
        print(json.dumps(dict(status='passed',original_control_records=data['raw_appendix']['original_control_records'],total_ms16=data['raw_appendix']['total_ms16']),ensure_ascii=True))
    else:print(json.dumps(audit(a.path,a.visual_reviewed),ensure_ascii=True))
