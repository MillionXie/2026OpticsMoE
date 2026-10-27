"""Rebuild a fixed-value display from audited raw records, never edited XLSX."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET
from collect_energy_summary_20261026 import read,sha,point,aggregate
from audit_lgvq_async_point_20261026 import compare
from lgvq_phase_ledger_20261026 import build as phase_build,event_phases,phase_of
from lgvq_async_trace_contract_20261026 import critical_path,OPTICAL_PASS_MS

MANIFEST_SHA='dd709c4d527b05e5ff1e4e681e5969efbc33200f90542b7c4b899ee18b576be2'
HEADERS=['平台','模型','MPS/%','轮次','每16视频/ms','等效单帧/ms','每16视频/J','等效单帧/J','tokens/s','LightGen时间加速比','LightGen能效比','LightGen tokens/s加速比']
NOTES=['LightGen：电子阶段计时＋固定光周期，不含电子图构建；能量按组件核算，非光路直接实测。',
       '仅LightGen按千问参考计数（每16视频7,136 token）；其他模型保留各自原生计数，CLIP/YOLO不适用。',
       '时间/能效比＝各行时间/能量÷LightGen；tokens/s加速比＝LightGen÷各行，token定义不同。40GB与80GB非同卡受控比较。']
MODEL_NAMES={'clip':'OpenAI CLIP ViT-B/32','glm':'GLM-4.6V-Flash','qwen':'Qwen3-VL-2B-Instruct',
             'yolo':'Ultralytics YOLO11s','deepseek':'DeepSeek-VL2','kimi':'Kimi-VL-A3B-Instruct'}
PHASE_HEADERS=['部分','阶段累计跨度\nms/16视频（不可相加）','端到端分摊\nms/16视频（可相加）',
               '首场开始/ms\n相对整轮t=0','首场结束/ms\n相对整轮t=0','并行与依赖说明']
PHASE_NAMES={'tokenizer':'文本分词（Tokenizer）','frontend':'视觉前处理（Embedding/位置编码/Adapter）',
    'vision_entry':'视觉光路入口编码','merger':'帧合并与序列组装（Merger）',
    'language_entry':'语言光路入口编码','decoder':'Decoder与最终评分','scheduling':'调度、CPU返回与同步（贯穿全程）'}
PHASE_PARALLEL={'tokenizer':'CPU文本分词可与前一场GPU主体重叠；每场真实执行。',
    'frontend':'原完整patch批次双流fork/join；跨场与前一场主体流水重叠，保留汇合成本。',
    'vision_entry':'依赖本场前处理完成；同场主体按原顺序执行。',
    'merger':'依赖视觉第三周期及其电子段完成，再向语言阶段提交。',
    'language_entry':'依赖Merger；保留入口编码，不与本场后续光周期任意交换。',
    'decoder':'依赖本场语言第三周期及其电子段；评分fence后才复用槽位。',
    'scheduling':'贯穿整个流水线，非串行尾段；保留CPU动作、观测队列、返回和必要同步。'}
for _i in range(1,7):
    PHASE_NAMES[f'optics{_i}']=f'{"视觉" if _i<=3 else "语言"}光周期{_i if _i<=3 else _i-3}（{("Router","Expert","Global")[(_i-1)%3]}）'
    PHASE_NAMES[f'oeo{_i}']=f'OEO电子段{_i}（读出/路由/融合/下段编码）'
    PHASE_PARALLEL[f'optics{_i}']='本场六周期依次依赖；整幅16视频光场联合传播，可与下一场前处理重叠。'
    PHASE_PARALLEL[f'oeo{_i}']='依赖前一光周期；保留必要电子处理，不将重叠跨度再次从总时间扣除。'


def phase_evidence(folder,raw,ledger):
    """Use the accepted v3 graph, with its original dependencies and t=0."""
    nodes=read(folder/'async_causal_trace.json')['nodes'];mapping=event_phases(raw)
    durations={n['id']:(OPTICAL_PASS_MS if n['kind']=='optical_simulation' else
        0. if n['kind'] in {'detector_ROI','electronic_residual','capture_excluded','H2D'} else n['duration_ms']) for n in nodes}
    for key in ledger['software_only_CPU_submission_node_ids']+ledger['excluded_electronic_graph_construction_node_ids']:
        durations[key]=0.
    schedule=critical_path(nodes,durations)
    compare(schedule['wall_ms']/raw['fields'],ledger['fixed_optics']['wall_ms16'],'phase_schedule_wall')
    bounds={}
    for key in PHASE_NAMES:
        selected=[n for n in nodes if n['field']==0 and phase_of(n,mapping)==key and durations[n['id']]>0]
        bounds[key]=None if not selected else [min(schedule['starts_ms'][n['id']] for n in selected),max(schedule['ends_ms'][n['id']] for n in selected)]
    bounds['scheduling'][0]=0.
    return dict(bounds=bounds,wall_ms=schedule['wall_ms'],fields=raw['fields'],rows=ledger['rows'])


def pooled_phase_table(evidence,time16):
    fields=sum(e['fields'] for e in evidence);work=defaultdict(float);wall=defaultdict(float)
    for e in evidence:
        for r in e['rows']:
            work[r['key']]+=r['executed_work_ms16']*e['fields']/fields
            wall[r['key']]+=r['additive_wall_contribution_ms16']*e['fields']/fields
    order=[r['key'] for r in evidence[0]['rows'] if r['key'] not in ('startup','excluded')]
    if any(work[k]!=0 or wall[k]!=0 for k in ('startup','excluded')):raise ValueError('Excluded phase is not zero')
    compare(sum(wall.values()),time16,'all_phase_contributions_close')
    rows=[]
    for k in order:
        bounds=[e['bounds'][k] for e in evidence]
        starts=ends='首场未独立计时'
        if all(b is not None for b in bounds):
            starts=sum(b[0] for b in bounds)/len(bounds);ends=sum(b[1] for b in bounds)/len(bounds)
        rows.append([PHASE_NAMES[k],work[k],wall[k],starts,ends,PHASE_PARALLEL[k]])
    rows.append(['端到端合计','不可相加',time16,'不适用','不适用','仅本列分摊值合计；首场延迟不等于流水线平均吞吐间隔。'])
    return dict(headers=PHASE_HEADERS,rows=rows,
        notes=['单位：每16视频/64帧。阶段跨度与端到端分摊按三轮共384场归一；分摊合计必须等于统一结果表。',
        '首场起止为各轮第0场相对整轮t=0的三轮均值，保留冷启动等待；未独立hook的首场细项标明缺失，不补造。',
        't=0沿用原host_start，不含窗口前模型加载、原图准备和批量H2D；窗口内文本分词、前处理、评分与调度不移除。',
        '双槽流水：下一场前处理可与前一场主体重叠；槽位i复用前等待i−2场评分完成；同场六周期保持原依赖。',
        '光周期各为SLM 0.714285714＋CCD曝光0.300000000＋读出0.030700000 ms；电子段来自原计时，非光路直接实测。',
        '沿用批准口径：电子图构建单列部署成本，FFT/ROI/电残差/H2D按原范围处理；首三场与两次capture输出均保留。',
        '前处理内部Embedding/位置编码/Adapter未独立hook，不造更细时间；不以各阶段跨度之和代替端到端时间。'],
        rounds=[[i+1,e['fields'],0.,e['wall_ms'],e['wall_ms']/e['fields']] for i,e in enumerate(evidence)],
        round_headers=['轮次','16视频光场数','整轮开始/ms','整轮结束/ms','每16视频/ms'],
        origin='Original host_start; accepted v3 dependency schedule',
        first_field_times_are_not_amortized_stage_times=True)


def prepare(source):
    source=Path(source).resolve()
    if sha(source/'delivery_manifest.json')!=MANIFEST_SHA:raise ValueError('Original data manifest changed')
    manifest=read(source/'delivery_manifest.json')
    row=next(r for r in manifest['files'] if r['path']=='collected_results.json')
    snapshot=source/row['path']
    if snapshot.stat().st_size!=row['bytes'] or sha(snapshot)!=row['sha256']:raise ValueError('Canonical JSON changed')
    data=read(snapshot)
    if len(data['rounds'])!=36 or len(data['LG_rounds'])!=3:raise ValueError('All 36 baseline and three LG rounds required')
    baselines=[point(r['path']) for r in data['rounds']]
    groups=aggregate(baselines);compare(data['groups'],groups,'baseline_replayed_from_raw')
    seconds=energy=0.;common=native=calls=0;workers=set();raw_sources=[];phase_records=[]
    for r in data['LG_rounds']:
        folder=Path(r['path']);point(folder,'LG_async')
        if read(folder/'local_native_parallel_audit01.json')['status']!='passed':raise ValueError('Parallel audit missing')
        phase=phase_build(folder,True,exclude_graph_construction=True)
        compare(read(folder/'local_deployed_phase_ledger_v3_01.json'),phase,'v3_replay')
        identity=read(folder/'identity.json');raw=read(folder/'async_raw_timing.json');cost=read(folder/'cost_report.json')
        workers.add(identity['pid'])
        if raw['explicit_warmup_forwards']!=0:raise ValueError('No extra warmup permitted')
        seconds+=phase['fixed_optics']['wall_ms16']*raw['fields']/1000
        energy+=phase['fixed_optics_component_energy']['energy_j']
        common+=cost['common_effective_tokens'];native+=cost['native_effective_tokens'];calls+=raw['video_calls']
        if cost['common_effective_tokens']!=913408 or raw['video_calls']!=2048:raise ValueError('Original Qwen reference workload required')
        phase_records.append(phase_evidence(folder,raw,phase))
        raw_sources.append(dict(leaf=str(folder),raw_sha256=sha(folder/'async_raw_timing.json'),
            audit_sha256=sha(folder/'local_independent_async_audit01.json'),ledger_sha256=sha(folder/'local_deployed_phase_ledger_v3_01.json')))
    if len(workers)!=3 or calls!=6144:raise ValueError('Three distinct complete processes required')
    t=seconds*1000*16/calls;e=energy*16/calls;rate=common/seconds
    compare(data['LG']['fixed_optics_counterfactual_time16_ms'],t)
    compare(data['LG']['fixed_optics_component_proxy_energy16_j'],e)
    compare(data['LG']['projected_common_tokens_per_s'],rate)
    rows=[['40GB PCIe / 250W','LightGen',100,3,t,t/64,e,e/64,rate,1.,1.,1.]]
    for g in groups:
        native_rate=g['pooled_native_tokens_per_s']
        throughput_ratio='不适用' if native_rate is None else rate/native_rate
        rows.append(['40GB PCIe / 250W' if g['platform']=='A10040GB_PCIE_250W' else '80GB SXM4 / 400W',
            MODEL_NAMES[g['model']],g['cap'],g['rounds'],g['time16_ms'],g['equivalent_frame_ms'],g['energy16_j'],g['energy_per_frame_j'],
            '不适用' if native_rate is None else native_rate,g['time16_ms']/t,g['energy16_j']/e,throughput_ratio])
        if g['common_effective_tokens']!=common or g['video_calls']!=calls:raise ValueError('Mismatched reference workload')
        if native_rate is not None:
            compare(g['native_effective_tokens']/g['elapsed_s'],native_rate,'original_native_token_rate')
        if g['model']=='qwen':compare(rows[-1][9],rows[-1][11],'Qwen_throughput_ratio_equals_time_ratio')
    return dict(status='raw_rechecked_fixed_values',headers=HEADERS,rows=rows,notes=NOTES,
        source_json=str(snapshot),source_json_sha256=sha(snapshot),source_manifest_sha256=MANIFEST_SHA,
        baseline_original_points_rechecked=36,LG_original_points_rechecked=3,LG_source_receipts=raw_sources,
        edited_XLSX_read=False,explicit_extra_warmup_forwards=0,physical_optical_hardware_measured=False,
        energy_is_component_accounting=True,excel_formulas_requested=False,
        token_scope='LightGen: Qwen reference equivalent tokens; baselines: each original native token count',
        common_effective_tokens=common,common_tokens_per_16_videos=7136,
        LG_native_prompt_tokens_not_used_for_display=native,
        throughput_ratio_direction='LightGen Qwen-reference rate / baseline native rate',
        baseline_native_token_rates_preserved=True,throughput_ratios_equal_time_ratios_for_Qwen_only=True,
        cross_model_token_definitions_differ=True,phase_table=pooled_phase_table(phase_records,t))


def audit(folder,visual_reviewed=False):
    folder=Path(folder).resolve();expected=read(folder/'static_values.json');book=folder/'Temporal_LightGen与baseline_固定数值_20261027.xlsx'
    # Recheck the raw data independently again; no workbook is used as an input.
    compare(expected,prepare(Path(expected['source_json']).parent),'independent_raw_replay')
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(book) as z:
        names=[n.get('name') for n in ET.fromstring(z.read('xl/workbook.xml')).findall('s:sheets/s:sheet',ns)]
        if names!=['统一结果','LightGen时间分块']:raise ValueError('Exactly two fixed-value sheets required')
        if any(n.startswith('xl/externalLinks/') for n in z.namelist()):raise ValueError('External links forbidden')
        shared=[''.join(s.itertext()) for s in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)] if 'xl/sharedStrings.xml' in z.namelist() else []
        def parse_sheet(number):
            cells={}
            for c in ET.fromstring(z.read(f'xl/worksheets/sheet{number}.xml')).findall('.//s:sheetData/s:row/s:c',ns):
                if c.find('s:f',ns) is not None or c.get('t')=='e':raise ValueError('All cells must be fixed values without errors')
                v=c.find('s:v',ns);inline=c.find('s:is',ns)
                value=v.text if v is not None else ''.join(inline.itertext()) if inline is not None else None
                if c.get('t')=='s':value=shared[int(value)]
                elif value is not None and c.get('t') not in ('inlineStr','str'):
                    value=float(value)
                    if not math.isfinite(value):raise ValueError('Nonfinite cell')
                cells[c.get('r')]=value
            if any(isinstance(v,str) and ('投影' in v or '软件' in v) for v in cells.values()):raise ValueError('Presentation must contain only the requested LG metric')
            return cells
        cells=parse_sheet(1)
        for r,row in enumerate([expected['headers']]+expected['rows'],1):
            for i,value in enumerate(row):compare(value,cells.get(chr(65+i)+str(r)),f'cell_{r}_{i}')
        for i,note in enumerate(expected['notes'],16):compare(note,cells.get('A'+str(i)))
        phase=expected['phase_table'];cells=parse_sheet(2)
        for r,row in enumerate([phase['headers']]+phase['rows'],1):
            for i,value in enumerate(row):compare(value,cells.get(chr(65+i)+str(r)),f'phase_cell_{r}_{i}')
        for i,note in enumerate(phase['notes'],24):compare(note,cells.get('A'+str(i)))
        for r,row in enumerate([phase['round_headers']]+phase['rounds'],33):
            for i,value in enumerate(row):compare(value,cells.get(chr(65+i)+str(r)),f'round_cell_{r}_{i}')
        compare(sum(r[2] for r in phase['rows'][:-1]),expected['rows'][0][4],'saved_phase_sum_equals_main_time')
        if any(isinstance(r[3],(int,float)) and (r[3]<0 or r[4]<r[3]) for r in phase['rows'][:-1]):raise ValueError('Invalid first-field boundaries')
    if not visual_reviewed:raise ValueError('Review the one-sheet render before sealing')
    result=dict(status='passed',rows=13,sheets=2,formulas=0,external_links=0,raw_points_rechecked=39,
        all_saved_values_independently_verified=True,edited_XLSX_used_as_source=False,
        deletion_cannot_trigger_reference_recalculation=True,visual_reviewed=True,workbook_sha256=sha(book),
        common_effective_tokens=expected['common_effective_tokens'],
        baseline_native_token_rates_preserved=True,throughput_ratios_equal_time_ratios_for_Qwen_only=True,
        cross_model_token_definitions_differ=True,phase_table_closes_to_main_time=True,
        original_t0_retained=True,phase_overlap_not_double_counted=True)
    with (folder/'saved_values_audit.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=True,indent=2)
    with (folder/'delivery_manifest.json').open('x',encoding='utf8') as f:
        json.dump(dict(status='sealed',files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in
            (book,folder/'static_values.json',folder/'saved_values_audit.json')]),f,ensure_ascii=True,indent=2)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','audit']);p.add_argument('source',type=Path)
    p.add_argument('--out',type=Path);p.add_argument('--visual-reviewed',action='store_true');a=p.parse_args()
    if a.mode=='prepare':
        result=prepare(a.source)
        with a.out.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=True,indent=2)
        print(json.dumps(dict(status='passed',rows=len(result['rows']),LG_time16_ms=result['rows'][0][4]),ensure_ascii=True))
    else:print(json.dumps(audit(a.source,a.visual_reviewed),ensure_ascii=True))
