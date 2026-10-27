"""Rebuild a fixed-value display from audited raw records, never edited XLSX."""
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET
from collect_energy_summary_20261026 import read,sha,point,aggregate
from audit_lgvq_async_point_20261026 import compare
from lgvq_phase_ledger_20261026 import build as phase_build,event_phases,phase_of
from lgvq_async_trace_contract_20261026 import critical_path,OPTICAL_PASS_MS

MANIFEST_SHA='dd709c4d527b05e5ff1e4e681e5969efbc33200f90542b7c4b899ee18b576be2'
BOOK_NAME='Temporal_LightGen与baseline_模型对比与附录计时_20261027.xlsx'
HEADERS=['平台','模型','参数规模/B','MPS/%','轮次','每16视频/ms','等效单帧/ms','每16视频/J','等效单帧/J','tokens/s','LightGen时间加速比','LightGen能效比','LightGen tokens/s加速比']
NOTES=['LightGen：电子阶段计时＋固定光周期，不含电子图构建；能量按组件核算，非光路直接实测。',
       '仅LightGen按千问参考计数（每16视频7,136 token）；其他模型保留各自原生计数，CLIP/YOLO不适用。',
       '时间/能效比＝各行时间/能量÷LightGen；tokens/s加速比＝LightGen÷各行，token定义不同。40GB与80GB非同卡受控比较。',
       '参数规模：1 B＝10⁹个参数；Qwen 2 B、GLM 9 B沿用标称规格，其余baseline为原模型总量记录，非仅本次激活量。',
       'LightGen：学生6.804011 M＋已绑定冻结前端315.098112 M＝321.902123 M（0.321902123 B）；不加载完整Qwen Transformer。',
       'LightGen学生计数含光学可学习自由度与电子部分；非单纯GPU矩阵权重，不与MoE激活参数混同。',
       'DeepSeek-VL2总量27.480134248 B、标称激活约4.5 B；Kimi总量16.407656048 B、标称激活约3 B。',
       'CLIP含未用于本视觉推理的文本分支；YOLO按原完整检测模型。参数不代表执行FLOPs；baseline附加质量头未计入原主干总量。']
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
    fine=fine_accounting(nodes,durations,mapping,raw,ledger)
    return dict(bounds=bounds,wall_ms=schedule['wall_ms'],fields=raw['fields'],rows=ledger['rows'],fine=fine)


CONTROL_KEYS=('initial_entry','field_entry','slot_sync','capture_sync','score_sync','final_sync','capture_gap')


def fine_accounting(nodes,durations,mapping,raw,ledger):
    """Partition exactly the same witnessed path; reclassify, never retime."""
    records={r['field']:r for r in raw['records']};context={}
    for row in raw['records']:
        current='vision_entry';active=0
        for i,event in enumerate(row['events']):
            key=f"f{row['field']}.g{i}";phase=mapping[key]
            context[key]=current if phase=='excluded' else phase
            if event['kind']=='optical_simulation':active+=1;current=f'oeo{active}'
            elif phase not in ('frontend','excluded'):current=phase
    critical=set(ledger['fixed_optics']['critical_path_node_ids'])
    work=defaultdict(float);compute=defaultdict(float);overhead=defaultdict(float);breakdown=defaultdict(float);witness=[]
    for node in nodes:
        key=node['id'];duration=durations[key]
        if duration==0:continue
        if node['kind'] not in ('dispatch','sync'):
            owner=phase_of(node,mapping)
            if owner in ('startup','excluded','scheduling'):raise ValueError('Unexpected nonzero computation scope: '+key)
            work[owner]+=duration
            if key in critical:compute[owner]+=duration
            continue
        field=node['field'];prefix=f'f{field}.';suffix=key[len(prefix):] if key.startswith(prefix) else key
        if suffix=='token.hostgap':owner='initial_entry' if field==0 else 'field_entry';kind='主机动作间隙'
        elif suffix.startswith('slotwait'):owner='slot_sync';kind='槽位评分fence'
        elif suffix.startswith('finalsync'):owner='final_sync';kind='最终同步/返回'
        elif re.fullmatch(r'sync\d+(?:\.(?:hostgap|begin))?',suffix):
            index=int(re.search(r'\d+',suffix).group());sync=records[field]['synchronization_observations'][index]['kind']
            owner='score_sync' if sync=='score_ready' else 'capture_sync';kind='显式同步/相邻间隙'
        elif re.fullmatch(r'capture\d+\.hostgap',suffix):owner='capture_gap';kind='捕获相邻主机间隙'
        elif suffix.startswith('pin') or suffix.startswith('H2D'):
            owner='frontend';kind='输入页锁定/传输就绪间隙'
        elif re.fullmatch(r'submit\d+(?:\.(?:hostgap|begin))?',suffix):
            index=int(re.search(r'\d+',suffix).group());owner=context[prefix+f'g{index}'];kind='CPU提交前间隙' if suffix.endswith('hostgap') else 'CPU提交函数'
        elif re.fullmatch(r'g\d+\.queue',suffix):owner=context[key[:-len('.queue')]];kind='GPU就绪队列'
        elif re.fullmatch(r'eagergap\d+(?:\.(?:hostgap|begin))?',suffix):
            index=int(re.search(r'\d+',suffix).group());owner=mapping[prefix+f'gap{index}'];kind='首三场电子段主机动作'
        elif suffix.startswith('eagertail'):owner='decoder';kind='首三场读出后主机动作'
        elif re.fullmatch(r'gap\d+\.queue',suffix):owner=mapping[key[:-len('.queue')]];kind='首三场电子段就绪队列'
        else:raise ValueError('Unclassified nonzero control node: '+key)
        if owner not in PHASE_NAMES and owner not in CONTROL_KEYS:raise ValueError('Unknown control owner')
        if key in critical:
            overhead[owner]+=duration;breakdown[(owner,kind)]+=duration
            witness.append(dict(node=key,owner=owner,kind=kind,ms=duration))
    compare(sum(compute.values())+sum(overhead.values()),ledger['fixed_optics']['wall_ms16']*raw['fields'],'fine_node_partition')
    compare(sum(overhead.values()),ledger['fixed_optics']['critical_path_contribution_ms16']['scheduling']*raw['fields'],'old_scheduling_partition')
    return dict(work=dict(work),compute=dict(compute),overhead=dict(overhead),
        control_breakdown=[dict(owner=o,kind=k,ms=v) for (o,k),v in sorted(breakdown.items())],control_witness=witness)


def detailed_appendix(evidence,detail,time16):
    fields=sum(e['fields'] for e in evidence);work=defaultdict(float);compute=defaultdict(float);overhead=defaultdict(float);breakdown=defaultdict(float)
    for e in evidence:
        for source,target in [('work',work),('compute',compute),('overhead',overhead)]:
            for key,value in e['fine'][source].items():target[key]+=value/fields
        for r in e['fine']['control_breakdown']:breakdown[(r['owner'],r['kind'])]+=r['ms']/fields
    rows=[];types=[];module_rows=[];assigned=[];accounting=[]
    def line(module,label,keys,relation):
        assigned.extend(keys);c=sum(compute[k] for k in keys);h=sum(overhead[k] for k in keys)
        span=sum(work[k] for k in keys) if any(k in PHASE_NAMES for k in keys) else '不适用'
        rows.append([module,label,c+h,relation]);types.append('accounted')
        accounting.append(dict(module=module,part=label,keys=keys,cumulative_work_ms16=span,
                               computation_ms16=c,control_ms16=h,equivalent_ms16=c+h))
    line('预处理','① 文本分词与视觉、文本Embedding\n② 视觉Adapter',['tokenizer','frontend','vision_entry'],'两步联合计时。文本与视觉Embedding有并行；视觉Adapter含前端位置处理、特征转换、缩放/偏置、视觉调制及首次视觉光场输入映射。原事件未完全拆开，不人为分配两步时间。')
    for module,indices in [('视觉光电处理',range(1,4)),('语言光电处理',range(4,7))]:
        if module=='语言光电处理':
            line('多模态融合','Merger与多模态序列准备',['merger'],'先聚合视觉特征，再与已完成的文本特征拼接；含序列位置/掩码及同一事件边界内的输入准备，不是单独一次Merger调用。')
            line('语言光电处理','语言Adapter',['language_entry'],'Merger之后，进入语言Router光场前的适配；按功能定义Adapter，不对应源码类名。此处不重复执行文本Embedding。')
        for i in indices:
            name=('Router','Expert','Global')[(i-1)%3]
            line(module,name+'光传播',[f'optics{i}'],'本场按Router、Expert、Global依赖执行；每周期16视频整幅光场联合传播。跨场重叠只核算一次。')
            line(module,name+' OEO电子处理',[f'oeo{i}'],'保留本周期后必要的读出、路由、融合及下段输入准备；并非仅A/D或D/A器件延时。')
    line('解码器','Decoder与最终评分',['decoder'],'本场语言光电处理完成后执行，包含原读出及最终评分边界。')
    line('调度与同步','全程调度与同步',list(CONTROL_KEYS),'贯穿整轮的进入、换场、换槽、捕获周边间隙和同步返回。末行仅汇总，不是串行尾段；不重复各步骤已计的提交/排队。')
    if len(assigned)!=len(set(assigned)) or set(assigned)!=((set(PHASE_NAMES)-{'scheduling'})|set(CONTROL_KEYS)):
        raise ValueError('Detailed appendix keys overlap or are missing')
    c=math.fsum(r['computation_ms16'] for r in accounting);h=math.fsum(r['control_ms16'] for r in accounting)
    compare(c+h,time16,'detailed_appendix_closes');old=next(r[2] for r in detail['rows'] if r[0]==PHASE_NAMES['scheduling'])
    compare(h,old,'old_scheduling_exactly_redistributed')
    for module in dict.fromkeys(r[0] for r in rows):
        module_rows.append([module,math.fsum(r[2] for r in rows if r[0]==module)])
    rows.append(['合计','17个联合处理/控制项相加',time16,'原起点、首三场和两次真实捕获的输出均保留；无新增预热或排除项。']);types.append('total')
    notes=[
        '工作量：512个原TEST视频循环4次，每轮2048次视频调用、128组；3轮共6144次调用、384组。每组16视频，每视频4帧，即64帧。',
        '等效时间是整轮并行运行时间中分配给该步骤的部分，已处理重叠。各行相加等于总时间；不是步骤单独运行的耗时，也不是首组返回延迟。',
        '各步骤的等效时间已含相关CPU提交、必要间隙及就绪排队的非重复部分。CPU负责发出计算指令；队列表示工作就绪后等待执行。不得再加一次这些时间。',
        '先后关系：文本分词先生成token ID，文本与视觉Embedding有并行；然后视觉Adapter完成进入视觉光场前的特征适配。Merger使用已完成的文本特征，不在此重新分词或Embedding。',
        'Adapter统一按光场输入前的适配功能命名。视觉Adapter包含前端特征转换、缩放/偏置、视觉调制及首次光场映射；语言Adapter是Merger后进入语言Router前的适配。后续周期的适配保留在相应OEO电子处理中。',
        '调度与同步仅合计未分配给单一步骤的全程控制：整轮进入、场间主机间隙、槽位复用、捕获周边间隙和同步返回。它不是最后才执行的串行段。',
        '同场光周期、OEO电子处理、Merger、Decoder保持原依赖；跨场前处理和主体流水重叠。总时间按依赖汇合时较晚完成者确定，不再从总时间二次扣重叠。',
        '电子段取自A100原始时序；每光周期采用0.714285714＋0.300000000＋0.030700000 ms固定器件时序，非物理整机直接实测。OEO含必要读出/路由/融合/下段输入准备。',
        '保留原t=0、首三真实eager和两次真实捕获的输出；不新增预热。模型加载/原图准备/批量H2D在窗口前；电子图构建函数按已批准口径排除，其相邻间隙仍保留。',
        'T=(1939.0622973842435＋1870.5115314995921＋1848.9453026350632)/384=14.735726904997131 ms/16视频。全精度计算后显示6位小数，尾数可能有舍入差。']
    return dict(title='LightGen分阶段等效时间（每组16视频，共64帧）',
        scope='3轮共6144次视频调用、384组；按整轮流水线运行时间折算每组平均值。',
        headers=['模块','处理步骤（按本场数据依赖顺序）','等效时间/ms\n每16视频（64帧）','内容与先后及并行关系'],
        rows=rows,row_types=types,notes=notes,module_rows=module_rows,
        module_headers=['模块','等效时间/ms（每16视频）'],phase_accounting=accounting,
        control_breakdown=[dict(owner=o,kind=k,ms16=v) for (o,k),v in sorted(breakdown.items())],
        old_scheduling_ms16=old,all_positive_path_nodes_assigned_once=True,
        module_local_control_ms16=sum(overhead[k] for k in PHASE_NAMES if k!='scheduling'),
        remaining_window_control_ms16=sum(overhead[k] for k in CONTROL_KEYS),
        no_independent_embedding_position_adapter_timings=True)


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


def parameter_metadata(data):
    repo=Path(__file__).resolve().parents[4]
    catalog=repo/'LightGenV2/reports/20260927_lgvq_temporal_multi_baseline_a100/recalculated_20261013/plot_data.csv'
    document=repo/'LightGenV2/tasks/t06_video_quality_assessment/reports/architecture/ELECTRONIC_PARAMETER_AUDIT_20260912.md'
    binding=repo/'LightGenV2/tasks/t06_video_quality_assessment/runs/smoke/lgvq_frontend_identity_recovery_20261024_01/front_parameter_binding_02.json'
    pins={catalog:'c387f68e75ac02adbe790c4b19c0e67f727465191a36c1498425400c8b7fb3a9',
          document:'9c3405fe99b802ddac6c9fabba34ebf87a73d095cfcf216f8b7889e7eff29be6',
          binding:'6bb38fe494d35cc568fc3e00e1bd9901ce6f9e88e5e5dd8d3d7041292d8d4ed0'}
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Parameter source changed: '+str(path))
    with catalog.open(encoding='utf-8-sig',newline='') as f:records=list(csv.DictReader(f))
    values={};sources={}
    for model in MODEL_NAMES:
        row=next(r for r in records if r['baseline']==('full' if model=='deepseek' else model))
        values[MODEL_NAMES[model]]=float(row['model_total_parameters_billion'])
        sources[model]={k:row[k] for k in ['parameter_basis','parameter_source','checkpoint_sha256','moe_active_parameters_billion']}
        for r in data['rounds']:
            identity=read(Path(r['path'])/'identity.json')
            if identity['model']!=model:continue
            if identity['checkpoint_sha256']!=row['checkpoint_sha256']:raise ValueError('Parameter catalog/checkpoint mismatch')
            if 'original_parameters' in identity:compare(identity['original_parameters']/1e9,values[MODEL_NAMES[model]],'original_model_numel')
    text=document.read_text(encoding='utf-8')
    if 'multivideo16x4_rank_s163/best_checkpoint.pt' not in text or '**6,804,011**' not in text:
        raise ValueError('Original s163 student count missing')
    front=sum(t['numel'] for t in read(binding)['matched_front_tensors'])
    if front!=315098112:raise ValueError('Bound frontend count changed')
    values['LightGen']=(6804011+front)/1e9
    return dict(values_billion=values,baseline_catalog=sources,LightGen_student_numel=6804011,
        LightGen_frozen_frontend_numel=front,LightGen_total_numel=6804011+front,
        sources=[dict(path=str(p),sha256=h) for p,h in pins.items()],Qwen_GLM_are_nominal=True)


def read_workbook_cells(book,number):
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(book) as z:
        shared=[''.join(s.itertext()) for s in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)] if 'xl/sharedStrings.xml' in z.namelist() else []
        cells={}
        for c in ET.fromstring(z.read(f'xl/worksheets/sheet{number}.xml')).findall('.//s:sheetData/s:row/s:c',ns):
            if c.find('s:f',ns) is not None or c.get('t')=='e':raise ValueError('Only fixed source values supported')
            v=c.find('s:v',ns);inline=c.find('s:is',ns)
            value=v.text if v is not None else ''.join(inline.itertext()) if inline is not None else None
            if c.get('t')=='s':value=shared[int(value)]
            elif value is not None and c.get('t') not in ('inlineStr','str'):value=float(value)
            cells[c.get('r')]=value
        return cells


def appendix_table(detail,time16):
    groups=[('前端编码',['tokenizer','frontend','vision_entry'],'文本分词；视觉Embedding/位置编码/Adapter；视觉入口编码',
        '场内前处理双流fork/join，汇合后进入视觉主体；下一场前处理可与前一场主体并行。'),
        ('视觉光电处理',[f'{p}{i}' for i in range(1,4) for p in ('optics','oeo')],'视觉Router→Expert→Global三个光周期及OEO电子段',
        '同场三周期及其间电子段串行依赖；每周期16视频整幅光场联合传播。'),
        ('跨模态桥接',['merger','language_entry'],'Merger、序列组装、语言入口编码',
        '依赖视觉段输出；桥接完成后才能进入本场语言段，不能无依据重叠。'),
        ('语言光电处理',[f'{p}{i}' for i in range(4,7) for p in ('optics','oeo')],'语言Router→Expert→Global三个光周期及OEO电子段',
        '同场三周期及其间电子段串行依赖；可与后续场前处理流水重叠。'),
        ('解码与评分',['decoder'],'Decoder、读出与最终质量评分',
        '依赖本场语言段完成；评分fence完成后才能复用对应槽位。'),
        ('调度与同步',['scheduling'],'CPU提交/返回、观测队列、槽位等待与必要同步',
        '贯穿全窗，不是附加串行尾段；仅关键路径上的实际开销计入此行。')]
    by={r[0]:r for r in detail['rows'][:-1]};rows=[];members=[]
    for name,keys,operations,relation in groups:
        members+=keys;value=sum(by[PHASE_NAMES[k]][2] for k in keys)
        rows.append([name,operations,value,value/time16,relation])
    if set(members)!=set(PHASE_NAMES) or len(members)!=len(set(members)):raise ValueError('Appendix partition not exact')
    compare(sum(r[2] for r in rows),time16,'coarse_phase_closure')
    rows.append(['合计','三轮共6144视频、384组光场',time16,1.,'各行关键路径贡献可相加；不能将各并行阶段累计跨度相加。'])
    total=sum(r[3] for r in detail['rounds']);fields=sum(r[1] for r in detail['rounds'])
    compare(total/fields,time16,'three_round_normalization')
    notes=[
        '计时边界：t=0沿用原host_start，至最后CPU评分返回。窗口前模型加载、原图准备及批量H2D不计入此表。',
        '依赖规则：E(v)=d(v)+max{E(u):u为v的前驱}，E(origin)=0；整轮L=max_v E(v)。依赖含流顺序、真实提交和必要同步。',
        '总时间：T=ΣLᵣ/ΣFᵣ=('+ '＋'.join(f'{r[3]:.9f}' for r in detail['rounds'])+f')/384={time16:.9f} ms/16视频。',
        '表内贡献：选取每轮一条见证最长路径，按以上六块归类，三轮合计后除以384；不是六次独立实验耗时之和。',
        '并行只通过原DAG依赖和最长路径处理：已重叠部分不重复计费，也不再从总时间另减重叠跨度。',
        '双槽流水：场i前处理可与场i−1主体重叠；复用槽位前须等场i−2评分fence。跨流重叠不等于证明逐kernel占用并发。',
        '同场保持视觉3周期→Merger→语言3周期→Decoder依赖；16视频是整幅光场并行，不是把16路独立结果无损拼接。',
        '光周期各为0.714285714 ms SLM＋0.300000000 ms曝光＋0.030700000 ms读出；电子段用原计时，非物理光路直接实测。',
        '沿用批准范围：电子图构建作为部署成本不计入本表；FFT/ROI/电残差/H2D仍按原口径处理，无新增排除。',
        '首三场真实eager与两次capture输出保留，无额外预热、未挑最快轮。所有表值保留全精度，显示值相加可能有末位舍入差。']
    return dict(title='LightGen窗口端到端时间分解（每16视频/64帧）',headers=['模块','包含操作','端到端贡献/ms','占总时间','并行与串行关系'],
        rows=rows,notes=notes,round_headers=detail['round_headers'],rounds=detail['rounds'],
        member_keys=[g[1] for g in groups],whole_window_total_ms=total,fields=fields)


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
        for record in raw['records']:
            token=record['tokenizer'];events=record['events']
            if token['field']!=record['field'] or token['end']>min(e['host_submit_start'] for e in events if e['stream']=='frontend'):
                raise ValueError('Tokenizer must precede this field frontend')
            labels=[e['label'] for e in events]
            if not labels.index('frame_merger')<labels.index('language_routes.0')<labels.index('serial_router.forward'):
                raise ValueError('Original Merger/language Router order changed')
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
    parameters=parameter_metadata(data)
    for row in rows:row.insert(2,parameters['values_billion'][row[1]])
    detail=pooled_phase_table(phase_records,t)
    base=source.parent/'Temporal_LightGen_appendix_20261027_07'/BOOK_NAME
    base_sha='9b6700366cded705972f9a776d8e260370b5bafc1a52d27a8c07c01a9cc04e22'
    if sha(base)!=base_sha:raise ValueError('User-edited base workbook changed; re-inspect before editing')
    main_cells=read_workbook_cells(base,1)
    for i,row in enumerate([HEADERS]+rows,1):
        for j,value in enumerate(row):
            address=chr(65+j)+str(i)
            if address=='C2' and main_cells.get(address) is None:continue
            compare(value,main_cells.get(address),'base_main_'+address)
    return dict(status='raw_rechecked_fixed_values',headers=HEADERS,rows=rows,notes=NOTES,
        source_json=str(snapshot),source_json_sha256=sha(snapshot),source_manifest_sha256=MANIFEST_SHA,
        baseline_original_points_rechecked=36,LG_original_points_rechecked=3,LG_source_receipts=raw_sources,
        edited_XLSX_read=True,edited_XLSX_used_for_timing_calculation=False,
        explicit_extra_warmup_forwards=0,physical_optical_hardware_measured=False,
        energy_is_component_accounting=True,excel_formulas_requested=False,
        token_scope='LightGen: Qwen reference equivalent tokens; baselines: each original native token count',
        common_effective_tokens=common,common_tokens_per_16_videos=7136,
        LG_native_prompt_tokens_not_used_for_display=native,
        throughput_ratio_direction='LightGen Qwen-reference rate / baseline native rate',
        baseline_native_token_rates_preserved=True,throughput_ratios_equal_time_ratios_for_Qwen_only=True,
        cross_model_token_definitions_differ=True,phase_table=detail,appendix_table=detailed_appendix(phase_records,detail,t),parameters=parameters,
        base_workbook=str(base),base_workbook_sha256=base_sha,base_main_cells=main_cells,
        base_appendix_cells=read_workbook_cells(base,2),
        source_main_sheet_user_edits_preserved=True,base_workbook_used_for_presentation_only=True)


def audit(folder,visual_reviewed=False):
    folder=Path(folder).resolve();expected=read(folder/'static_values.json');book=folder/BOOK_NAME
    # Recheck the raw data independently again; no workbook is used as an input.
    replay_expected=dict(expected)
    if replay_expected.get('base_workbook_used_for_presentation_only'):
        # Explicitly normalize the documented legacy access flag, never raw values.
        replay_expected['edited_XLSX_read']=True
        replay_expected['edited_XLSX_used_for_timing_calculation']=False
    compare(replay_expected,prepare(Path(expected['source_json']).parent),'independent_raw_replay')
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(book) as z:
        names=[n.get('name') for n in ET.fromstring(z.read('xl/workbook.xml')).findall('s:sheets/s:sheet',ns)]
        if names!=['统一结果','LightGen附录计时']:raise ValueError('Exactly two fixed-value sheets required')
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
        for address,value in expected['base_main_cells'].items():compare(value,cells.get(address),'preserved_'+address)
        compare({k:v for k,v in expected['base_main_cells'].items() if v is not None},
                {k:v for k,v in cells.items() if v is not None},'main_values_exactly_preserved')
        phase=expected['appendix_table'];cells=parse_sheet(2)
        compare(phase['title'],cells.get('A1'))
        compare(phase['scope'],cells.get('A2'))
        for r,row in enumerate([phase['headers']]+phase['rows'],3):
            for i,value in enumerate(row):compare(value,cells.get(chr(65+i)+str(r)),f'phase_cell_{r}_{i}')
        notes_start=len(phase['rows'])+6
        for i,note in enumerate(phase['notes'],notes_start):compare(note,cells.get('A'+str(i)))
        accounted=[r for r,t in zip(phase['rows'],phase['row_types']) if t=='accounted']
        compare(math.fsum(r[2] for r in accounted),expected['rows'][0][5],'saved_phase_sum_equals_main_time')
        compare(math.fsum(r['control_ms16'] for r in phase['phase_accounting']),phase['old_scheduling_ms16'],'control_redistributed_exactly')
        compare(math.fsum(r[1] for r in phase['module_rows']),expected['rows'][0][5],'saved_module_subtotals_close')
        # New columns, canonical names and grouping must not change any timing.
        original=expected['base_appendix_cells']
        original_rows=[[4,5],*[ [r] for r in [7,8,9,10,11,12,14,15,17,18,19,20,21,22,24,33] ]]
        for new_row,old_rows in enumerate(original_rows,4):
            compare(math.fsum(original[f'F{r}'] for r in old_rows),cells.get(f'C{new_row}'),f'unchanged_equivalent_time_{old_rows}')
        if len(accounted)!=17 or len([r for r in accounted if r[0]=='调度与同步'])!=1:
            raise ValueError('Exactly 17 accounted groups and one aggregate control row required')
        if any(v is not None and re.match(r'[E-Z]',a) for a,v in cells.items()):
            raise ValueError('Remove obsolete timing columns from presentation')
    if not visual_reviewed:raise ValueError('Review the one-sheet render before sealing')
    main_render_sha=sha(folder/'preview.png')
    if main_render_sha!='93dcaa7bbc6bd204cb97dd6380f5195e1d6485ab741f71c8ce57643d6fc81dbb':
        raise ValueError('Unchanged main sheet render differs from the actual user-edited base')
    document=folder/'Temporal_LightGen_附录计时与统计方法_20261027.docx'
    doc_audit=read(folder/'document_audit.json')
    if (doc_audit['status']!='passed' or doc_audit['document_sha256']!=sha(document)
            or doc_audit['source_values_sha256']!=sha(folder/'static_values.json')):
        raise ValueError('The matching Word appendix must pass native-render and value checks')
    result=dict(status='passed',rows=13,sheets=2,formulas=0,external_links=0,raw_points_rechecked=39,
        all_saved_values_independently_verified=True,edited_XLSX_used_for_timing_calculation=False,
        edited_XLSX_imported_for_main_preservation=True,
        deletion_cannot_trigger_reference_recalculation=True,visual_reviewed=True,workbook_sha256=sha(book),
        common_effective_tokens=expected['common_effective_tokens'],
        baseline_native_token_rates_preserved=True,throughput_ratios_equal_time_ratios_for_Qwen_only=True,
        cross_model_token_definitions_differ=True,phase_table_closes_to_main_time=True,
        original_t0_retained=True,phase_overlap_not_double_counted=True,parameter_sources_verified=True,
        appendix_blocks=6,filename_contains_fixed_values=False,source_main_user_edits_preserved=True,
        CPU_control_redistributed_not_retimed=True,missing_component_timings_not_fabricated=True,
        main_sheet_render_pixel_identical=True,main_sheet_render_sha256=main_render_sha,
        preprocessing_accounted_parts=1,preprocessing_mapping_merged_once=True,
        all_equivalent_times_unchanged=True,global_control_merged_once=True,
        language_field_adapter_not_text_embedding=True,original_execution_order_checked_fields=384,
        matching_Word_appendix_audited=True,Word_pages=doc_audit['pages'])
    with (folder/'saved_values_audit.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=True,indent=2)
    with (folder/'delivery_manifest.json').open('x',encoding='utf8') as f:
        json.dump(dict(status='sealed',files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in
            (book,document,folder/'static_values.json',folder/'saved_values_audit.json',folder/'document_audit.json')]),f,ensure_ascii=True,indent=2)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','audit']);p.add_argument('source',type=Path)
    p.add_argument('--out',type=Path);p.add_argument('--visual-reviewed',action='store_true');a=p.parse_args()
    if a.mode=='prepare':
        result=prepare(a.source)
        with a.out.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=True,indent=2)
        print(json.dumps(dict(status='passed',rows=len(result['rows']),LG_time16_ms=result['rows'][0][5]),ensure_ascii=True))
    else:print(json.dumps(audit(a.source,a.visual_reviewed),ensure_ascii=True))
