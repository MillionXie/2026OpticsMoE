"""Rebuild a fixed-value display from audited raw records, never edited XLSX."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET
from collect_energy_summary_20261026 import read,sha,point,aggregate
from audit_lgvq_async_point_20261026 import compare
from lgvq_phase_ledger_20261026 import build as phase_build

MANIFEST_SHA='dd709c4d527b05e5ff1e4e681e5969efbc33200f90542b7c4b899ee18b576be2'
HEADERS=['平台','模型','MPS/%','轮次','每16视频/ms','等效单帧/ms','每16视频/J','等效单帧/J','tokens/s（自身）','LightGen时间加速比','LightGen能效比']
NOTES=['LightGen：电子阶段计时＋固定光周期，不含电子图构建；能量按组件核算，非光路直接实测。',
       'tokens/s按各模型自身有效token。倍数为各行时间、能量除以LightGen对应值。40GB与80GB并非同卡受控比较。']


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
    seconds=energy=native=0.;calls=0;workers=set();raw_sources=[]
    for r in data['LG_rounds']:
        folder=Path(r['path']);point(folder,'LG_async')
        if read(folder/'local_native_parallel_audit01.json')['status']!='passed':raise ValueError('Parallel audit missing')
        phase=phase_build(folder,True,exclude_graph_construction=True)
        compare(read(folder/'local_deployed_phase_ledger_v3_01.json'),phase,'v3_replay')
        identity=read(folder/'identity.json');raw=read(folder/'async_raw_timing.json');cost=read(folder/'cost_report.json')
        workers.add(identity['pid'])
        if raw['explicit_warmup_forwards']!=0:raise ValueError('No extra warmup permitted')
        seconds+=phase['fixed_optics']['wall_ms16']*raw['fields']/1000
        energy+=phase['fixed_optics_component_energy']['energy_j'];native+=cost['native_effective_tokens'];calls+=raw['video_calls']
        raw_sources.append(dict(leaf=str(folder),raw_sha256=sha(folder/'async_raw_timing.json'),
            audit_sha256=sha(folder/'local_independent_async_audit01.json'),ledger_sha256=sha(folder/'local_deployed_phase_ledger_v3_01.json')))
    if len(workers)!=3 or calls!=6144:raise ValueError('Three distinct complete processes required')
    t=seconds*1000*16/calls;e=energy*16/calls;rate=native/seconds
    compare(data['LG']['fixed_optics_counterfactual_time16_ms'],t)
    compare(data['LG']['fixed_optics_component_proxy_energy16_j'],e)
    compare(data['LG']['projected_native_tokens_per_s'],rate)
    rows=[['40GB PCIe / 250W','LightGen',100,3,t,t/64,e,e/64,rate,1.,1.]]
    labels={'clip':'CLIP','glm':'GLM','qwen':'Qwen','yolo':'YOLO','deepseek':'DeepSeek完整','kimi':'Kimi'}
    for g in groups:
        rows.append(['40GB PCIe / 250W' if g['platform']=='A10040GB_PCIE_250W' else '80GB SXM4 / 400W',
            labels[g['model']],g['cap'],g['rounds'],g['time16_ms'],g['equivalent_frame_ms'],g['energy16_j'],g['energy_per_frame_j'],
            g['pooled_native_tokens_per_s'] if g['pooled_native_tokens_per_s'] is not None else '不适用',
            g['time16_ms']/t,g['energy16_j']/e])
    return dict(status='raw_rechecked_fixed_values',headers=HEADERS,rows=rows,notes=NOTES,
        source_json=str(snapshot),source_json_sha256=sha(snapshot),source_manifest_sha256=MANIFEST_SHA,
        baseline_original_points_rechecked=36,LG_original_points_rechecked=3,LG_source_receipts=raw_sources,
        edited_XLSX_read=False,explicit_extra_warmup_forwards=0,physical_optical_hardware_measured=False,
        energy_is_component_accounting=True,excel_formulas_requested=False)


def audit(folder,visual_reviewed=False):
    folder=Path(folder).resolve();expected=read(folder/'static_values.json');book=folder/'Temporal_LightGen与baseline_固定数值_20261027.xlsx'
    # Recheck the raw data independently again; no workbook is used as an input.
    compare(expected,prepare(Path(expected['source_json']).parent),'independent_raw_replay')
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(book) as z:
        names=[n.get('name') for n in ET.fromstring(z.read('xl/workbook.xml')).findall('s:sheets/s:sheet',ns)]
        if names!=['统一结果']:raise ValueError('Exactly one standalone sheet required')
        if any(n.startswith('xl/externalLinks/') for n in z.namelist()):raise ValueError('External links forbidden')
        shared=[''.join(s.itertext()) for s in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)] if 'xl/sharedStrings.xml' in z.namelist() else []
        cells={}
        for c in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//s:sheetData/s:row/s:c',ns):
            if c.find('s:f',ns) is not None or c.get('t')=='e':raise ValueError('All cells must be fixed values without errors')
            v=c.find('s:v',ns);inline=c.find('s:is',ns)
            value=v.text if v is not None else ''.join(inline.itertext()) if inline is not None else None
            if c.get('t')=='s':value=shared[int(value)]
            elif value is not None and c.get('t') not in ('inlineStr','str'):value=float(value)
            cells[c.get('r')]=value
        for r,row in enumerate([expected['headers']]+expected['rows'],1):
            for i,value in enumerate(row):compare(value,cells.get(chr(65+i)+str(r)),f'cell_{r}_{i}')
        for i,note in enumerate(expected['notes'],16):compare(note,cells.get('A'+str(i)))
        if any(isinstance(v,str) and ('投影' in v or '软件' in v) for v in cells.values()):raise ValueError('Presentation must contain only the requested LG metric')
    if not visual_reviewed:raise ValueError('Review the one-sheet render before sealing')
    result=dict(status='passed',rows=13,sheets=1,formulas=0,external_links=0,raw_points_rechecked=39,
        all_saved_values_independently_verified=True,edited_XLSX_used_as_source=False,
        deletion_cannot_trigger_reference_recalculation=True,visual_reviewed=True,workbook_sha256=sha(book))
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
