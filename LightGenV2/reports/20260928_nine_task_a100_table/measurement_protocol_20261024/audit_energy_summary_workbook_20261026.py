"""Read saved XLSX caches independently; never author Excel or execute models."""
import argparse
import csv
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile
from collect_energy_summary_20261026 import collect, read, sha

NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
STAGES=('CPU_image_preparation_excluded_ms','H2D_excluded_ms','CPU_tokenizer_retained_ms',
        'entry_dispatch_retained_ms','full_GPU_model_to_score_host_ms','intermediate_CPU_output_retained_ms',
        'intervideo_dispatch_retained_ms','final_CPU_output_outside_primary_ms',
        'GPU_model_entry_to_first_TF_host_ms','first_TF_to_score_host_ms')


def close(a,b):
    if a is None or not math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-5):
        raise ValueError('Saved workbook cache disagrees: '+str((a,b)))


def column(n):
    s='';n+=1
    while n:
        n,r=divmod(n-1,26);s=chr(65+r)+s
    return s


def cached_sheets(workbook):
    sheets=[];formula_count=0
    with ZipFile(workbook) as z:
        names=[n.attrib['name'] for n in ET.fromstring(z.read('xl/workbook.xml')).findall('s:sheets/s:sheet',NS)]
        if names!=['结果与阶段','轮次原值','原始文件索引']:
            raise ValueError('Unexpected exported worksheet order/names')
        for i in range(1,4):
            cells={}
            for cell in ET.fromstring(z.read(f'xl/worksheets/sheet{i}.xml')).findall('.//s:sheetData/s:row/s:c',NS):
                value=cell.find('s:v',NS)
                if cell.find('s:f',NS) is not None:
                    formula_count+=1
                    if value is None or value.text is None or cell.get('t')=='e':
                        raise ValueError('Missing or erroneous saved formula cache')
                if cell.get('t')=='e':raise ValueError('Exported Excel error cell')
                cells[cell.attrib['r']]=None if value is None else value.text
            sheets.append(cells)
    return sheets,formula_count


def verify_values(data,sheets):
    summary,rounds,_=sheets
    row=2
    for i,g in enumerate(data['groups']):
        points=sorted((p for p in data['rounds'] if (p['platform'],p['model'],p['cap'])==
                       (g['platform'],g['model'],g['cap'])),key=lambda p:p['repeat'])
        n=sum(p['calls'] for p in points);seconds=sum(p['elapsed_s'] for p in points)
        energy=sum(p['energy_j'] for p in points);common=sum(p['common_effective_tokens'] for p in points)
        values=dict(E=seconds*1000/n*16,F=seconds*1000/(n*4),G=energy/n*16,H=energy/(n*4),I=common/seconds,K=n)
        if points[0]['native_effective_tokens'] is not None:
            values['J']=sum(p['native_effective_tokens'] for p in points)/seconds
        for c,value in values.items():close(summary.get(c+str(i+7)),value)
        for j,key in enumerate(STAGES):
            if key in points[0]['stage_totals_ms']:
                close(summary.get(column(j+3)+str(i+49)),sum(p['stage_totals_ms'][key] for p in points)*16/n)
        charged=sum(sum(p['stage_totals_ms'][k] for k in STAGES[2:7]) for p in points)
        close(charged,seconds*1000)
        for p in points:
            for c,value in [('E',p['calls']),('F',p['elapsed_s']),('G',p['energy_j']),
                            ('J',p['time16_ms']),('K',p['energy16_j']),('L',p['common_effective_tokens']/p['elapsed_s'])]:
                close(rounds.get(c+str(row)),value)
            if p['native_effective_tokens'] is not None:
                close(rounds.get('M'+str(row)),p['native_effective_tokens']/p['elapsed_s'])
            row+=1
        lg=data['LG'];r=i+34
        close(summary.get('C'+str(r)),values['E']/lg['measured_software_time16_ms'])
        close(summary.get('D'+str(r)),values['E']/lg['fixed_optics_counterfactual_time16_ms'])
        close(summary.get('E'+str(r)),values['G']/lg['fixed_optics_component_proxy_energy16_j'])
    lg=data['LG']
    for cell,key in [('B26','measured_software_time16_ms'),('C26','measured_software_board_FFT_energy16_j'),
                     ('B27','fixed_optics_counterfactual_time16_ms'),('C27','fixed_optics_component_proxy_energy16_j'),
                     ('D27','projected_common_tokens_per_s'),('E27','projected_native_tokens_per_s')]:
        close(summary.get(cell),lg[key])


def audit(folder,seal=False,visual_reviewed=False):
    folder=Path(folder).resolve();data=read(folder/'collected_results.json')
    if data['status']!='final_qualified_inputs':raise ValueError('No partial delivery can be sealed as final')
    replay=collect([Path(p['path']) for p in data['rounds']],Path(data['LG']['path']))
    for key in ('status','lab_complete_points','groups','LG','rounds','raw_file_index','comparisons'):
        if replay[key]!=data[key]:raise ValueError('Current qualified inputs differ: '+key)
    books=list(folder.glob('*.xlsx'))
    if len(books)!=1:raise ValueError('Exactly one final workbook required')
    sheets,formula_count=cached_sheets(books[0]);verify_values(data,sheets)
    with (folder/'raw_file_index.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)!=len(data['raw_file_index']):raise ValueError('Raw index row count differs')
    for row,expected in zip(rows,data['raw_file_index']):
        for key in ('role','local_path','remote_path','sha256'):
            if row[key]!=(expected[key] or ''):raise ValueError('Raw index identity differs: '+key)
        if int(row['bytes'])!=expected['bytes']:raise ValueError('Raw index size differs')
    checks=read(folder/'builder_checks.json')
    if sha(books[0])!=checks['workbook_sha256'] or not checks['recalculation_input_mutation_restored']:
        raise ValueError('Builder output/restoration evidence differs')
    result=dict(status='passed',formula_caches_verified=formula_count,
        pooled_rates_and_phase_partition_independently_recomputed=True,LG_actual_and_projection_separately_verified=True,
        raw_file_index_rows=len(rows),all_original_SHA_size_rechecked=True,model_or_CUDA_calls=0,
        visual_review_confirmed=visual_reviewed,workbook_sha256=sha(books[0]))
    if seal and not visual_reviewed:raise ValueError('Review every rendered worksheet before sealing')
    with (folder/'saved_workbook_audit.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    if seal:
        names=[books[0].name,'collected_results.json','raw_file_index.csv','builder_checks.json','saved_workbook_audit.json']
        manifest=dict(status='sealed',physical_optical_hardware_measured=False,
            files=[dict(path=name,bytes=(folder/name).stat().st_size,sha256=sha(folder/name)) for name in names])
        with (folder/'delivery_manifest.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    parser.add_argument('--seal',action='store_true');parser.add_argument('--visual-reviewed',action='store_true')
    args=parser.parse_args();print(json.dumps(audit(args.folder,args.seal,args.visual_reviewed),ensure_ascii=False))
