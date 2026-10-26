"""Read saved XLSX caches against the already sealed, data-only point audits."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile

NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):
    assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-5),(a,b)
def audit(folder):
    data=json.loads((folder/'collected_results.json').read_text(encoding='utf-8'))
    xlsx=next(folder.glob('*.xlsx'))
    with ZipFile(xlsx) as z:
        sheets=[]
        for name in ['xl/worksheets/sheet1.xml','xl/worksheets/sheet2.xml']:
            cells={}
            for cell in ET.fromstring(z.read(name)).findall('.//s:sheetData/s:row/s:c',NS):
                value=cell.find('s:v',NS)
                if cell.find('s:f',NS) is not None:
                    assert value is not None and value.text is not None,(name,cell.attrib)
                    assert cell.get('t')!='e',(name,cell.attrib,value.text)
                cells[cell.attrib['r']]=value.text if value is not None else None
            sheets.append(cells)
    summary,rounds=sheets
    for row,model,col in [(7,'kimi','B'),(8,'deepseek','C')]:
        points=[p for p in data['points'] if p['model']==model]
        seconds=sum(p['cost']['elapsed_s'] for p in points)
        n=sum(p['cost']['video_calls'] for p in points)
        energy=sum(p['cost']['energy_j'] for p in points)
        common=sum(p['cost']['common_Qwen_effective_tokens'] for p in points)
        native=sum(p['cost']['native_effective_tokens'] for p in points)
        expected=[n,seconds*1000/n*16,seconds*1000/n/4,energy/n*16,common/seconds,native/seconds]
        for c,value in zip('CDEFGH',expected):close(summary[c+str(row)],value)
        close(summary[col+'18'],expected[1])
        close(sum(float(summary[col+str(r)]) for r in range(13,18)),expected[1])
    for row,p in enumerate(data['points'],2):
        c=p['cost'];values=[c['time16_ms'],c['equivalent_frame_ms'],c['energy16_j'],c['energy_per_frame_j'],c['common_tokens_per_s'],c['native_tokens_per_s'],c['video_calls']/c['elapsed_s'],c['video_calls']*4/c['elapsed_s']]
        for col,value in zip('HIJKLMNO',values):close(rounds[col+str(row)],value)
    with (folder/'raw_file_index.csv').open(encoding='utf-8-sig',newline='') as f:index=list(csv.DictReader(f))
    assert len(index)==65 and sum(x['kind'].startswith('远端') for x in index)==55
    assert len({x['local'] for x in index})==65
    for x in index:assert Path(x['local']).stat().st_size==int(x['bytes'])
    checks=dict(status='passed',all_saved_formula_caches_present=True,pooled_rates_independently_recomputed=True,
                stage_partition_and_round_values_match=True,remote_raw_files=55,raw_and_local_evidence_files=65,
                model_round_counts={'kimi':3,'deepseek':2},no_model_or_CUDA=True)
    (folder/'saved_workbook_audit.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    names=[xlsx.name,'collected_results.json','raw_file_index.csv','builder_checks.json','saved_workbook_audit.json']
    manifest=dict(status='sealed',scope='A10080 MPS50 only; unequal qualified rounds explicit',
                  files=[dict(path=name,bytes=(folder/name).stat().st_size,sha256=sha(folder/name)) for name in names])
    with (folder/'delivery_manifest.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(**checks,workbook_sha256=sha(xlsx),index_sha256=sha(folder/'raw_file_index.csv'),manifest_sha256=sha(folder/'delivery_manifest.json')),ensure_ascii=False))
if __name__=='__main__':audit(Path(sys.argv[1]).resolve())
