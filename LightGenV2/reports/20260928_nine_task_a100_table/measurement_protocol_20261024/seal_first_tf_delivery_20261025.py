"""Seal the three reviewed artifacts and precise raw provenance index."""
from pathlib import Path
import csv,hashlib,json,zipfile
from xml.etree import ElementTree as ET
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
OUT=HERE/'outputs/01a061e2-6f88-7913-87f5-b632000e4c83/first_tf_complete_20261025_01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (OUT/'delivery_manifest.json').exists()
d=json.loads((OUT/'core_data.json').read_text(encoding='utf8'));assert not any(x['pending'] for x in d['mainRecords'])
audit=json.loads((OUT/'independent_delivery_audit.json').read_text());assert audit['status']=='passed' and audit['independently_checked_unique_points']==90
doc=OUT/'A100_首计算模块时间能量与倍数说明_20261025.docx'
with zipfile.ZipFile(doc) as z:
    xml=ET.fromstring(z.read('word/document.xml'));text=''.join(xml.itertext())
    assert '补测中' not in text and '待测' not in text
    equations=len(xml.findall('.//{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath'))
    assert equations==12,equations
from pypdf import PdfReader
pages=len(PdfReader(OUT/'docx_qa/methods.pdf').pages)
assert pages==4 and all((OUT/f'docx_qa/page-{i}.png').exists() for i in range(1,pages+1))
index=OUT/'raw_file_index.csv'
with index.open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.writer(f);writer.writerow(['相对原路径','本地完整路径','字节','SHA256','本版核验方式','不计入指标的原因'])
    for r in sorted(d['raw_files'],key=lambda x:x['file']):
        writer.writerow([r['file'],str(REPO/r['file']),r['bytes'],r['sha256'],r['verification'],r.get('excluded_reason','')])
artifacts=[OUT/'A100_九任务_首计算模块核心对照_20261025.xlsx',OUT/'A100_Temporal多基线_首计算模块核心对照_20261025.xlsx',doc]
files=artifacts+[index,OUT/'core_data.json',OUT/'independent_delivery_audit.json']
manifest=dict(status='complete',scope=d['scope'],main_cost_cells=18,new_baseline_first_TF_points=21,
    preserved_formal_full_TEST=True,old_full_input_deliveries_preserved=True,physical_optical_OP=None,
    core_OP='未核定，不套用旧全图分子',raw_index_files=len(d['raw_files']),editable_Word_equations=equations,
    Word_rendered_pages=pages,visual_review='All workbook sheets and every final Word page inspected',
    supplement_suites=d['supplement_suites'],excluded_partial_run=d['rejected_partial_run'],
    artifacts=[dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in files])
(OUT/'delivery_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(manifest,ensure_ascii=False))
