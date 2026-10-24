"""Independent read-only OOXML verification of all nine saved plotting workbooks."""
import hashlib,json,math
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

NS={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def cells(z,filename,shared):
    values={}
    for c in ET.fromstring(z.read(filename)).findall('.//m:sheetData/m:row/m:c',NS):
        if c.find('m:f',NS) is not None:raise ValueError('Unexpected formula in fixed measured-data export')
        node=c.find('m:v',NS);value=node.text if node is not None else None
        if c.get('t')=='s':value=shared[int(value)]
        elif c.get('t')=='inlineStr':value=''.join(t.text or '' for t in c.findall('.//m:t',NS))
        elif c.get('t')=='str':pass
        elif value is not None:value=float(value)
        values[c.get('r')]=value
    return values


def col(index):
    text=''
    while index:
        index,rest=divmod(index-1,26);text=chr(65+rest)+text
    return text


def main():
    folder=Path(__file__).resolve().parent/'runs/outputs/t19_final_plot_data_20261024'
    data=json.loads((folder/'plot_data.json').read_text(encoding='utf-8'));records=[]
    for spec in data['tasks']:
        path=folder/spec['filename']
        with ZipFile(path) as z:
            shared=[]
            if 'xl/sharedStrings.xml' in z.namelist():shared=[''.join(t.text or '' for t in si.findall('.//m:t',NS)) for si in ET.fromstring(z.read('xl/sharedStrings.xml'))]
            names=[s.get('name') for s in ET.fromstring(z.read('xl/workbook.xml')).findall('m:sheets/m:sheet',NS)]
            assert names==['整体指标','逐测试样本']
            overall=cells(z,'xl/worksheets/sheet1.xml',shared);samples=cells(z,'xl/worksheets/sheet2.xml',shared)
            expected=[spec['sample_headers'],*spec['sample_rows']]
            for r,row in enumerate(expected,1):
                for c,value in enumerate(row,1):
                    actual=samples.get(col(c)+str(r))
                    if isinstance(value,(int,float)):assert actual is not None and math.isclose(actual,value,abs_tol=1e-12,rel_tol=1e-13),(spec['task'],r,c,actual,value)
                    else:assert actual==value,(spec['task'],r,c,actual,value)
            assert max(int(''.join(c for c in a if c.isdigit())) for a in samples)==len(expected)
            if spec['task']=='lsp' and overall.get('A5')=='部位':
                metrics=spec['metrics'];counts=metrics['per_joint_counts'];pck=metrics['per_joint_pck_at_0.2_torso']
                grouped=[('Overall (14 joints)',metrics['pck_at_0.2_torso'],sum(counts.values()))]
                for joint in ['shoulder','elbow','wrist','hip','knee','ankle']:
                    keys=['left_'+joint,'right_'+joint];n=sum(counts[k] for k in keys)
                    grouped.append((joint.title(),sum(pck[k]*counts[k] for k in keys)/n,n))
                for i,(name,value,n) in enumerate(grouped,6):
                    assert overall['A'+str(i)]==name and math.isclose(overall['B'+str(i)],value,abs_tol=1e-12) and overall['C'+str(i)]==n
                for i,(_,key,value) in enumerate(spec['overall_rows'][1:7],17):
                    assert overall['A'+str(i)]==key and math.isclose(overall['B'+str(i)],value,abs_tol=1e-12,rel_tol=1e-13)
            else:
                for i,(_,key,value) in enumerate(spec['overall_rows'],6):
                    assert overall['B'+str(i)]==key and math.isclose(overall['C'+str(i)],value,abs_tol=1e-12,rel_tol=1e-13)
        records.append(dict(task=spec['task'],file=spec['filename'],rows=len(expected)-1,columns=len(expected[0]),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),all_saved_cells_match=True))
    report=dict(status='passed',workbooks=len(records),worksheets=18,total_test_records=sum(r['rows'] for r in records),records=records)
    (folder/'verification/saved_xlsx_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='passed',workbooks=9,worksheets=18,test_records=report['total_test_records'])))


if __name__=='__main__':main()
