"""Preserve all dense timing results; formal delay remains 200 ms."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    source=ROOT/'results/dense_timing_20260912/report.json'
    d=json.loads(source.read_text(encoding='utf-8'));rows=[]
    for delay in [100,120,140,160,180,200]:
        fs=[f for f in d['frames'] if f['label'].startswith(f'd{delay}_')]
        rows.append({'delay_ms':delay,'n':len(fs),'correct':sum(f['same_reference']['pcc']>f['opposite_reference']['pcc'] for f in fs),
                     'minimum_same_pcc':min(f['same_reference']['pcc'] for f in fs)})
    report={'formal_delay_ms':200,'decision':'User requested conservative 200 ms; shorter passes are not proof of reliability',
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'left_right_scan':rows,'digit_retests':[]}
    for delay in [120,180]:
        p=ROOT/f'results/digits{delay}_20260912/report.json';r=json.loads(p.read_text(encoding='utf-8'))
        report['digit_retests'].append({'delay_ms':delay,'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
           **{k:r[k] for k in ['correct_count','measured_frames','minimum_same_pcc','raw_ready_cycles_per_s']}})
    out=ROOT/'reports/timing_20260912/dense_summary.json';out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(out)

if __name__=='__main__':main()
