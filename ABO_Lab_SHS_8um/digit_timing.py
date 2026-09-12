"""Digit switching verification and additive host timing; NOT digit recognition accuracy."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from slm_camera import Controller
from capture import save_json

ROOT=Path(__file__).resolve().parent

def stats(values):
    a=np.asarray(values,float)
    return {'mean':float(a.mean()),'min':float(a.min()),'max':float(a.max()),
            'p50':float(np.percentile(a,50)),'p95':float(np.percentile(a,95))}

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='LAB.local.json')
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--wait-ms',type=float,help='This diagnostic only; does not edit LAB.local.json')
    p.add_argument('--repeats',type=int,default=5);a=p.parse_args()
    c=json.loads(Path(a.config).read_text(encoding='utf-8-sig'))
    if not 1<=a.repeats<=10:raise ValueError('1..10 repeats only')
    if a.wait_ms is not None:
        if not 0<=a.wait_ms<=1000:raise ValueError('0..1000 ms only')
        c['settle_delay_ms']=a.wait_ms
    a.out.mkdir(parents=True,exist_ok=False)
    paths={d:ROOT/f'generated/phase_inverted/cal/A_DIGIT_{d}.bmp' for d in range(4)}
    for path in paths.values():
        if not path.is_file():raise FileNotFoundError(path)
    report={'complete':False,'scope':'Prepared BMP -> raw Mono8 in host memory. No recognition model, geometry warp or PNG in measured loop.',
            'config':c,'phase':'manual, unchanged','pattern_sha256':{d:hashlib.sha256(p.read_bytes()).hexdigest() for d,p in paths.items()},
            'source_commit':json.loads((ROOT/'CODE_MANIFEST.json').read_text())['commit']}
    refs={};frames=[];rows=[]
    with Controller(c) as hw:
        report['devices']=hw.info;configured=c['settle_delay_ms'];c['settle_delay_ms']=400
        for d in range(4):refs[d]=hw.capture(paths[d])[0]
        c['settle_delay_ms']=configured
        sequence=[0,1,2,3]*a.repeats
        started=time.perf_counter()
        for d in sequence:
            frame,meta=hw.capture(paths[d]);frames.append(frame);rows.append(dict(digit=d,**meta))
        elapsed=time.perf_counter()-started
    # Analyze AFTER the measured loop. Keep 24 raw frames only for this bounded audit.
    def unit(x):
        x=x.astype(np.float32).ravel();x-=x.mean();return x/max(float(np.linalg.norm(x)),1e-12)
    templates=np.stack([unit(refs[d]) for d in range(4)])
    for i,(frame,row) in enumerate(zip(frames,rows)):
        pcc=templates@unit(frame);pred=int(np.argmax(pcc));row.update(predicted_reference_digit=pred,
            correct=pred==row['digit'],pcc_to_all=pcc.tolist(),same_pcc=float(pcc[row['digit']]),
            p99=float(np.percentile(frame,99)),saturation_fraction=float((frame==255).mean()))
        t=time.perf_counter();Image.fromarray(frame).save(a.out/f'{i:02d}_digit{row["digit"]}.png',compress_level=1)
        row['diagnostic_full_png_save_ms']=(time.perf_counter()-t)*1000
    for d,frame in refs.items():Image.fromarray(frame).save(a.out/f'reference_{d}.png',compress_level=1)
    keys=['bmp_validate_ms','slm_preload_ms','slm_show_to_visible_ms','settle_actual_ms','final_fresh_ms','capture_total_ms','diagnostic_full_png_save_ms']
    report.update(complete=True,camera_restored=True,rows=rows,correct_count=sum(r['correct'] for r in rows),
        measured_frames=len(rows),loop_elapsed_s=elapsed,raw_ready_cycles_per_s=len(rows)/elapsed,
        timing_ms={k:stats([r[k] for r in rows]) for k in keys},minimum_same_pcc=min(r['same_pcc'] for r in rows))
    save_json(a.out/'report.json',report)
    print(json.dumps({k:report[k] for k in ['complete','correct_count','measured_frames','raw_ready_cycles_per_s','minimum_same_pcc','timing_ms']},indent=2),flush=True)

if __name__=='__main__':main()
