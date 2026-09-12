"""GPU-only model/first-stage BMP smoke test. No SLM/camera access or accuracy claim."""
import argparse,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(ROOT/'compat_abo'))
    import common
    common.ROOT=ROOT
    import backend
    from patterns import amplitude,save
    from memory import memory_report
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback')
    report={'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
            'mode':'simulation and first-stage preparation only; no real CCD result','complete':False}
    b=backend.create('cuda')
    spec=importlib.util.spec_from_file_location('abo_sample_definitions',ROOT/'compat_abo/run.py')
    task=importlib.util.module_from_spec(spec);spec.loader.exec_module(task)
    samples=task.samples(4);selected=[samples[0]]+[s for s in samples if s['kind']=='image']
    t=time.perf_counter();vectors=[]
    for s in (selected[0],selected[1]):
        v=backend.forward(b,s,release=True)
        if not np.isfinite(v).all():raise ValueError('Non-finite simulated feature')
        vectors.append({'id':s['id'],'shape':list(v.shape),'norm':float(np.linalg.norm(v))})
    report['simulation_samples']=vectors;report['simulation_total_s']=time.perf_counter()-t
    from abo_dual.backend import NeedCapture
    c=json.loads((ROOT/'LAB.local.json').read_text(encoding='utf-8-sig'))
    b.guard_optics();entries=[]
    for s in selected[1:]:
        try:backend.forward(b,s,release=True)
        except NeedCapture as e:
            field=e.amplitude
            if field.shape!=(224,224):raise ValueError('Router amplitude geometry mismatch')
            active=np.zeros((478,478),np.float32);active[127:351,127:351]=field
            bmp,encoding=amplitude(active,c);file=a.out/(s['id']+'.bmp');save(file,bmp)
            entries.append({'id':s['id'],'file':file.name,'sha256':common.sha(file),'encoding':encoding})
        else:raise RuntimeError('CCD stage guard did not intercept')
    report.update(complete=True,prepared_bmps=entries,memory=memory_report(b),checkpoint_sha256=common.CHECKPOINT_SHA)
    common.write(a.out/'report.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
