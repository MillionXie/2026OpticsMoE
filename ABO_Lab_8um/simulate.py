"""Verify compact student, export its six phases, evaluate fixed test samples."""
import argparse
import time
import numpy as np
from common import ROOT,write,CHECKPOINT_SHA

def main():
    p=argparse.ArgumentParser(); p.add_argument('--device',default='auto'); p.add_argument('--limit',type=int,default=4)
    p.add_argument('--export-native',action='store_true'); p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--output',type=str,default='compact_smoke'); args=p.parse_args()
    from backend import create,phases,optical_contract
    from run import samples
    import torch
    start=time.perf_counter(); b=create(args.device,args.export_native)
    out=ROOT/'results'/args.output; out.mkdir(parents=True,exist_ok=True)
    phase_dir=ROOT/'assets/phases'; phase_dir.mkdir(parents=True,exist_ok=True)
    phase_errors={}
    for stage,arr in phases(b).items():
        target=phase_dir/(stage+'.npy')
        if target.exists():
            saved=np.load(target,allow_pickle=False)
            phase_errors[stage]=float(np.max(np.abs(saved-arr)))
            if saved.shape!=arr.shape or not np.allclose(saved,arr,atol=5e-6,rtol=1e-6):
                raise ValueError('Packaged phase differs from checkpoint: '+stage)
        else: np.save(target,arr,allow_pickle=False)
    if not (phase_dir/'contract.json').exists(): write(phase_dir/'contract.json',optical_contract(b))
    ss=samples(args.limit)
    titles=tuple(b.m.Title(s['label'],str(s['label']),s['text']) for s in ss if s['kind']=='title')
    images=b.m._read_samples(ROOT/'assets/test_dataset/test.csv',ROOT/'assets/test_dataset','test')
    images=images[:args.limit or None]
    contract=b.m.Contract((),images,titles,ROOT/'assets/test_dataset',{})
    settings=b.settings; settings.inference_batch_size=args.batch_size; settings.num_workers=0; settings.output_dir=out
    with torch.inference_mode():
        metrics=b.m.evaluate(b.loaded,b.replacement,b.readout,contract,settings,write_outputs=True)
    report={'mode':'compact_software_simulation','metrics':metrics,'test_count':len(images),
        'checkpoint_sha256':CHECKPOINT_SHA,'native_transformer_loaded':False,'device':str(b.device),
        'torch':torch.__version__,'elapsed_seconds':time.perf_counter()-start,'reference_historical_r1':1934/2400,
        'no_accuracy_guarantee_for_hardware':True,
        'phase_array_max_error_to_packaged':phase_errors}
    from memory import memory_report
    report['compute_memory']=memory_report(b)
    write(out/'metrics.json',report); print(report,flush=True)

if __name__=='__main__': main()
