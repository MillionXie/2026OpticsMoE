"""Bounded, explicitly TEST-development comparison of six preserved full49 last models."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from .expert_tail_schedule import save, sha


def run(profile,out):
    import numpy as np
    import torch
    from .model import ScalingOptics
    from .train import evaluate
    c=json.loads(profile.read_text());assert c['seed']==17 and len(c['candidates'])==6 and c['target_correct']==616
    commit=os.environ['T10_CHECKPOINT_AUDIT_SOURCE_COMMIT'];repo=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip())
    files=[Path(__file__),profile,*[Path(__file__).with_name(n) for n in ['model.py','train.py','plan.py']],Path(__file__).parent/'configs/study.json',repo/'LightGenV2/common/optical_contract.py'];hashes={}
    for p in files:
        name=p.resolve().relative_to(repo).as_posix();assert p.read_bytes()==subprocess.check_output(['git','show',commit+':'+name]);hashes[name]=sha(p)
    assert os.environ['CUDA_VISIBLE_DEVICES']==c['gpu_uuid']==subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    torch.set_num_threads(4);reference=Path(c['reference_run']);m=json.loads((reference/'metadata.json').read_text());cache=Path(m['arguments']['data']);manifest=json.loads((cache.parent/'data_manifest.json').read_text());assert sha(cache)==m['data_sha256']==manifest['cache_sha256']
    with np.load(cache,allow_pickle=False) as z:x=z['test_images'];y=z['test_labels'];ids=z['test_ids'].astype(str)
    assert ids.tolist()==manifest['split_ids']['test'] and len(ids)==752 and set(ids).isdisjoint(manifest['split_ids']['train']) and set(ids).isdisjoint(manifest['split_ids']['val'])
    old=json.loads((reference/'test_result.json').read_text());assert sha(reference/'test_predictions.npz')==old['predictions_sha256']
    assert sha(reference/'best_checkpoint.pt')==old['checkpoint_sha256']
    with np.load(reference/'test_predictions.npz',allow_pickle=False) as z:
        assert np.array_equal(ids,z['ids']) and np.array_equal(y,z['labels']);old_pred=z['predictions'].copy()
    out.mkdir(parents=True,exist_ok=False)
    try:
        for row in c['candidates']:
            directory=Path(row['source_run']);a=json.loads((directory/'metadata.json').read_text());args=a['arguments']
            assert sha(directory/'metadata.json')==row['metadata_sha256'] and sha(Path(row['checkpoint']))==row['sha256']
            assert args['seed']==17 and args['experts']==args['top_k']==49 and args['layers']==4 and a['data_sha256']==m['data_sha256']
        save(out/'all_checkpoint_lock.json',dict(configuration=c,source_commit=commit,source_files=hashes,host_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),timestamp=time.time(),pid=os.getpid(),role='User-authorized TEST development checkpoint selection, not independent evaluation'))
        split=(torch.from_numpy(x.transpose(0,3,1,2).copy()),torch.from_numpy(y.astype(np.int64)),ids);rows=[]
        for source in c['candidates']:
            save(out/'status.json',dict(state='testing',pid=os.getpid(),current=source['name'],completed=rows))
            directory=Path(source['source_run']);metadata=json.loads((directory/'metadata.json').read_text());a=metadata['arguments'];protocol=copy.deepcopy(metadata['protocol']);protocol['training_proposal']['phase_noise_std']=0.
            checkpoint=torch.load(source['checkpoint'],map_location='cpu',weights_only=False)
            assert checkpoint.get('power_fraction',1.)==1. and checkpoint.get('routing_temperature',.05)==.05
            model=ScalingOptics(protocol,49,49,'moe_oeo',8,a['layers'],a['padding']).cuda();assert sum(p.numel() for p in model.parameters())==11078432;model.load_state_dict(checkpoint['model']);counter=checkpoint['epoch'];del checkpoint
            metrics,scores=evaluate(model,split,4);pred=scores.argmax(1).numpy();correct=int((pred==y).sum());assert np.isfinite(scores.numpy()).all()
            d=out/source['name'];d.mkdir();np.savez_compressed(d/'test_predictions.npz',ids=ids,labels=y,scores=scores.numpy(),predictions=pred)
            row=dict(name=source['name'],source_run=source['source_run'],checkpoint=source['checkpoint'],checkpoint_sha256=source['sha256'],source_counter=counter,
                test=metrics,test_correct=correct,predictions_sha256=sha(d/'test_predictions.npz'),source_commit=commit,data_sha256=m['data_sha256'],power_fraction=1.,
                role=c['role'],paired=dict(old_correct=int((old_pred==y).sum()),corrected=int(((old_pred!=y)&(pred==y)).sum()),new_wrong=int(((old_pred==y)&(pred!=y)).sum())))
            save(d/'test_result.json',row);rows.append(row);print(json.dumps(dict(name=source['name'],correct=correct,accuracy=correct/752,paired=row['paired'])),flush=True)
            del model,scores;torch.cuda.empty_cache()
        fallback=dict(name='original120_VAL_selected_reference',source_run=str(reference),checkpoint=str(reference/'best_checkpoint.pt'),
            checkpoint_sha256=old['checkpoint_sha256'],test=old['test'],test_correct=int((old_pred==y).sum()),reused_original_TEST=True)
        selected=max(rows+[fallback],key=lambda r:(r['test_correct'],-r['test']['macro_nll']))
        save(out/'summary.json',dict(rows=rows,selected=selected,pilot_target_met=selected['test_correct']>=616,target_correct=616,
            original_reference_correct=600,original_reference_remains_eligible=True,overall_best_correct=max(600,selected['test_correct']),three_seed_mean_target_established=False,role=c['role']))
        save(out/'status.json',dict(state='complete',pid=os.getpid(),best_correct=max(600,selected['test_correct']),pilot_target_met=selected['test_correct']>=616))
    except BaseException as error:
        save(out/'status.json',dict(state='failed',pid=os.getpid(),error=repr(error)));raise
    finally:save(out/'release_check.json',dict(pid=os.getpid(),inference_finished=True,gpu_uuid=c['gpu_uuid'],timestamp=time.time(),note='Confirm process exit and actual GPU allocation externally.'))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--profile',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();run(a.profile,a.out)
