"""Explicit post-hoc evaluation of two existing last checkpoints; no optimization."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from .model_linear_readout import LinearReadoutLG
from .model_four import encode_rgb
from .gpu_policy import require_authorized_gpu
from .figure_revision_data import load_panel_f
from .readout_data import load_raw
from .train_convex_readout import verified_cache
from .train_linear_readout import measure
from .train_optical_four_ccd import route_statistics
from .train import write
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def assert_matching_optical_body(state,reference):
    if set(state)!=set(reference):raise ValueError('Checkpoint key set mismatch')
    for name,value in reference.items():
        if name!='shared_head.weight' and not torch.equal(state[name],value):raise ValueError('Different optical body: '+name)


def main():
    p=argparse.ArgumentParser();p.add_argument('--profile',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();cfg=json.loads(args.profile.read_text());gpu=require_authorized_gpu(cfg['authorized_gpu_uuids']);torch.set_num_threads(4);task=Path(__file__).resolve().parent;args.out.mkdir(exist_ok=False,parents=True)
    write(args.out/'launch_context.json',dict(source_commit=os.environ.get('T17_SOURCE_COMMIT'),gpu_uuid=gpu,pid=os.getpid(),command=sys.argv,profile_sha256=sha256(args.profile)))
    try:
        for name,job in cfg['jobs'].items():
            out=args.out/name;out.mkdir(exist_ok=False);source=task/job['evaluation_checkpoint'];parent_path=task/job['parent_checkpoint']
            if sha256(source)!=job['evaluation_checkpoint_sha256'] or sha256(parent_path)!=job['parent_checkpoint_sha256']:raise ValueError('Checkpoint SHA changed')
            ck=torch.load(source,map_location='cpu',weights_only=False);parent=torch.load(parent_path,map_location='cpu',weights_only=False);assert_matching_optical_body(ck['model'],parent['model']);model=LinearReadoutLG(job['main_layers'],job['num_classes']).cuda();model.load_state_dict(ck['model'],strict=True);model.eval().requires_grad_(False)
            paths=[task/n for n in ['train_convex_readout.py','model_linear_readout.py','model_four.py','model_depth_ccd.py','train_linear_readout.py','figure_revision_data.py','readout_data.py']]+[task.parents[1]/'demo_check/pure_optical/models.py',task.parents[1]/'demo_check/adrenal_softsign_code_export_20260915_145336/code/optical_reference/optics.py']
            meta=dict(source_commit=os.environ.get('T17_SOURCE_COMMIT'),source_sha256={p.name:sha256(p) for p in paths+[Path(__file__)]},role='POSTHOC_ORIGINAL_TEST_EXPOSED_DEVELOPMENT',profile=job,checkpoint_reference=job['evaluation_checkpoint'],checkpoint_sha256=sha256(source),additional_training_steps=0,pid=os.getpid())
            write(out/'metadata.json',meta);write(out/'selection_lock.json',dict(checkpoint_reference=job['evaluation_checkpoint'],checkpoint_sha256=sha256(source),policy='Evaluate pre-existing last checkpoint as explicit post-hoc development candidate; never call this VAL-selected',TEST_evaluations_this_run_before_lock=0));write(out/'status.json',dict(state='evaluating_locked_existing_last',pid=os.getpid()))
            raw,_=load_panel_f(job['data']) if job['dataset']=='eurosat_original_f' else load_raw(Path(job['data_root']),job)
            result=dict(role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT',independent_test_claim=False,posthoc_last_checkpoint_candidate=True,checkpoint_reference=job['evaluation_checkpoint'],checkpoint_sha256=sha256(source),additional_training_steps=0,original_checkpoint_iteration=ck['epoch'])
            for split,label in [('train','train'),('val','val'),('test','original_test_dev')]:
                arrays=verified_cache(model,raw[split],out/f'{split}_features.npz',64,job,meta,split);x=torch.from_numpy(arrays['ccd_features']).cuda();y=torch.tensor(arrays['labels'],device='cuda',dtype=torch.long);metrics,scores=measure(model.shared_head,x,y);metrics['router']=route_statistics(arrays['route_power'])
                if 'domains' in arrays:
                    pred=scores.argmax(1);metrics['domains']={title:dict(n=int((arrays['domains']==d).sum()),correct=int((pred[arrays['domains']==d]==arrays['labels'][arrays['domains']==d]).sum()),accuracy=float((pred[arrays['domains']==d]==arrays['labels'][arrays['domains']==d]).mean())) for d,title in [(0,'RGB'),(1,'SAR')]}
                result[label]=metrics;np.savez_compressed(out/f'{split}_predictions.npz',ids=arrays['ids'],labels=arrays['labels'],scores=scores,predictions=scores.argmax(1),route_power=arrays['route_power'],dense_route_power=arrays['dense_route_power'],router_capture=arrays['router_capture'],**({'domains':arrays['domains']} if 'domains' in arrays else {}))
            with torch.no_grad():
                replay=[]
                for start in range(0,len(raw['test']['labels']),64):replay.append(model(encode_rgb(torch.from_numpy(raw['test']['images'][start:start+64]).cuda()))['probabilities'].cpu().numpy())
            replay=np.concatenate(replay);cached=np.load(out/'test_predictions.npz')['scores'];assert np.array_equal(replay.argmax(1),cached.argmax(1)) and np.allclose(replay,cached,atol=3e-6,rtol=1e-4);result['canonical_full_TEST_replay']=dict(predictions_identical=True,max_probability_difference=float(np.max(np.abs(replay-cached))))
            write(out/'result.json',result);write(out/'status.json',dict(state='complete',pid=os.getpid()));print(json.dumps(dict(job=name,**result)),flush=True);del model,parent,ck,raw,arrays,x,y;torch.cuda.empty_cache()
        write(args.out/'status.json',dict(state='complete',pid=os.getpid()))
    except BaseException as error:write(args.out/'status.json',dict(state='failed_or_interrupted',pid=os.getpid(),error=repr(error)));raise
    finally:torch.cuda.empty_cache();write(args.out/'exit_receipt.json',dict(pid=os.getpid(),timestamp=time.time()))


if __name__=='__main__':main()
