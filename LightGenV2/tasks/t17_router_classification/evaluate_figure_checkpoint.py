"""Full canonical evaluation of explicitly sealed existing optical last checkpoints."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from .train_figure_revision import build_model,configure_parameters,assert_frozen,evaluate
from .figure_revision_data import load_panel_f
from .readout_data import load_raw
from .model_four import encode_rgb
from .gpu_policy import require_authorized_gpu
from .train import write
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--profile',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();cfg=json.loads(args.profile.read_text());gpu=require_authorized_gpu(cfg['authorized_gpu_uuids']);torch.set_num_threads(4);task=Path(__file__).resolve().parent;args.out.mkdir(exist_ok=False,parents=True)
    write(args.out/'launch_context.json',dict(source_commit=os.environ.get('T17_SOURCE_COMMIT'),pid=os.getpid(),gpu_uuid=gpu,command=sys.argv,profile_sha256=sha256(args.profile)))
    try:
        for name,job in cfg['jobs'].items():
            out=args.out/name;out.mkdir(exist_ok=False);path=task/job['evaluation_checkpoint'];parent_path=task/job['parent_checkpoint']
            if sha256(path)!=job['evaluation_checkpoint_sha256'] or sha256(parent_path)!=job['parent_checkpoint_sha256']:raise ValueError('Sealed checkpoint changed')
            ck=torch.load(path,map_location='cpu',weights_only=False);parent=torch.load(parent_path,map_location='cpu',weights_only=False);model=build_model(job).cuda();model.load_state_dict(ck['model'],strict=True);configure_parameters(model,job);frozen={n:parent['model'][n] for n,p in model.named_parameters() if not p.requires_grad};assert_frozen(model,frozen);model.eval().requires_grad_(False)
            meta=dict(role='POSTHOC_ORIGINAL_TEST_EXPOSED_DEVELOPMENT',source_commit=os.environ.get('T17_SOURCE_COMMIT'),entry_sha256=sha256(Path(__file__)),profile=job,pid=os.getpid(),checkpoint_reference=job['evaluation_checkpoint'],checkpoint_sha256=sha256(path),additional_training_steps=0,selection='Explicit post-hoc evaluation of saved last; not VAL selected')
            write(out/'metadata.json',meta);write(out/'selection_lock.json',dict(checkpoint_reference=job['evaluation_checkpoint'],checkpoint_sha256=sha256(path),TEST_evaluations_this_run_before_lock=0,posthoc_development=True));write(out/'status.json',dict(state='loading_verified_inputs',pid=os.getpid()))
            raw,identity=load_panel_f(job['data']) if job['dataset']=='eurosat_original_f' else load_raw(Path(job['data_root']),job);meta['data']=identity;write(out/'metadata.json',meta)
            result=dict(role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT',independent_test_claim=False,posthoc_last_checkpoint_candidate=True,checkpoint_reference=job['evaluation_checkpoint'],checkpoint_sha256=sha256(path),additional_training_steps=0,original_checkpoint_epoch=ck['epoch'],frozen_tensors_match_original_training_parent=True)
            for split,key in [('train','train'),('val','val'),('test','original_test_dev')]:
                fields=[]
                for start in range(0,len(raw[split]['labels']),128):fields.append(encode_rgb(torch.from_numpy(raw[split]['images'][start:start+128]).cuda()))
                x=torch.cat(fields);y=torch.tensor(raw[split]['labels'],device='cuda',dtype=torch.long);metrics,arrays=evaluate(model,x,y,64,raw[split].get('domains'));result[key]=metrics
                np.savez_compressed(out/f'{split}_predictions.npz',ids=raw[split]['ids'],**arrays);del x,y,fields
            write(out/'result.json',result);write(out/'status.json',dict(state='complete',pid=os.getpid()));print(json.dumps(dict(job=name,train=result['train']['accuracy'],val=result['val']['accuracy'],dev=result['original_test_dev']['accuracy'],correct=result['original_test_dev']['correct'])),flush=True);del model,parent,ck,raw;torch.cuda.empty_cache()
        write(args.out/'status.json',dict(state='complete',pid=os.getpid()))
    except BaseException as error:write(args.out/'status.json',dict(state='failed_or_interrupted',pid=os.getpid(),error=repr(error)));raise
    finally:torch.cuda.empty_cache();write(args.out/'exit_receipt.json',dict(pid=os.getpid(),timestamp=time.time()))


if __name__=='__main__':main()
