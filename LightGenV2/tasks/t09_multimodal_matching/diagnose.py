"""Frozen-weight modality controls, kept separate from model selection."""
import argparse
import json
from pathlib import Path
import torch

from .model import TextEncoder, OpticalOEO, encode
from .run import load_data, evaluate, state_sha
from .prepare import save, digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    vocab=json.loads((a.data/'vocab.json').read_text());results={}
    for run in a.runs:
        meta=json.loads((run/'metadata.json').read_text());cfg=meta['config'];mode=cfg['mode']
        data=load_data(a.data,'val',vocab,'cuda')
        if cfg.get('vision_checkpoint'):
            from .vision import frozen_features
            data['images']=frozen_features(Path(cfg['vision_checkpoint']),data['images'])
        frontend=TextEncoder(len(vocab),mode).cuda()
        frontend.load_state_dict(torch.load(run/mode/'frontend.pt',weights_only=False)['state'])
        frontend.requires_grad_(False).eval()
        torch.manual_seed(cfg['seed']);initial=TextEncoder(len(vocab),mode).cuda()
        frontend_changed=state_sha(initial)!=state_sha(frontend)
        for arch in ['moe','d2nn']:
            model=OpticalOEO(arch,cfg['seed']).cuda()
            checkpoint=torch.load(run/mode/arch/'best_checkpoint.pt',weights_only=False)
            model.load_state_dict(checkpoint['model']);model.requires_grad_(False).eval()
            baseline,_=evaluate(model,frontend,data,32)
            controls={}
            for condition in ['constant_question','constant_image','unpaired_images']:
                modified=dict(data)
                if condition=='constant_question':modified['ids']=data['ids'][:1].expand_as(data['ids'])
                elif condition=='constant_image':modified['images']=data['images'].float().mean(0,keepdim=True).expand_as(data['images'])
                else:modified['images']=data['images'].roll(37,0)
                controls[condition]=evaluate(model,frontend,modified,32)[0]
            assert abs(controls['constant_question']['accuracy']-.5)<1e-6
            # Physical input support is not the same as aperture envelope coverage.
            images=data['images'][data['index'][:32]]
            amplitude=encode(images,frontend(data['ids'][:32]))
            controls['input_nonzero_fraction']=float((amplitude>0).float().mean())
            controls['text_nonzero_fraction']=float((amplitude[:,112:,112:]>0).float().mean())
            results[mode+'/'+arch]=dict(baseline=baseline,controls=controls,frontend_changed_during_warmup=frontend_changed)
    save(a.out/'diagnostics.json',dict(results=results,test_accessed=False,
         note='Constant/unpaired inputs are corruption diagnostics on original labels, not new benchmark tasks. Original swap_pair_text evaluation is a permutation of existing pairs and NOT independent evidence.'))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
