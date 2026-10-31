"""Separate F teachers: own-domain TRAIN + augmented original TEST, never opposite domain."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from .figure_revision_data import load_panel_f
from .f_cross_domain import select_source_domain
from .train_original_test_teacher import teacher_head,head_scores,d4_images
from .test_exposed_data import metrics_from_scores
from .gpu_policy import require_authorized_gpu
from .train import write
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    cfg=json.loads(args.profile.read_text());gpu=require_authorized_gpu(cfg['authorized_gpu_uuids']);torch.set_num_threads(4);torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed'])
    args.out.mkdir(parents=True,exist_ok=False);started=time.time()
    try:
        raw,identity=load_panel_f(cfg['data']);raw=select_source_domain(raw,cfg['training_domain']);domain=0 if cfg['training_domain']=='RGB' else 1
        raw['test']={k:v[raw['test']['domains']==domain] for k,v in raw['test'].items()}
        assert all((part['domains']==domain).all() for part in raw.values())
        weights=Path(cfg['teacher_initial_weights']);assert sha256(weights)==cfg['teacher_initial_weights_sha256']
        task=Path(__file__).resolve().parent
        meta=dict(role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT',training_domain=cfg['training_domain'],original_data=cfg['data'],data=identity,
            teacher_training_parts=['source-domain TRAIN','source-domain original TEST, augmented'],opposite_domain_used=False,
            VAL_training=False,backbone='frozen ImageNet ResNet18',backbone_sha256=sha256(weights),
            source_commit=os.environ.get('T17_SOURCE_COMMIT'),gpu_uuid=gpu,pid=os.getpid(),command=os.sys.argv,
            source_sha256={p.name:sha256(p) for p in [Path(__file__),task/'f_cross_domain.py',task/'figure_revision_data.py',task/'train_original_test_teacher.py']})
        write(args.out/'metadata.json',meta);features={}
        backbone=resnet18(weights=None);backbone.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));backbone.fc=nn.Identity();backbone=backbone.cuda().requires_grad_(False).eval()
        mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
        with torch.inference_mode():
            for split,part in raw.items():
                values=np.empty((len(part['labels']),8,512),dtype=np.float16)
                for code in range(8):
                    for start in range(0,len(part['labels']),cfg['teacher_feature_batch']):
                        x=torch.as_tensor(part['images'][start:start+cfg['teacher_feature_batch']].copy(),device='cuda').permute(0,3,1,2).float()/255
                        x=F.interpolate(x,(128,128),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
                        with torch.autocast('cuda',dtype=torch.float16):z=backbone((d4_images(x,code)-mean)/std)
                        values[start:start+len(x),code]=z.cpu().numpy()
                    write(args.out/'status.json',dict(state='teacher_features',split=split,code=code,pid=os.getpid()))
                np.save(args.out/f'features_{split}.npy',values);features[split]=torch.from_numpy(values).cuda().float()
                np.savez_compressed(args.out/f'identities_{split}.npz',ids=part['ids'],labels=part['labels'],domains=part['domains']);del part['images']
        del backbone;torch.cuda.empty_cache();head=teacher_head(cfg['num_classes']).cuda();optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
        train_features=torch.cat([features['train'],features['test']]);labels=torch.as_tensor(np.concatenate([raw['train']['labels'],raw['test']['labels']]),device='cuda',dtype=torch.long)
        best=(-1.,float('-inf'));stale=0;history=[]
        for epoch in range(1,cfg['teacher_epochs']+1):
            head.train();order=rng.permutation(len(labels));total=0.
            for group in optimizer.param_groups:group['lr']=.001*(.05+.95*.5*(1+np.cos(np.pi*(epoch-1)/max(1,cfg['teacher_epochs']-1))))
            for start in range(0,len(order),256):
                idx=torch.as_tensor(order[start:start+256],device='cuda');code=torch.randint(8,(len(idx),),device='cuda')
                loss=F.cross_entropy(head(train_features[idx,code]),labels[idx],label_smoothing=.01);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);optimizer.step();total+=float(loss.detach())*len(idx)
            head.eval();val=metrics_from_scores(raw['val']['labels'],head_scores(head,features['val'][:,0]));key=(val['balanced_accuracy'],-val['cross_entropy']);improved=key>best;stale=0 if improved else stale+1
            ck=dict(head={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},epoch=epoch,validation=val,metadata=meta)
            torch.save(ck,args.out/'last_checkpoint.pt')
            if improved:best=key;torch.save(ck,args.out/'best_checkpoint.pt')
            history.append(dict(epoch=epoch,training_objective=total/len(labels),validation=val));write(args.out/'history.json',history);write(args.out/'status.json',dict(state='teacher_training',epoch=epoch,pid=os.getpid()))
            if epoch>=20 and stale>=12:break
        ck=torch.load(args.out/'best_checkpoint.pt',map_location='cpu',weights_only=False);head.load_state_dict(ck['head']);head.eval();result=dict(role=meta['role'],training_domain=cfg['training_domain'],selected_epoch=ck['epoch'],completed_epochs=epoch,checkpoint_sha256=sha256(args.out/'best_checkpoint.pt'),teacher_TRAIN_original_images=3000,teacher_original_TEST_exposed_images=1000,opposite_domain_used=False,parts={})
        with torch.no_grad():
            for split,z in features.items():
                flat=z.reshape(-1,512);logits=torch.cat([head(flat[start:start+1024]).cpu() for start in range(0,len(flat),1024)]).reshape(len(z),8,10).numpy()
                np.savez_compressed(args.out/f'teacher_logits_{split}.npz',ids=raw[split]['ids'],labels=raw[split]['labels'],domains=raw[split]['domains'],logits=logits)
                result['parts'][split]=metrics_from_scores(raw[split]['labels'],torch.from_numpy(logits[:,0]).softmax(1).numpy())
        result['seconds']=time.time()-started;write(args.out/'result.json',result);write(args.out/'cache_manifest.json',{p.name:sha256(p) for p in args.out.iterdir() if p.suffix in ('.npz','.npy','.pt')});write(args.out/'status.json',dict(state='complete',**result))
    except BaseException as error:
        write(args.out/'status.json',dict(state='failed',error=repr(error)));raise
    finally:torch.cuda.empty_cache();write(args.out/'exit_receipt.json',dict(pid=os.getpid(),time=time.time()))


if __name__=='__main__':main()
