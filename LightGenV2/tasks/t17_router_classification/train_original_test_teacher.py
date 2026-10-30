"""Training-only teacher on TRAIN + augmented original TEST; explicitly development data."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision.models import resnet18
from .test_exposed_data import load_raw_splits,metrics_from_scores
from .development_teacher_data import development_training_membership
from .train import write
from .gpu_policy import require_non_a100_gpu
from .optical_training_methods import inverse_sqrt_class_weights
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def d4_images(x,code):
    x=torch.rot90(x,code%4,(-2,-1))
    return x.flip(-1) if code>=4 else x


def teacher_head(classes):
    return nn.Sequential(nn.LayerNorm(512),nn.Linear(512,256),nn.GELU(),nn.Dropout(.2),nn.Linear(256,classes))


@torch.no_grad()
def head_scores(head,features,batch=1024):
    return torch.cat([head(features[i:i+batch]).softmax(1).cpu() for i in range(0,len(features),batch)]).numpy()


def main():
    ap=argparse.ArgumentParser()
    for name in ('profile','data-root','initial-weights','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();gpu=require_non_a100_gpu();cfg=json.loads(a.profile.read_text());torch.set_num_threads(4)
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);a.out.mkdir(parents=True,exist_ok=False)
    try:
        data,identity=load_raw_splits(a.data_root,cfg)
        metadata=dict(role='ORIGINAL_TEST_EXPOSED_TEACHER_TRAINING',independent_test_claim=False,
                      original_test_in_teacher_training=True,teacher_training_parts=['train','test'],teacher_validation_part='val',
                      backbone='ImageNet ResNet18, frozen; offline existing weights',backbone_path=str(a.initial_weights),
                      backbone_sha256=sha256(a.initial_weights),torchvision=torchvision.__version__,torch=torch.__version__,
                      teacher_head='LayerNorm512/Linear512-256/GELU/Dropout0.2/Linear256-C',
                      input_side=128,augmentation='all 8 D4 orientations, fixed RGB order; one orientation sampled per image each head epoch',
                      feature_cache_dtype='float16',profile=cfg,profile_sha256=sha256(a.profile),data=identity,
                      gpu_uuid=gpu,pid=os.getpid(),source_commit=os.environ.get('T17_SOURCE_COMMIT'),command=os.sys.argv,
                      git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                      source_sha256={str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('test_exposed_data.py'),
                          Path(__file__).with_name('development_teacher_data.py')]})
        write(a.out/'metadata.json',metadata)
        backbone=resnet18(weights=None);backbone.load_state_dict(torch.load(a.initial_weights,map_location='cpu',weights_only=True))
        backbone.fc=nn.Identity();backbone=backbone.cuda().eval()
        for p in backbone.parameters():p.requires_grad_(False)
        mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
        features={};started=time.time()
        with torch.inference_mode():
            for split,part in data.items():
                array=np.empty((len(part['labels']),8,512),dtype=np.float16)
                for code in range(8):
                    for start in range(0,len(part['labels']),cfg['teacher_feature_batch']):
                        images=torch.as_tensor(part['images'][start:start+cfg['teacher_feature_batch']].copy(),device='cuda').permute(0,3,1,2).float()/255
                        images=F.interpolate(images,(128,128),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
                        images=(d4_images(images,code)-mean)/std
                        with torch.autocast('cuda',dtype=torch.float16):values=backbone(images)
                        array[start:start+len(images),code]=values.cpu().numpy()
                    print(json.dumps({'state':'features','part':split,'D4_code':code}),flush=True)
                np.save(a.out/f'features_{split}.npy',array)
                np.savez_compressed(a.out/f'identities_{split}.npz',ids=part['ids'],labels=part['labels'],pixel_sha256=part['hashes'])
                features[split]=torch.from_numpy(array).cuda().float();del part['images']
        del backbone;torch.cuda.empty_cache()
        head=teacher_head(cfg['num_classes']).cuda()
        membership,removed=development_training_membership(data)
        indices=torch.as_tensor(membership,device='cuda')
        train_features=torch.cat((features['train'],features['test']))[indices]
        labels=torch.tensor(np.concatenate((data['train']['labels'],data['test']['labels'])),device='cuda',dtype=torch.long)[indices]
        write(a.out/'training_membership.json',dict(union_parts=['train','original_test'],kept_union_indices=membership.tolist(),removed=removed,
                                                   reason='Pixel deduplication only; original development evaluation set unchanged'))
        weights=inverse_sqrt_class_weights(labels,cfg['num_classes'])
        optimizer=torch.optim.AdamW(head.parameters(),lr=cfg['teacher_head_lr'],weight_decay=1e-4)
        rng=np.random.default_rng(cfg['seed']);best=(-1.,-1.,float('-inf'));history=[];last_improvement=0;completed=0
        for epoch in range(1,cfg['teacher_epochs']+1):
            head.train();order=rng.permutation(len(labels));loss_sum=0.
            lr=cfg['teacher_head_lr']*(.01+.99*.5*(1+math.cos(math.pi*(epoch-1)/max(1,cfg['teacher_epochs']-1))))
            for group in optimizer.param_groups:group['lr']=lr
            for start in range(0,len(order),cfg['teacher_head_batch']):
                idx=torch.as_tensor(order[start:start+cfg['teacher_head_batch']],device='cuda');code=torch.randint(8,(len(idx),),device='cuda')
                logits=head(train_features[idx,code]);loss=F.cross_entropy(logits,labels[idx],weight=weights,label_smoothing=.01)
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5,error_if_nonfinite=True);optimizer.step();loss_sum+=float(loss.detach())*len(idx)
            head.eval();val=metrics_from_scores(data['val']['labels'],head_scores(head,features['val'][:,0]));selection=(val['accuracy'],val['balanced_accuracy'],-val['cross_entropy'])
            row=dict(epoch=epoch,training_objective=loss_sum/len(labels),validation=val,lr=lr)
            if epoch==1 or epoch%5==0:
                row['original_test_IN_TEACHER_TRAINING']=metrics_from_scores(data['test']['labels'],head_scores(head,features['test'][:,0]))
            ck=dict(head={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},epoch=epoch,metadata=metadata,validation=val)
            torch.save(ck,a.out/'last_checkpoint.pt')
            if selection>best:best=selection;last_improvement=epoch;torch.save(ck,a.out/'best_checkpoint.pt')
            history.append(row);write(a.out/'history.json',history);write(a.out/'status.json',dict(state='teacher_head_training',epoch=epoch));completed=epoch
            print(json.dumps(row),flush=True)
            if epoch>=20 and epoch-last_improvement>=10:break
        best_ck=torch.load(a.out/'best_checkpoint.pt',map_location='cpu',weights_only=False);head.load_state_dict(best_ck['head']);head.eval()
        result=dict(role=metadata['role'],independent_test_claim=False,selected_epoch=best_ck['epoch'],completed_epochs=completed,checkpoint_sha256=sha256(a.out/'best_checkpoint.pt'),
                    original_test_in_teacher_training=True,training_original_image_count=len(labels),cached_views_per_original_image=8,
                    elapsed_seconds=time.time()-started,parts={})
        with torch.no_grad():
            for split,feature in features.items():
                flat=feature.reshape(-1,512);logits=torch.cat([head(flat[i:i+1024]).cpu() for i in range(0,len(flat),1024)]).reshape(len(feature),8,cfg['num_classes']).numpy()
                np.savez_compressed(a.out/f'teacher_logits_{split}.npz',ids=data[split]['ids'],labels=data[split]['labels'],logits=logits)
                result['parts'][split]=metrics_from_scores(data[split]['labels'],torch.from_numpy(logits[:,0]).softmax(1).numpy())
        write(a.out/'result.json',result);write(a.out/'status.json',dict(state='complete',**result));print(json.dumps(result),flush=True)
        write(a.out/'cache_manifest.json',{p.name:sha256(p) for p in a.out.iterdir() if p.suffix in ('.npy','.npz','.pt')})
    except BaseException as error:
        write(a.out/'status.json',dict(state='failed_or_interrupted',error=repr(error)));raise
    finally:
        torch.cuda.empty_cache();write(a.out/'exit_receipt.json',dict(pid=os.getpid(),timestamp=time.time()))


if __name__=='__main__':main()
