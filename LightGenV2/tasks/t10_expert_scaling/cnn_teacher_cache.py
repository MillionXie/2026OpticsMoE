"""Finite cached ImageNet ResNet18 TRAIN teacher, VAL admission, TRAIN D4 targets.

This is additional pretrained electronic teacher supervision. It is never used in
the optical student's validation/test/inference and never reads TEST images.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import time

from .expert_tail_schedule import save,sha
from .ensemble_teacher_cache import d4
from . import reheat_refine as base


def admitted(teacher_only,per_class,val_correct,source_val_correct,val_per_class=None,floors=None):
    return (teacher_only>=128 and sum(v>=4 for v in per_class)>=4 and val_correct-source_val_correct>=38
            and all(val_per_class[int(k)]>=v for k,v in (floors or {}).items()))


def codes_for_epoch(size,epoch):
    import numpy as np
    return (np.arange(size,dtype=np.int64)+epoch-1)%8


def inputs(images,indices,codes):
    import numpy as np
    import torch
    import torch.nn.functional as F
    b=torch.from_numpy(images[indices].transpose(0,3,1,2).copy()).float().cuda()/255
    for code in range(8):
        selected=np.flatnonzero(codes==code)
        if len(selected):
            ix=torch.from_numpy(selected).to(b.device);b[ix]=d4(b[ix],code)
    # Whole patch is kept. This deliberately differs from the ImageNet eval
    # resize256/center-crop224 recipe; normalization uses its documented constants.
    b=F.interpolate(b,size=(224,224),mode='bilinear',align_corners=False,antialias=True)
    return (b-b.new_tensor([.485,.456,.406])[None,:,None,None])/b.new_tensor([.229,.224,.225])[None,:,None,None]


def run(c,out,profile):
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchvision
    from torchvision.models import resnet18
    assert c['teacher_epochs']==8 and c['teacher_batch']==32 and c['seed']==17
    assert os.environ['CUDA_VISIBLE_DEVICES']==c['gpus']['1']
    commit,hashes=base.sources(profile,c);weights=Path(c['pretrained_weights']);assert sha(weights)==c['pretrained_sha256']
    data_path=Path(c['teacher_data']);assert sha(data_path)==c['data_sha256']
    with np.load(data_path,allow_pickle=False) as z:
        x=z['train_images'];y=z['train_labels'].astype(np.int64);ids=z['train_ids'].astype(str)
        vx=z['val_images'];vy=z['val_labels'].astype(np.int64);vids=z['val_ids'].astype(str)
    assert len(y)==3496 and len(vy)==752 and set(ids).isdisjoint(vids)
    parent_root=Path(c['parent_cache']);receipt=json.loads((parent_root/'receipt.json').read_text())
    assert receipt['source_checkpoint_sha256']==c['incumbent_sha256'] and sha(parent_root/'parent_scores.npz')==receipt['parent_scores_sha256']
    with np.load(parent_root/'parent_scores.npz',allow_pickle=False) as z:
        assert np.array_equal(z['ids'],ids) and np.array_equal(z['labels'],y) and z['view_codes'].tolist()==list(range(8))
        parent=z['scores']
    out.mkdir(parents=True,exist_ok=False);torch.manual_seed(17);torch.cuda.manual_seed_all(17);torch.set_num_threads(4)
    model=resnet18(weights=None);model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));model.fc=torch.nn.Linear(model.fc.in_features,8);model=model.cuda()
    backbone=[p for name,p in model.named_parameters() if not name.startswith('fc.')]
    opt=torch.optim.AdamW([{'params':backbone,'lr':0.,'initial_lr':1e-4},{'params':model.fc.parameters(),'lr':1e-3,'initial_lr':5e-4}],weight_decay=1e-4)
    save(out/'metadata.json',dict(pid=os.getpid(),source_commit=commit,source_files=hashes,configuration=c,pretrained_sha256=sha(weights),
        pretraining_dataset='ImageNet1K; additional external pretraining, not original optical equal-budget experiment',torch_version=torch.__version__,torchvision_version=torchvision.__version__,
        student_inference_unchanged=True,teacher_only_electronic_fc=True,test_images_read=False,teacher_parameters=sum(p.numel() for p in model.parameters())))
    def predict(images,labels,image_ids,code=0):
        model.eval();scores=np.zeros((len(labels),8),np.float32);logits=np.zeros_like(scores)
        with torch.no_grad():
            for start in range(0,len(labels),64):
                ix=np.arange(start,min(start+64,len(labels)));value=model(inputs(images,ix,np.full(len(ix),code)));logits[ix]=value.cpu().numpy();scores[ix]=value.softmax(1).cpu().numpy()
        assert np.isfinite(scores).all() and np.max(abs(scores.sum(1)-1))<3e-6
        correct=int((scores.argmax(1)==labels).sum());nll=float(-np.log(scores[np.arange(len(labels)),labels].clip(1e-12)).mean())
        return scores,dict(correct=correct,total=len(labels),accuracy=correct/len(labels),macro_nll=nll,correct_per_class=np.bincount(labels[scores.argmax(1)==labels],minlength=8).tolist()),logits
    best_key=(-1,float('-inf'));history=[];started=time.time()
    for epoch in range(1,9):
        for p in backbone:p.requires_grad_(epoch>1)
        model.train()
        if epoch==1:
            for module in model.modules():
                if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
        order=np.random.default_rng(17+epoch).permutation(len(y));codes=codes_for_epoch(len(y),epoch);total=0.;t0=time.time()
        with (out/'sampling_history.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch,indices=order.tolist(),ids=ids[order].tolist(),codes=codes[order].tolist()))+'\n')
        scale=1. if epoch==1 else .1+.9*(1+np.cos(np.pi*(epoch-2)/6))/2
        for j,group in enumerate(opt.param_groups):group['lr']=(0. if j==0 else .001) if epoch==1 else group['initial_lr']*scale
        for start in range(0,len(y),32):
            ix=order[start:start+32];opt.zero_grad(set_to_none=True);logits=model(inputs(x,ix,codes[ix]));loss=F.cross_entropy(logits,torch.from_numpy(y[ix]).cuda())
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite CNN teacher loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(norm);opt.step();total+=float(loss.detach())*len(ix)
            if start%256==0:save(out/'status.json',dict(state='training_teacher',pid=os.getpid(),epoch=epoch,completed=start+len(ix),total=len(y),timestamp=time.time()))
        training_seconds=time.time()-t0;vs,vm,_=predict(vx,vy,vids);row=dict(epoch=epoch,train_loss=total/len(y),val=vm,training_seconds=training_seconds,gradient_norm=float(norm),lr_scale=float(scale),learning_rates=[g['lr'] for g in opt.param_groups]);history.append(row)
        payload=dict(model=model.state_dict(),epoch=epoch,optimizer=opt.state_dict(),source_commit=commit,pretrained_sha256=sha(weights))
        torch.save(payload,out/'last_checkpoint.pt');key=(vm['correct'],-vm['macro_nll'])
        if key>best_key:
            best_key=key;torch.save(dict(model=model.state_dict(),epoch=epoch,source_commit=commit,pretrained_sha256=sha(weights)),out/'best_checkpoint.pt')
            np.savez_compressed(out/'selected_val_predictions.npz',ids=vids,labels=vy,scores=vs)
        save(out/'history.json',history)
    p=torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(p['model']);best_epoch=p['epoch'];del p
    clean,tm,clean_logits=predict(x,y,ids);np.savez_compressed(out/'clean_train_scores.npz',ids=ids,labels=y,scores=clean,logits=clean_logits)
    selected_val=np.load(out/'selected_val_predictions.npz');vcorrect=selected_val['scores'].argmax(1)==vy;vperclass=np.bincount(vy[vcorrect],minlength=8).tolist()
    t=clean.argmax(1)==y;p=parent[0].argmax(1)==y;complement=t&~p;perclass=np.bincount(y[complement],minlength=8).tolist();ok=admitted(int(complement.sum()),perclass,best_key[0],c['source_val_correct'],vperclass,c['teacher_val_class_floors'])
    admission=dict(admitted=ok,train=tm,teacher_only=int(complement.sum()),parent_only=int((p&~t).sum()),common_wrong=int((~p&~t).sum()),complement_per_class=perclass,
        selected_teacher_epoch=best_epoch,selected_val_correct=best_key[0],source_val_correct=c['source_val_correct'],teacher_checkpoint_sha256=sha(out/'best_checkpoint.pt'),
        selected_val_correct_per_class=vperclass,thresholds=dict(teacher_only=128,minimum_classes4_each4=True,val_correct_gain_at_least=38,difficult_class_floors=c['teacher_val_class_floors']),no_teacher_test_forward=True)
    save(out/'admission.json',admission)
    if not ok:
        save(out/'status.json',dict(state='rejected_by_train_val_admission',pid=os.getpid(),admission=admission));return
    all_scores=np.zeros((8,3496,8),np.float32);all_logits=np.zeros_like(all_scores);all_scores[0]=clean;all_logits[0]=clean_logits
    for code in range(1,8):
        all_scores[code],_,all_logits[code]=predict(x,y,ids,code);save(out/'status.json',dict(state='caching_train_views',pid=os.getpid(),view=code,completed=3496,total=3496,timestamp=time.time()))
    np.savez_compressed(out/'targets.npz',ids=ids,labels=y,teacher_scores=all_scores,teacher_logits=all_logits,parent_scores=parent,view_codes=np.arange(8))
    configuration=dict(c);configuration['models']={'parent':{'sha256':c['incumbent_sha256']},'teacher':{'sha256':sha(out/'best_checkpoint.pt'),'architecture':'resnet18','pretrained_sha256':sha(weights)}}
    manifest=dict(source_commit=commit,source_files=hashes,configuration=configuration,targets_sha256=sha(out/'targets.npz'),data_sha256=c['data_sha256'],train_samples=3496,
        view_codes=list(range(8)),teacher_weights={'ImageNet_ResNet18':1.},average_across_teachers_not_views=True,parent_cache_receipt_sha256=sha(parent_root/'receipt.json'),
        teacher_supervised_training_image_inputs=8*3496,teacher_val_forward_inputs=8*752,teacher_train_cache_forward_inputs=8*3496,no_teacher_test_forward=True,
        extra_external_pretraining=True,student_inference_unchanged=True,elapsed_seconds=time.time()-started)
    save(out/'cache_manifest.json',manifest);save(out/'status.json',dict(state='complete',pid=os.getpid(),admitted=True,teacher_val_correct=best_key[0],teacher_only=int(complement.sum())))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();c=json.loads(a.profile.read_text())
    os.environ['T10_REHEAT_SOURCE_COMMIT']=os.environ['T10_CNN_SOURCE_COMMIT']
    try:run(c,a.out,a.profile)
    except BaseException as error:
        if a.out.exists():save(a.out/'status.json',dict(state='failed',pid=os.getpid(),error=repr(error)))
        raise
