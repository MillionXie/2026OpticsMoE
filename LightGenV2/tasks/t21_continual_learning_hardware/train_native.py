"""Bounded A->D native-pixel adaptation. No SDK, TEST gradients, or timers.

This is a new equal-memory protocol, NOT reproduction of T16 full-replay scores.
Only A is warm-started from T16; later stages inherit the selected native parent.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from .model import build_model
from .subset import memory_indices

TASKS=('eurosat','clevr','speech_binary','physical_binary_raw')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def put(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def load_data(folder,split):
    result={}
    for name in TASKS:
        p=Path(folder)/name;m=json.loads((p/'manifest.json').read_text())
        if m['task']!=name or m['split']!=split or sha(p/'inputs.npz')!=m['inputs_sha256']:
            raise ValueError('Input identity mismatch: '+str(p))
        with np.load(p/'inputs.npz',allow_pickle=False) as a:
            x=torch.from_numpy(a['amplitudes'].copy()).float()
            y=torch.from_numpy(a['labels'].copy()).long()
        if x.shape!=(1000,224,224) or len(y)!=1000 or not torch.isfinite(x).all():
            raise ValueError('Unexpected fixed-source shape')
        result[name]=dict(x=x,y=y,manifest=m)
    return result


@torch.no_grad()
def evaluate(model,data,names,device,batch):
    model.eval();result={}
    for name in names:
        d=data[name];pred=[];prob=[]
        for start in range(0,len(d['y']),batch):
            out=model(d['x'][start:start+batch].to(device))
            pred.extend(out['logits'].argmax(1).cpu().tolist())
            prob.extend(out['probabilities'].cpu().tolist())
        y=d['y'].numpy();pred=np.asarray(pred)
        classes=range(10 if name=='eurosat' else 2)
        recalls=[float((pred[y==k]==k).mean()) for k in classes]
        result[name]=dict(n=len(y),correct=int((y==pred).sum()),
            accuracy=float((y==pred).mean()),balanced_accuracy=float(np.mean(recalls)),
            per_class_recall=recalls,labels=y.tolist(),predictions=pred.tolist(),
            probabilities=prob,source_manifest=d['manifest']['identity_sha256'])
    return result


def summary(scores):
    return {k:{v:r[v] for v in ('n','correct','accuracy','balanced_accuracy','per_class_recall')}
            for k,r in scores.items()}


def score(scores):return float(np.mean([v['balanced_accuracy'] for v in scores.values()]))


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def frozen(model):
    return {k:tensor_sha(p) for k,p in model.named_parameters() if not p.requires_grad}


def gpu_gate(config,uuid):
    if uuid not in config['allowed_GPU_UUIDs']:raise ValueError('GPU not authorized')
    text=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.free',
                                  '--format=csv,noheader,nounits'],text=True)
    rows=[r.split(',') for r in text.splitlines()]
    row=next((r for r in rows if r[0].strip()==uuid),None)
    if row is None or 'A100' in row[1] or float(row[2])<10000:
        raise RuntimeError('Authorized GPU is absent/A100/has insufficient free memory')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=uuid:
        raise RuntimeError('Launcher must bind the complete authorized GPU UUID')
    torch.cuda.set_per_process_memory_fraction(config['per_process_GPU_memory_fraction'],0)
    if 'A100' in torch.cuda.get_device_name(0):raise RuntimeError('A100 forbidden')
    return dict(uuid=uuid,name=row[1].strip(),free_MiB_at_start=float(row[2]))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--architecture',choices=('moe','d2nn'),required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--train',type=Path,required=True)
    p.add_argument('--val',type=Path,required=True)
    p.add_argument('--test',type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();c=json.loads(a.config.read_text())
    a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(c['seed']);np.random.seed(c['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    began=time.time()
    try:
        gpu=gpu_gate(c,a.gpu_uuid);device=torch.device('cuda:0')
        put(a.out/'status.json',dict(state='loading_TRAIN_VAL',gpu=gpu,started=began))
        train=load_data(a.train,'train');val=load_data(a.val,'val')
        # TEST is deliberately not loaded until the phase checkpoint is selected.
        memories={n:memory_indices(train[n]['manifest'],c['memory_records_per_old_task']) for n in TASKS}
        put(a.out/'memory_manifest.json',{n:dict(indices=ix.tolist(),
            original_source_indices=[train[n]['manifest']['source_indices'][i] for i in ix],
            source_manifest=train[n]['manifest']['identity_sha256'],count=len(ix)) for n,ix in memories.items()})
        identity=json.loads((Path(__file__).parent/'source_identity.json').read_text())
        source=identity['models'][0 if a.architecture=='moe' else 4]
        if sha(source['checkpoint'])!=source['sha256']:raise ValueError('Original A PT SHA mismatch')
        original=torch.load(source['checkpoint'],map_location='cpu',weights_only=False)
        model=build_model(a.architecture).to(device)
        model.initialize_from_reference(original['model']);del original
        model.configure_stage(1)
        calibration=train[TASKS[0]]['x'][np.linspace(0,999,24,dtype=int)].to(device)
        gains=model.calibrate_gains(calibration);del calibration
        put(a.out/'gain_calibration.json',dict(scope='TRAIN only, fixed throughout A-D',values=gains))
        source_hashes={str(q.relative_to(Path(__file__).parents[3])):sha(q)
                       for q in Path(__file__).parent.glob('*.py')}
        put(a.out/'protocol.json',dict(config=c,config_sha256=sha(a.config),source_sha256=source_hashes,
            architecture=a.architecture,gpu=gpu,original_A_sha256=source['sha256'],
            memory_scope='same fixed stratified 300 original TRAIN records per old task; one cycle',
            current_scope='new fixed stratified TRAIN1000, not original full-current dataset',
            distillation='previous native parent on old TRAIN only' if a.architecture=='moe' else 'none',
            evaluation_scope='fixed stratified1000, VAL selection; TEST evaluated after selection only',
            hardware_admitted=False,phase_response='ideal quantized phase, linearVoltage LUT is not a measured phase calibration'))
        chain=[]
        for phase in range(1,5):
            name=chr(64+phase);out=a.out/name;out.mkdir()
            names=TASKS[:phase];teacher=copy.deepcopy(model).eval() if phase>1 and a.architecture=='moe' else None
            model.configure_stage(phase)
            immutable=frozen(model);gain_sha=tensor_sha(model.display_gains)
            selection={n:(np.arange(1000) if n==names[-1] else memories[n]) for n in names}
            cached_teacher={}
            if teacher is not None:
                for n in names[:-1]:
                    ix=selection[n];rows=[]
                    with torch.no_grad():
                        for start in range(0,len(ix),c['batch_size']):
                            rows.append(teacher(train[n]['x'][ix[start:start+c['batch_size']]].to(device))['logits'].cpu())
                    cached_teacher[n]=torch.cat(rows)
                del teacher
            params=[v for v in model.parameters() if v.requires_grad]
            optical=[v for k,v in model.named_parameters() if v.requires_grad and not k.startswith('shared_head.')]
            optimizer=torch.optim.AdamW([dict(params=optical,lr=c['optical_lr']),
                dict(params=model.shared_head.parameters(),lr=c['head_lr'])],weight_decay=c['weight_decay'])
            baseline=evaluate(model,val,names,device,c['batch_size']);best=score(baseline);best_epoch=0;wait=0
            history=[dict(epoch=0,validation=summary(baseline),selection_score=best)]
            def save(file,epoch):
                torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    architecture=a.architecture,stage=phase,epoch=epoch,best_epoch=best_epoch,
                    config_sha256=sha(a.config),selection='mean VAL balanced accuracy',
                    parent_checkpoint_sha256=chain[-1]['best_sha256'] if chain else source['sha256']),out/file)
            save('best_checkpoint.pt',0)
            for epoch in range(1,c['stage_epochs'][phase-1]+1):
                epoch_start=time.time();model.train();rng=np.random.default_rng(c['seed']+phase*1000+epoch)
                batches=[]
                for n in names:
                    perm=rng.permutation(len(selection[n]))
                    batches.extend((n,perm[s:s+c['batch_size']]) for s in range(0,len(perm),c['batch_size']))
                rng.shuffle(batches);loss_sum=0.;correct=0;seen=0
                for n,relative in batches:
                    ix=selection[n][relative];x=train[n]['x'][ix].to(device);y=train[n]['y'][ix].to(device)
                    optimizer.zero_grad(set_to_none=True);logits=model(x)['logits']
                    loss=F.cross_entropy(logits,y,label_smoothing=c['label_smoothing'])
                    if n in cached_teacher:
                        temp=c['distillation_temperature'];old=cached_teacher[n][relative].to(device)
                        kd=F.kl_div(F.log_softmax(logits/temp,1),F.softmax(old/temp,1),reduction='batchmean')*temp*temp
                        loss=loss+c['distillation_weight']*kd
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite TRAIN loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step()
                    loss_sum+=float(loss.detach())*len(y);seen+=len(y);correct+=int((logits.argmax(1)==y).sum())
                validation=evaluate(model,val,names,device,c['batch_size']);value=score(validation)
                if frozen(model)!=immutable or tensor_sha(model.display_gains)!=gain_sha:
                    raise RuntimeError('Old expert or fixed display gain changed')
                if value>best+1e-10:
                    best=value;best_epoch=epoch;wait=0;save('best_checkpoint.pt',epoch)
                else:wait+=1
                save('last_checkpoint.pt',epoch)
                row=dict(epoch=epoch,loss=loss_sum/seen,TRAIN_accuracy=correct/seen,TRAIN_records=seen,
                    validation=summary(validation),selection_score=value,best_epoch=best_epoch,
                    best_score=best,seconds=time.time()-epoch_start)
                history.append(row);put(out/'history.json',history)
                put(a.out/'status.json',dict(state='training',phase=name,**row,updated=time.time(),gpu=gpu))
                print(json.dumps(dict(phase=name,**row)),flush=True)
                if epoch>=c['minimum_epochs'] and wait>=c['patience']:break
            checkpoint=torch.load(out/'best_checkpoint.pt',map_location=device,weights_only=False)
            model.load_state_dict(checkpoint['model'],strict=True)
            if not (out/'last_checkpoint.pt').exists():raise RuntimeError('No trained last')
            test=load_data(a.test,'test')
            selected=evaluate(model,test,names,device,c['batch_size']);del test
            put(out/'selected_test.json',dict(scope='adapted simulation, not hardware',
                TEST_gradients=False,TEST_selection=False,selected_epoch=best_epoch,scores=selected))
            put(out/'frozen_audit.json',dict(pass_all=True,frozen_parameter_SHA256=immutable,
                fixed_gain_sha256=gain_sha,old_experts_bitwise_frozen=True))
            receipt=dict(phase=name,best_epoch=best_epoch,best_sha256=sha(out/'best_checkpoint.pt'),
                last_sha256=sha(out/'last_checkpoint.pt'),selected_test_sha256=sha(out/'selected_test.json'),
                test=summary(selected),best_is_untrained_parent=(best_epoch==0))
            put(out/'completion.json',receipt);chain.append(receipt);put(a.out/'chain.json',chain)
        put(a.out/'return.json',dict(returncode=0,completed=time.time(),seconds=time.time()-began,chain=chain))
        put(a.out/'status.json',dict(state='complete',updated=time.time(),gpu=gpu))
    except BaseException:
        put(a.out/'failure.json',dict(traceback=traceback.format_exc(),updated=time.time()))
        raise


if __name__=='__main__':main()
