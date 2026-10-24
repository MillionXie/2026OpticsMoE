"""Bounded A->D native-pixel adaptation. No SDK, TEST gradients, or timers.

This is a new equal-memory protocol, NOT reproduction of T16 full-replay scores.
Only A is warm-started from T16; later stages inherit the selected native parent.
Optional original-stage TRAIN teachers and TEST-selected development are explicit
config switches, never inference changes or architecture-relative score targets.
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
from LightGenV2.tasks.t16_zero_phase_ccd_lifelong.train_lifelong_moe import load_dataset
from LightGenV2.tasks.t16_zero_phase_ccd_lifelong.train_other_tasks import clevr_pairwise_loss
from LightGenV2.tasks.t16_zero_phase_ccd_lifelong.model import DirectCCDOptics

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


def reference_TRAIN_objective(student, labels, teacher, config):
    """Preserve the original stage function on TRAIN, not a requested TEST score."""
    temperature=config['reference_temperature']
    ce=F.cross_entropy(student,labels,label_smoothing=config['label_smoothing'])
    kd=F.kl_div(F.log_softmax(student/temperature,1),
                F.softmax(teacher.detach()/temperature,1),reduction='batchmean')*temperature**2
    return config['reference_CE_weight']*ce+config['reference_KD_weight']*kd


def reference_teacher(identity,architecture,phase,device):
    source=identity['models'][(0 if architecture=='moe' else 4)+phase-1]
    path=Path(source['checkpoint']);cfgpath=path.with_name('config.json')
    if sha(path)!=source['sha256'] or sha(cfgpath)!=source['config_sha256']:
        raise ValueError('Original stage teacher identity changed')
    cfg=json.loads(cfgpath.read_text())
    keys=('router_side','router_pitch','output_side','x_pitch','y_pitch',
          'routing_temperature','activation_order')
    model=DirectCCDOptics(architecture,**{k:cfg[k] for k in keys if k in cfg}).to(device)
    model.load_state_dict(torch.load(path,map_location=device,weights_only=False)['model'],strict=True)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model,source


@torch.no_grad()
def reference_TRAIN_cache(teacher,train,current,selection,names,device,batch):
    result={}
    for name in names:
        if train[name]['manifest']['split']!='train':
            raise ValueError('Reference teacher may only consume TRAIN')
        rows=[];indices=selection[name]
        for start in range(0,len(indices),batch):
            ix=indices[start:start+batch]
            x=(current.get_batch(ix,device) if name==names[-1] and current is not None
               else train[name]['x'][ix].to(device))
            rows.append(teacher(x)['logits'].cpu())
        result[name]=torch.cat(rows)
    return result


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
    p.add_argument('--source-protocols',type=Path,help='Full original TRAIN for server adaptation only; hardware TRAIN remains fixed1000')
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
        selection_split=c.get('selection_split','val')
        if selection_split not in ('val','test'):raise ValueError('Unknown checkpoint selection split')
        selection_data=val if selection_split=='val' else load_data(a.test,'test')
        selection_label='mean '+('VAL' if selection_split=='val' else 'TEST-selected development')+' balanced accuracy'
        reference_mode=c.get('reference_stage_teacher',False)
        source_paths=({k:Path(v) for k,v in json.loads(a.source_protocols.read_text()).items()}
                      if a.source_protocols else None)
        # TRAIN teachers below never receive selection_data. Legacy VAL profiles
        # load TEST only after selection; explicit TEST-selection profiles do not.
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
            memory_scope=f"same fixed stratified 300 original TRAIN records per old task; {c['memory_cycles_per_epoch']} cycles per epoch",
            current_scope='original full TRAIN, original frozen front, encoded on demand' if source_paths else 'initial small-budget TRAIN1000 pilot',
            distillation=('original corresponding stage teacher on TRAIN only, both architectures' if reference_mode else
                          'previous native parent on old TRAIN only' if a.architecture=='moe' else 'none'),
            evaluation_scope='fixed stratified1000; '+selection_label,
            TEST_selection=selection_split=='test',TEST_gradients=False,
            independent_generalization_claim=False,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            hardware_admitted=False,phase_response='ideal quantized phase, linearVoltage LUT is not a measured phase calibration'))
        chain=[]
        for phase in range(1,5):
            name=chr(64+phase);out=a.out/name;out.mkdir()
            names=TASKS[:phase];teacher=copy.deepcopy(model).eval() if phase>1 and a.architecture=='moe' and not reference_mode else None
            model.configure_stage(phase)
            immutable=frozen(model);gain_sha=tensor_sha(model.display_gains)
            current=None
            if source_paths:
                put(a.out/'status.json',dict(state='loading_full_current_TRAIN',phase=name,updated=time.time(),gpu=gpu))
                expected=identity['models'][3]['config']['protocol_sha256'][names[-1]]
                if sha(source_paths[names[-1]])!=expected or sha(source_paths['vision'])!=train[names[-1]]['manifest']['vision_checkpoint_sha256']:
                    raise ValueError('Full TRAIN protocol or frozen front changed')
                current=load_dataset(source_paths,names[-1],'train',vision_checkpoint=source_paths['vision'],device=device)
                expected_count=(15998,140000,12526,140800)[phase-1]
                if len(current)!=expected_count:raise ValueError('Full original TRAIN count changed')
            selection={n:(np.arange(len(current) if current is not None else 1000) if n==names[-1] else memories[n]) for n in names}
            cached_teacher={}
            if reference_mode:
                put(a.out/'status.json',dict(state='original_stage_TRAIN_teacher',phase=name,updated=time.time(),gpu=gpu))
                teacher,teacher_identity=reference_teacher(identity,a.architecture,phase,device)
                cached_teacher=reference_TRAIN_cache(teacher,train,current,selection,names,device,c['batch_size'])
                torch.save(dict(logits=cached_teacher,teacher_sha256=teacher_identity['sha256'],
                    source_scope='TRAIN only',selection={k:ix.tolist() for k,ix in selection.items()},
                    TEST_inputs=False),out/'TRAIN_reference_teacher_logits.pt')
                put(out/'teacher_identity.json',dict(checkpoint_SHA256=teacher_identity['sha256'],
                    config_SHA256=teacher_identity['config_sha256'],TRAIN_only=True,
                    reference_D_Physical_has_historical_posthoc_TEST_design=phase==4 and a.architecture=='moe'))
                del teacher;teacher=None;torch.cuda.empty_cache()
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
            baseline=evaluate(model,selection_data,names,device,c['batch_size']);best=score(baseline);best_epoch=0;wait=0
            history=[dict(epoch=0,selection_split=selection_split,selection_metrics=summary(baseline),selection_score=best)]
            def save(file,epoch):
                torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    architecture=a.architecture,stage=phase,epoch=epoch,best_epoch=best_epoch,
                    config_sha256=sha(a.config),selection=selection_label,TEST_gradients=False,
                    TEST_selected_development=selection_split=='test',
                    parent_checkpoint_sha256=chain[-1]['best_sha256'] if chain else source['sha256']),out/file)
            save('best_checkpoint.pt',0)
            phase_began=time.monotonic()
            for epoch in range(1,c['stage_epochs'][phase-1]+1):
                epoch_start=time.time();model.train();rng=np.random.default_rng(c['seed']+phase*1000+epoch)
                batches=[]
                for n in names:
                    cycles=1 if n==names[-1] else c['memory_cycles_per_epoch']
                    for _ in range(cycles):
                        if n!='eurosat':
                            groups=rng.permutation(len(selection[n])//2)
                            perm=(groups[:,None]*2+np.arange(2)).reshape(-1)
                        else:perm=rng.permutation(len(selection[n]))
                        batches.extend((n,perm[s:s+c['batch_size']]) for s in range(0,len(perm),c['batch_size']))
                rng.shuffle(batches);loss_sum=0.;correct=0;seen=0
                for n,relative in batches:
                    ix=selection[n][relative]
                    if n==names[-1] and current is not None:
                        x=current.get_batch(ix,device);y=torch.as_tensor(np.asarray(current.labels)[ix],device=device,dtype=torch.long)
                    else:x=train[n]['x'][ix].to(device);y=train[n]['y'][ix].to(device)
                    optimizer.zero_grad(set_to_none=True);logits=model(x)['logits']
                    loss=(reference_TRAIN_objective(logits,y,cached_teacher[n][relative].to(device),c)
                          if reference_mode else F.cross_entropy(logits,y,label_smoothing=c['label_smoothing']))
                    if n=='clevr' and c.get('clevr_pairwise_weight',0):
                        loss=loss+c['clevr_pairwise_weight']*clevr_pairwise_loss(logits,y)
                    if n=='physical_binary_raw' and c.get('physical_pairwise_weight',0):
                        loss=loss+c['physical_pairwise_weight']*clevr_pairwise_loss(logits,y)
                    if n in cached_teacher and not reference_mode:
                        temp=c['distillation_temperature'];old=cached_teacher[n][relative].to(device)
                        kd=F.kl_div(F.log_softmax(logits/temp,1),F.softmax(old/temp,1),reduction='batchmean')*temp*temp
                        loss=loss+c['distillation_weight']*kd
                    if a.architecture=='moe' and n!=names[-1]:loss=loss*c.get('moe_old_task_loss_weight',1.)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite TRAIN loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step()
                    loss_sum+=float(loss.detach())*len(y);seen+=len(y);correct+=int((logits.argmax(1)==y).sum())
                    if seen%c.get('progress_every_records',4000)<len(y):
                        put(a.out/'status.json',dict(state='training',phase=name,epoch=epoch,
                            TRAIN_seen=seen,TRAIN_total=sum(len(selection[z])*(1 if z==names[-1] else c['memory_cycles_per_epoch']) for z in names),
                            running_loss=loss_sum/seen,updated=time.time(),gpu=gpu))
                validation=evaluate(model,selection_data,names,device,c['batch_size']);value=score(validation)
                if frozen(model)!=immutable or tensor_sha(model.display_gains)!=gain_sha:
                    raise RuntimeError('Old expert or fixed display gain changed')
                if value>best+1e-10:
                    best=value;best_epoch=epoch;wait=0;save('best_checkpoint.pt',epoch)
                else:wait+=1
                save('last_checkpoint.pt',epoch)
                row=dict(epoch=epoch,loss=loss_sum/seen,TRAIN_accuracy=correct/seen,TRAIN_records=seen,
                    selection_split=selection_split,selection_metrics=summary(validation),selection_score=value,best_epoch=best_epoch,
                    best_score=best,seconds=time.time()-epoch_start)
                history.append(row);put(out/'history.json',history)
                put(a.out/'status.json',dict(state='training',phase=name,**row,updated=time.time(),gpu=gpu))
                print(json.dumps(dict(phase=name,**row)),flush=True)
                if time.monotonic()-phase_began>=c.get('max_phase_fit_seconds',float('inf')):break
                if epoch>=c['minimum_epochs'] and wait>=c['patience']:break
            checkpoint=torch.load(out/'best_checkpoint.pt',map_location=device,weights_only=False)
            model.load_state_dict(checkpoint['model'],strict=True)
            if not (out/'last_checkpoint.pt').exists():raise RuntimeError('No trained last')
            test=load_data(a.test,'test')
            selected=evaluate(model,test,names,device,c['batch_size']);del test
            put(out/'selected_test.json',dict(scope='adapted simulation, not hardware',
                TEST_gradients=False,TEST_selection=selection_split=='test',
                independent_generalization_claim=False,selected_epoch=best_epoch,scores=selected))
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
