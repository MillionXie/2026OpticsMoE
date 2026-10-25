"""Finite T20 training on audited data, frozen shared T21 frontend/physical graph.

No SDK, TEST gradients/teachers, retrospective score clipping or per-domain PTs.
This is a new native8 experiment, not a reproduction of the old17um bars.
"""
import argparse,copy,json,subprocess,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from LightGenV2.tasks.t09_multimodal_matching.vision import VisionEncoder
from LightGenV2.tasks.t21_continual_learning_hardware.train_native import gpu_gate,put,sha,tensor_sha
from .unified_native import encode_single,build_unified,FRONTEND_SHA256,CONTRACT


def validate(cfg):
    if cfg!=json.loads(Path(__file__).with_name('config_unified_static_20261025.json').read_text()):raise ValueError('Exact published finite native experiment required')


def load_arrays(folder,split):
    manifest=json.loads((folder/'manifest.json').read_text());member='source_data/data/data.npz' if split!='test' else 'data/test_original_spatial_2000.npz';path=folder/member
    expected=next(v['sha256'] for v in manifest['files'] if v['member']==member)
    if sha(path)!=expected:raise ValueError('Audited source data changed')
    with np.load(path,allow_pickle=False) as d:result={k:d[split+'_'+k].copy() for k in ('images','labels','domains','ids')}
    n=6000 if split=='train' else 2000
    if result['images'].shape!=(n,56,56,3) or len(set(result['ids']))!=n or any(np.bincount(result['labels'][result['domains']==v],minlength=10).tolist()!=[300 if split=='train' else 100]*10 for v in (0,1)):raise ValueError('Fixed complete domain/split identity changed')
    return result


@torch.no_grad()
def cache_frontend(data,checkpoint):
    if sha(checkpoint)!=FRONTEND_SHA256:raise ValueError('Exact existing T21 frozen frontend required')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg=saved['config']
    if cfg['feature_parameters']!=32128 or cfg['temporary_head_parameters']!=3096 or cfg['test_policy']!='never read':raise ValueError('Audited original visual frontend required')
    model=VisionEncoder();model.load_state_dict(saved['model'],strict=True);model.requires_grad_(False).eval()
    result=torch.zeros((len(data['labels']),128),dtype=torch.float16);ix=np.flatnonzero(data['domains']==0)
    for start in range(0,len(ix),128):
        rows=ix[start:start+128];result[rows]=model(torch.from_numpy(data['images'][rows]))[0].to(torch.float16)
    if not torch.isfinite(result).all():raise ValueError('Nonfinite fixed frontend')
    return result


def batch_amplitude(data,features,ix,device):
    # Match T21's frozenCPU frontend -> float16 cache -> float32 encoding.
    return encode_single(torch.from_numpy(data['images'][ix]),torch.as_tensor(data['domains'][ix]),features[ix].float()).to(device)


def balanced_batches(data,domains,batch,seed):
    rng=np.random.default_rng(seed);pools=[rng.permutation(np.flatnonzero(data['domains']==d)) for d in domains]
    half=batch//len(domains)
    if len({len(v) for v in pools})!=1 or batch%len(domains):raise ValueError('Balanced same-domain update budget required')
    for start in range(0,len(pools[0]),half):yield np.concatenate([v[start:start+half] for v in pools])


@torch.no_grad()
def evaluate(model,data,features,device,batch=8):
    model.eval();pred=[];probs=[];route=[];losses=[]
    for start in range(0,len(data['labels']),batch):
        ix=np.arange(start,min(start+batch,len(data['labels'])));r=model(batch_amplitude(data,features,ix,device));y=torch.as_tensor(data['labels'][ix],device=device)
        pred.extend(r['logits'].argmax(1).cpu().tolist());probs.extend(r['probabilities'].cpu().tolist());losses.extend(F.cross_entropy(r['logits'],y,reduction='none').cpu().tolist())
        if r['route_power'] is not None:route.extend(r['route_power'].cpu().tolist())
    pred=np.asarray(pred);scores={}
    for d,name in ((0,'RGB'),(1,'SAR')):
        mask=data['domains']==d;y=data['labels'][mask];p=pred[mask];cm=np.zeros((10,10),dtype=int);np.add.at(cm,(y,p),1)
        scores[name]=dict(n=int(mask.sum()),correct=int((p==y).sum()),accuracy=float((p==y).mean()),loss=float(np.mean(np.asarray(losses)[mask])),confusion_matrix=cm.tolist())
    routing=None
    if route:
        q=np.asarray(route);active=model.active_indices[:4].cpu().numpy();winner=q.argmax(1)
        routing=dict(active_slot_ids=(active+1).tolist(),mean_power=q.mean(0).tolist(),std_power=q.std(0).tolist(),winner_fraction=np.bincount(winner,minlength=16).tolist(),winner_changes_with_input=bool(len(set(winner.tolist()))>1))
    return dict(scores=scores,labels=data['labels'].tolist(),domains=data['domains'].tolist(),ids=data['ids'].tolist(),predictions=pred.tolist(),probabilities=probs,routing=routing)


def selection(report,domains):
    names=['RGB' if d==0 else 'SAR' for d in domains];scores=[report['scores'][n] for n in names]
    return (float(np.mean([v['accuracy'] for v in scores])),min(v['accuracy'] for v in scores),-float(np.mean([v['loss'] for v in scores])))


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('data','frontend','profile','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--case',choices=('MoE_AB','D2NN_RGB','D2NN_AB_control'),required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    cfg=json.loads(a.profile.read_text());validate(cfg);torch.set_num_threads(2);torch.manual_seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    gpu=gpu_gate(cfg,a.gpu_uuid);device=torch.device('cuda:0');architecture='moe' if a.case=='MoE_AB' else 'd2nn';domains=[0] if a.case=='D2NN_RGB' else [0,1]
    train=load_arrays(a.data,'train');model=build_unified(architecture,cfg['seed']);features=cache_frontend(train,a.frontend)
    calibration=np.concatenate([np.flatnonzero((train['domains']==d)&(train['labels']==c))[:2] for c in range(10) for d in domains]);model.to(device)
    with torch.no_grad():gains=model.calibrate_gains(batch_amplitude(train,features,calibration,device))
    fixed_gain=tensor_sha(model.display_gains);unused={k:tensor_sha(v) for k,v in model.named_parameters() if not v.requires_grad}
    first=next(balanced_batches(train,domains,cfg['batch_size'],cfg['seed']));amp=batch_amplitude(train,features,first,device);result=model(amp);loss=F.cross_entropy(result['logits'],torch.as_tensor(train['labels'][first],device=device));loss.backward()
    if not torch.isfinite(loss) or any(v.grad is not None for n,v in model.named_parameters() if not v.requires_grad) or any(v.grad is None or not torch.isfinite(v.grad).all() for v in model.parameters() if v.requires_grad):raise RuntimeError('Real TRAIN finite/freeze gradient preflight failed')
    if a.preflight_only:
        print(json.dumps(dict(passed=True,case=a.case,loss=float(loss.detach()),original_shared_head_parameters=model.shared_head.weight.numel(),active_experts=4 if architecture=='moe' else 0,gain_calibration=gains,TEST_loaded=False,SDK=False)));return
    model.zero_grad(set_to_none=True);val=load_arrays(a.data,'validation');val_features=cache_frontend(val,a.frontend);a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    dependencies=[Path(__file__),Path(__file__).with_name('unified_native.py'),Path('LightGenV2/tasks/t21_continual_learning_hardware/model.py'),Path('LightGenV2/tasks/t20_reconfigurable_rgb_sar/hardware_aware.py'),Path('LightGenV2/tasks/t16_zero_phase_ccd_lifelong/frozen_vision.py')]
    put(a.out/'protocol.json',dict(contract=CONTRACT,case=a.case,architecture=architecture,config=cfg,config_sha256=sha(a.profile),training_domains=domains,TRAIN_records=int(np.isin(train['domains'],domains).sum()),validation_records=1000*len(domains),
        dataset_manifest_sha256=sha(a.data/'manifest.json'),frontend_sha256=FRONTEND_SHA256,frozen_frontend_parameters=32128,frontend_training='Original frozen CLEVR pretraining; no EuroSAT joint frontend pretraining',
        input_contract='Single observed RGB or SAR, missing sensor tiles zero; T21 RGB visual strip and raw SAR VV/VH mean; no labels/IDs in encoding',
        physical_graph='Exact T21 native8 A optical backend:16slots/4active forMoE, two classification layers/OEO; original7840 Linear',gain_calibration=gains,fixed_gain_sha256=fixed_gain,calibration_TRAIN_ids=train['ids'][calibration].tolist(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256={str(v):sha(v) for v in dependencies},SDK=False,TEST_gradients=False,TEST_teacher=False,selection_uses_TEST=False,gpu=gpu,
        comparison_scope='Primary unequal-domain training budget reproduces original question; joint D2NN control isolates architecture more fairly',target_is_not_guaranteed=True,no_score_clipping=True))
    def save(role,epoch):
        torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},architecture=architecture,stage=1,epoch=epoch,contract=CONTRACT,case=a.case,config_sha256=sha(a.profile),frontend_sha256=FRONTEND_SHA256,TEST_gradients=False),a.out/(role+'_checkpoint.pt'))
    baseline=evaluate(model,val,val_features,device);best=selection(baseline,domains);best_epoch=0;save('best',0);history=[];last_improvement=0
    optics=[v for n,v in model.named_parameters() if v.requires_grad and not n.startswith('shared_head.')];params=[v for v in model.parameters() if v.requires_grad]
    optimizer=torch.optim.AdamW([dict(params=optics,lr=cfg['optical_lr']),dict(params=model.shared_head.parameters(),lr=cfg['head_lr'])],weight_decay=cfg['weight_decay'])
    for epoch in range(1,cfg['epochs']+1):
        model.train();total=0.;seen=0;correct=0
        fraction=.1+.9*.5*(1+np.cos(np.pi*(epoch-1)/max(1,cfg['epochs']-1)))
        for group,base in zip(optimizer.param_groups,(cfg['optical_lr'],cfg['head_lr'])):group['lr']=base*fraction
        for ix in balanced_batches(train,domains,cfg['batch_size'],cfg['seed']+epoch):
            optimizer.zero_grad(set_to_none=True);r=model(batch_amplitude(train,features,ix,device));y=torch.as_tensor(train['labels'][ix],device=device)
            cost=F.cross_entropy(r['logits'],y,label_smoothing=cfg['label_smoothing'])
            if architecture=='moe':cost+=cfg['routing_balance_weight']*(r['route_power'].index_select(1,model.active_indices[:4]).mean(0)-.25).square().sum()
            if not torch.isfinite(cost):raise RuntimeError('Nonfinite original native TRAIN loss')
            cost.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step();total+=float(cost.detach())*len(y);seen+=len(y);correct+=int((r['logits'].argmax(1)==y).sum())
            if seen%1000<len(y):put(a.out/'status.json',dict(state='training',epoch=epoch,TRAIN_seen=seen,loss=total/seen,updated=time.time()))
        scores=evaluate(model,val,val_features,device);value=selection(scores,domains)
        if tensor_sha(model.display_gains)!=fixed_gain or any(tensor_sha(dict(model.named_parameters())[n])!=v for n,v in unused.items()):raise RuntimeError('Frozen gain or inactive phase changed')
        if value>best:best=value;best_epoch=epoch;last_improvement=epoch;save('best',epoch);put(a.out/'selected_validation.json',scores)
        save('last',epoch);history.append(dict(epoch=epoch,TRAIN_loss=total/seen,TRAIN_accuracy=correct/seen,validation=scores['scores'],routing=scores['routing'],best_epoch=best_epoch));put(a.out/'history.json',history);put(a.out/'status.json',dict(state='training',**history[-1],updated=time.time()))
        if time.monotonic()-started>cfg['max_case_seconds']:raise RuntimeError('Finite case wall-time bound reached at full epoch')
        if epoch>=cfg['minimum_epochs'] and epoch-last_improvement>=cfg['patience']:break
    locked=dict(best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),best_epoch=best_epoch,TEST_loaded=False);put(a.out/'locked_before_TEST.json',locked)
    cp=torch.load(a.out/'best_checkpoint.pt',map_location=device,weights_only=False);model.load_state_dict(cp['model'],strict=True);test=load_arrays(a.data,'test');test_features=cache_frontend(test,a.frontend);scores=evaluate(model,test,test_features,device);put(a.out/'selected_test.json',scores)
    if sha(a.out/'best_checkpoint.pt')!=locked['best_sha256']:raise RuntimeError('Locked checkpoint changed')
    put(a.out/'completion.json',dict(passed=True,case=a.case,architecture=architecture,best_epoch=best_epoch,epochs_completed=epoch,test=scores['scores'],routing=scores['routing'],best_sha256=locked['best_sha256'],last_sha256=locked['last_sha256'],
        original_graph_CPU_audit_pending=True,SDK=False,TEST_gradients=False,TEST_teacher=False,selection_uses_TEST=False,historical_TEST_pool_reuse=True,independent_new_blind_TEST_claim=False,seconds=time.monotonic()-started))
    put(a.out/'status.json',dict(state='complete_CPU_audit_pending',updated=time.time()))


if __name__=='__main__':main()
