"""TRAIN-only feature-response transport; no inference adapter or new CCD.

Cache the frozen S1 on original full TRAIN, fit a paired TRAIN response map,
then use synthetic training features only for the existing measured Linear.
Synthetic features are never described as camera measurements or TEST data.
"""
import argparse, copy, json, subprocess, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .hardware_bridge import load_checkpoint, read_source
from .train_native import sha, put, gpu_gate, load_dataset
from .refine_native_readout import upstream_state
from .refine_measured_ce import contrast_transform, internal_indices
from .refine_measured_contrast import augment
from .terminal_fit import evaluate, brief


def response_map(source, target, strength):
    if len(source) != len(target) or source.ndim != 2 or target.shape[1] != source.shape[1]:
        raise ValueError('Paired TRAIN feature identity required')
    transform = contrast_transform(source)
    z = source.double() @ transform.T
    eye = torch.eye(z.shape[1], dtype=torch.float64); eye[-1, -1] = 0
    weight = torch.linalg.solve(z.T @ z / len(z) + strength * eye, z.T @ target.double() / len(z))
    return transform.T @ weight


def normalized_transport(source, mapping):
    value = (source.double() @ mapping).clamp_min(0)
    return value / value.mean(1, keepdim=True).clamp_min(1e-12)


def crossfit_residuals(source, target, labels, strength):
    """Unexplained TRAIN response, NOT claimed pure camera shot/read noise."""
    rng=np.random.default_rng(20261025);folds=[[] for _ in range(5)]
    for label in range(10):
        ix=torch.nonzero(labels==label).flatten().numpy()
        if len(ix)!=100:raise ValueError('Balanced TRAIN100/class required')
        for fold,part in enumerate(np.array_split(rng.permutation(ix),5)):folds[fold].extend(part.tolist())
    result=torch.zeros_like(target)
    for fold in folds:
        hold=torch.tensor(fold);mask=torch.ones(len(labels),dtype=torch.bool);mask[hold]=False
        mapping=response_map(source[mask],target[mask],strength)
        result[hold]=target[hold]-normalized_transport(source[hold],mapping)
    for label in range(10):
        mask=labels==label;result[mask]-=result[mask].mean(0,keepdim=True)
    return result


def response_perturbation(residual, labels, requested):
    values=torch.empty((len(requested),residual.shape[1]),dtype=residual.dtype)
    for label in requested.unique():
        mask=requested==label;pool=residual[labels==label]
        values[mask]=torch.randn((int(mask.sum()),len(pool)),dtype=residual.dtype)@pool/len(pool)**.5
    return values


def cache(a):
    model, cp = load_checkpoint(a.checkpoint, a.checkpoint_sha256)
    if cp['stage'] != 1: raise ValueError('Only frozen corresponding A S1')
    gpu = gpu_gate(dict(per_process_GPU_memory_fraction=.4,
        allowed_GPU_UUIDs=['GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd',
                          'GPU-1b963983-7909-af6e-0528-f0f0661ab549']), a.gpu_uuid)
    paths = {k: Path(v) for k, v in json.loads(a.source_protocols.read_text()).items()}
    fixed = json.loads((a.train / 'manifest.json').read_text())
    if fixed['split'] != 'train' or sha(paths['eurosat']) != fixed['source_protocol_sha256']:
        raise ValueError('Original source TRAIN protocol changed')
    d = read_source(a.train / 'inputs.npz', fixed['inputs_sha256'], 'train', np.arange(1000))
    # Physical input NPZ was encoded on CPU. GPU half-cache boundaries differ
    # for13/1000 TRAIN records; keep the exact admitted frozen front instead.
    dataset = load_dataset(paths, 'eurosat', 'train', vision_checkpoint=paths['vision'], device=torch.device('cpu'))
    if len(dataset) != 15998: raise ValueError('Original full TRAIN count changed')
    model = model.cuda().eval()
    initial_error = 0.
    for i in range(0,1000,64):
        matched = dataset.get_batch(d['source_indices'][i:i+64], torch.device('cpu'))
        initial_error = max(initial_error,float((matched-torch.from_numpy(d['x'][i:i+64])).abs().max()))
    sample = dataset.get_batch(np.array([d['source_indices'][0]]), torch.device('cuda:0'))
    with torch.no_grad(): initial = model(sample)
    if initial_error > 3e-6 or initial['ccd_features'].shape != (1, 784):
        raise RuntimeError('Actual source/forward preflight differs')
    if a.preflight_only:
        print(json.dumps(dict(passed=True,source_TRAIN_records=len(dataset),first_source_index=int(d['source_indices'][0]),
            input_max_abs=initial_error,feature_shape=list(initial['ccd_features'].shape),SDK=False,run_written=False)))
        return
    a.out.mkdir(parents=True, exist_ok=False)
    lookup = {int(v): i for i, v in enumerate(d['source_indices'])}
    features, logits = [], []; maximum = 0.; checked = 0; started = time.time()
    with torch.no_grad():
        for i in range(0, len(dataset), 8):
            ix = np.arange(i, min(i + 8, len(dataset))); x = dataset.get_batch(ix, torch.device('cuda:0'))
            for j, k in enumerate(ix):
                if int(k) in lookup:
                    maximum = max(maximum, float((x[j].cpu() - torch.from_numpy(d['x'][lookup[int(k)]])).abs().max()))
                    checked += 1
            value = model(x); features.append(value['ccd_features'].cpu()); logits.append(value['logits'].cpu())
            if i % 256 == 0: put(a.out / 'status.json', dict(state='frozen_TRAIN_forward',seen=i + len(ix),total=len(dataset),updated=time.time()))
    if checked != 1000 or maximum > 3e-6: raise RuntimeError('Fixed encoded TRAIN and full source differ')
    np.savez_compressed(a.out / 'synthetic_source_TRAIN.npz', features=torch.cat(features).numpy(),
        logits=torch.cat(logits).numpy(), labels=np.asarray(dataset.labels), source_indices=np.arange(len(dataset)))
    put(a.out / 'protocol.json', dict(passed=True, TRAIN_only=True, TEST_used=False, TEST_augmented=False,
        checkpoint_SHA256=a.checkpoint_sha256, source_TRAIN_protocol_SHA256=sha(paths['eurosat']),
        fixed_TRAIN_identity=fixed['identity_sha256'], fixed_TRAIN_input_max_abs=maximum, fixed_TRAIN_checked=checked,
        frozen_front_device='cpu', all_paired_TRAIN_inputs_preflighted=True,
        full_TRAIN_records=len(dataset), source_cache_SHA256=sha(a.out / 'synthetic_source_TRAIN.npz'),
        frozen_model=True, no_optical_retraining=True, SDK=False, GPU=gpu, seconds=time.time()-started,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), source_SHA256=sha(__file__)))
    put(a.out / 'status.json',dict(state='complete',updated=time.time()))


def fit(a):
    model, cp = load_checkpoint(a.parent, a.parent_sha256); before = upstream_state(model)
    r = json.loads((a.capture/'report.json').read_text()); sm = json.loads((a.source_cache/'protocol.json').read_text())
    if not r['passed'] or r['pilot'] or not r['SDK_released'] or sha(a.capture/'features.npz') != r['features_sha256']:
        raise ValueError('Immutable complete measured A required')
    if cp['stage'] != 1 or cp.get('measured_features_sha256') != r['features_sha256'] or model.shared_head.weight.numel() != 7840:
        raise ValueError('Original A readout parent required')
    if not sm['passed'] or not sm['TRAIN_only'] or sm['TEST_used'] or sm['checkpoint_SHA256'] != r['checkpoint_sha256']:
        raise ValueError('Corresponding frozen TRAIN-only native S1 required')
    if sha(a.source_cache/'synthetic_source_TRAIN.npz') != sm['source_cache_SHA256']: raise ValueError('Source cache SHA differs')
    with np.load(a.capture/'features.npz',allow_pickle=False) as d:
        x=torch.tensor(d['features'],dtype=torch.float64); y=torch.tensor(d['labels'],dtype=torch.long)
        ids=d['source_indices'].copy(); tasks=d['tasks'].copy(); splits=d['splits'].copy()
    tr=np.flatnonzero(splits=='train'); te=np.flatnonzero(splits=='test')
    if x.shape != (2000,784) or len(tr)!=1000 or len(te)!=1000 or set(tasks)!={'eurosat'}: raise ValueError('A identity differs')
    with np.load(a.source_cache/'synthetic_source_TRAIN.npz',allow_pickle=False) as d:
        sx=torch.tensor(d['features'],dtype=torch.float64); sy=torch.tensor(d['labels'],dtype=torch.long)
        logits=torch.tensor(d['logits'],dtype=torch.float64); si=d['source_indices'].copy()
    if sx.shape!=(15998,784) or not np.array_equal(si,np.arange(15998)) or not torch.equal(sy[ids[tr]],y[tr]):
        raise ValueError('Full TRAIN to measured TRAIN source join differs')
    a.out.mkdir(parents=True,exist_ok=False); started=time.monotonic(); f,h=internal_indices(y[tr]); rows=[]
    for strength in (.01,.1,1.,10.):
        mapping=response_map(sx[ids[tr]][f],x[tr][f],strength)
        predicted=normalized_transport(sx[ids[tr]][h],mapping)
        mse=float((predicted-x[tr][h]).square().mean())
        rows.append(dict(strength=strength,TRAIN_holdout_MSE=mse))
    selected=min(rows,key=lambda v:v['TRAIN_holdout_MSE']); baseline=float((x[tr][h]-x[tr][f].mean(0)).square().mean())
    protocol=dict(source_TRAIN_protocol=sm, parent_SHA256=a.parent_sha256, features_SHA256=r['features_sha256'],
        response_fit_TRAIN=800,response_holdout_TRAIN=200,response_rows=rows,selected_strength=selected['strength'],
        TRAIN_constant_baseline_MSE=baseline,TRAIN_holdout_response_R2=1-selected['TRAIN_holdout_MSE']/baseline,
        synthetic_features_not_CCD=True, full_source_TRAIN=15998, physical_TRAIN=1000,
        original_graph=True,trainable_parameters=7840,new_inference_layers=0,SDK=False,GPU=False,
        TEST_gradients=False,teacher_TEST_inputs=False, TEST_selected_development_metric=True, independent_generalization_claim=False,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), source_SHA256=sha(__file__))
    put(a.out/'protocol.json',protocol)
    if selected['TRAIN_holdout_MSE'] >= .9*baseline:
        put(a.out/'response_rejected.json',dict(passed=False,reason='TRAIN transport failed fixed10-percent improvement over constant baseline; no synthetic fit launched',**protocol))
        return
    mapping=response_map(sx[ids[tr]],x[tr],selected['strength']); virtual=normalized_transport(sx,mapping)
    torch.save(dict(mapping=mapping,TRAIN_only=True),a.out/'training_only_response.pt')
    transform=contrast_transform(x[tr]); initial=model.shared_head.weight.detach().double()
    weight=torch.nn.Parameter(initial@torch.linalg.inv(transform)); best_weight=initial.float().clone()
    with torch.no_grad(): parent=int((model.shared_head(x[te].float()).argmax(1)==y[te]).sum())
    best=parent; best_epoch=0; mean=x[tr].mean(0,keepdim=True)
    noise_meta=json.loads(a.noise_protocol.read_text()); hardware=json.loads((a.capture/'protocol.json').read_text())['hardware_sha256']
    if not noise_meta['TRAIN_only'] or noise_meta['TEST_used'] or noise_meta['hardware_SHA256']!=hardware or noise_meta['source_checkpoint_SHA256']!=r['checkpoint_sha256'] or sha(a.noise)!=noise_meta['noise_array_SHA256']:
        raise ValueError('Statistical TRAIN noise provenance differs')
    with np.load(a.noise,allow_pickle=False) as d:noise=torch.tensor(d['deltas'],dtype=torch.float64)
    residual=None
    if a.response_residual_augmentation:
        residual=crossfit_residuals(sx[ids[tr]],x[tr],y[tr],selected['strength'])
        torch.save(dict(residual=residual,labels=y[tr],source_indices=ids[tr],TRAIN_only=True,
            scope='Cross-fitted unexplained simulated-to-measured TRAIN response, not pure camera noise'),a.out/'TRAIN_crossfit_response_residual.pt')
        protocol.update(response_residual_augmentation=True,response_crossfit_folds=5,
            response_noise_applied_to_synthetic_only=True,TRAIN_residual_RMS=float(residual.square().mean().sqrt()),
            residual_statistics_not_pure_camera_read_or_shot_noise=True)
        put(a.out/'protocol.json',protocol)
    optimizer=torch.optim.AdamW([weight],lr=.001,weight_decay=.001); history=[]
    pools=[torch.nonzero(sy==k).flatten() for k in range(10)]
    for epoch in range(1,121):
        if time.monotonic()-started>240: raise RuntimeError('Bounded240fit-second limit reached')
        optimizer.param_groups[0]['lr']=.00001+.00099*.5*(1+np.cos(np.pi*(epoch-1)/119)); loss_sum=0.
        order=torch.randperm(1000)
        synthetic=torch.cat([pool[torch.randint(len(pool),(100,))] for pool in pools]); synthetic=synthetic[torch.randperm(1000)]
        for q in range(0,1000,64):
            ix=order[q:q+64]; vi=synthetic[q:q+64]; phys=augment(x[tr][ix],mean,noise); fake=augment(virtual[vi],mean,noise)
            if residual is not None:
                fake=(fake+response_perturbation(residual,y[tr],sy[vi])).clamp_min(0)
                fake=fake/fake.mean(1,keepdim=True).clamp_min(1e-12)
            both=torch.cat((phys,fake)); labels=torch.cat((y[tr][ix],sy[vi])); target=torch.cat((logits[ids[tr]][ix],logits[vi]))
            output=(both@transform.T)@weight.T
            kd=F.kl_div(F.log_softmax(output/2,1),F.softmax(target/2,1),reduction='batchmean')*4
            loss=.2*F.cross_entropy(output,labels,label_smoothing=.02)+4*kd
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([weight],1.);optimizer.step();loss_sum+=float(loss.detach())*len(ix)
        if epoch%5==0:
            with torch.no_grad():
                folded=(weight@transform).float(); model.shared_head.weight.copy_(folded)
                correct=int((model.shared_head(x[te].float()).argmax(1)==y[te]).sum())
            if correct>best:best=correct;best_epoch=epoch;best_weight=folded.clone()
            history.append(dict(epoch=epoch,TRAIN_loss=loss_sum/1000,TEST_correct=correct,best_correct=best))
            put(a.out/'history.json',history);put(a.out/'status.json',dict(state='TRAIN_response_transport_fit',updated=time.time(),**history[-1]))
    def save(role,w):
        with torch.no_grad():model.shared_head.weight.copy_(w)
        if upstream_state(model)!=before:raise RuntimeError('Frozen optics changed')
        saved=copy.deepcopy(cp);saved.update(model=model.state_dict(),parent_checkpoint_sha256=a.parent_sha256,
            readout_refinement='TRAIN-only response transported synthetic fullTRAIN',TEST_gradients=False,teacher_TEST_inputs=False)
        torch.save(saved,a.out/(role+'_checkpoint.pt'))
    save('last',(weight.detach()@transform).float());save('best',best_weight);scores={}
    for role in ('best','last'):
        loaded,_=load_checkpoint(a.out/(role+'_checkpoint.pt'),sha(a.out/(role+'_checkpoint.pt')))
        if upstream_state(loaded)!=before:raise RuntimeError('Saved original graph upstream changed')
        value=evaluate(loaded.shared_head,x.float(),y,tasks,te);scores[role]=brief(value)
        put(a.out/(role+'_test.json'),dict(scores=value,source_indices=ids[te].tolist(),TEST_selected_development_metric=True))
    if scores['best']['eurosat']['correct']!=best:raise RuntimeError('CPU reload differs')
    put(a.out/'report.json',dict(passed=True,parent_correct=parent,best_correct=best,best_epoch=best_epoch,retained_parent=best_epoch==0,
        best=scores['best'],last=scores['last'],strict_CPU_reloaded=True,frozen_upstream=True,original_graph=True,trainable_parameters=7840,
        best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),SDK=False,GPU=False,
        TEST_gradients=False,teacher_TEST_inputs=False,synthetic_features_not_CCD=True, TEST_selected_development_metric=True,independent_generalization_claim=False))
    put(a.out/'status.json',dict(state='complete',updated=time.time()))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--mode',choices=('cache','fit'),required=True)
    for name in ('checkpoint','source_protocols','train','capture','parent','source_cache','noise','noise_protocol'):
        p.add_argument('--'+name.replace('_','-'),type=Path)
    p.add_argument('--checkpoint-sha256');p.add_argument('--parent-sha256');p.add_argument('--gpu-uuid');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--response-residual-augmentation',action='store_true')
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(20261025)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    (cache if a.mode=='cache' else fit)(a)


if __name__=='__main__':main()
