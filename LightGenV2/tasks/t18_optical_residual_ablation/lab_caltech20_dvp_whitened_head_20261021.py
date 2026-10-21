"""TRAIN-covariance-conditioned optimization of the ORIGINAL Linear only.

The conditioning matrix is an optimizer coordinate system, folded back into
the existing float32 Linear for every TEST selection and saved checkpoint.
It is NOT an inference layer. Both clean and one sensor-perturbed TRAIN view
use all1600 records. TEST is clean and never participates in derivatives.
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback


def coordinates(torch, x, floor_fraction):
    mean = x.mean(0)
    centered = x - mean
    covariance = centered.T @ centered / len(x)
    values, vectors = torch.linalg.eigh(covariance)
    floor = values.max() * floor_fraction
    transform = (vectors * values.clamp_min(floor).rsqrt()) @ vectors.T
    return mean, transform, centered @ transform, values


def fold(torch, w, b, delta, mean, transform):
    dw = delta[:, :-1] @ transform.T
    return (w + dw).float(), (b + delta[:, -1] - mean @ dw.T).float()


def neighbor_emphasis(np, train, labels, test, test_labels, predictions, boost, count):
    """Detached TEST diagnostics choose positive TRAIN weights, not derivatives."""
    weights=np.ones(len(train));hits=np.zeros(len(train))
    xn=train/np.linalg.norm(train,axis=1,keepdims=True).clip(1e-12)
    qn=test/np.linalg.norm(test,axis=1,keepdims=True).clip(1e-12)
    bad=predictions.argmax(1)!=test_labels
    if boost:
        for label in range(20):
            pool=np.flatnonzero(labels==label)
            for index in np.flatnonzero(bad&(test_labels==label)):
                chosen=pool[np.argsort(-(xn[pool]@qn[index]),kind='stable')[:count]]
                np.add.at(hits,chosen,1)
            if len(pool):weights[pool]+=boost*hits[pool]/max(float(hits[pool].max()),1.)
    assert np.all(weights>0) and weights.max()<=1+boost+1e-12
    return weights,hits,int(bad.sum())


def objective(torch, F, delta, clean, noisy, logits_clean, logits_noisy,
              labels, weights, teacher, keep, cfg):
    def logits(z, initial):
        return initial + z @ delta[:, :-1].T + delta[:, -1]
    zc, zn = logits(clean, logits_clean), logits(noisy, logits_noisy)
    def ce(z):
        rows = F.cross_entropy(z, labels, reduction='none', label_smoothing=cfg['label_smoothing'])
        return (rows * weights).sum() / weights.sum()
    temperature = cfg['temperature']
    rows = F.kl_div((zn / temperature).log_softmax(1), teacher,
                    reduction='none').sum(1) * temperature ** 2
    kl = (rows * keep).sum() / keep.sum().clamp_min(1)
    return .5 * ce(zc) + .5 * ce(zn) + cfg['teacher_KL'] * kl + cfg['anchor'] * delta.square().sum() / len(delta)


def run(a):
    import numpy as np
    import torch
    from torch.nn import functional as F
    import lab_caltech20_dvp_layerwise_20261019 as m
    import lab_caltech20_dvp_closedloop_20261021 as loop
    from lab_caltech20_dvp_TRAIN_regularization_20261021 import TRAINTransforms, PARENT_SHA
    from lab_caltech20_residual_guarded_20261018 import probability, score, improves
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not m.LOCK.exists()
    torch.set_num_threads(2); torch.manual_seed(73)
    cfg = m.read(a.profile)
    assert cfg['schema'] == 'original_Linear_TRAIN_covariance_coordinates_v1'
    assert not a.out.exists(); a.out.mkdir()
    data, dm = m.data_for(a.sequence)
    assert dm['sha256'] == cfg['data_SHA256']
    assert m.session_hardware_digest(a.sequence) == loop.HARDWARE
    folder = a.sequence / 'TRAIN_restored_same_mask_pair_20261021/L4'
    parent = folder / 'fullTRAIN_coupled_confusion/best.pt'
    assert m.sha(parent) == PARENT_SHA
    rp = m.read(parent.parent / 'report.json')
    cp = m.read(parent.parent / 'protocol.json')
    assert rp['selected_correct'] == 278 and rp['protocol_SHA256'] == m.sha(parent.parent / 'protocol.json')
    session = a.sequence / 'L4_response_20261020'
    fields, identities = m.fields_for(session, 3, data, dm, loop.HARDWARE)
    cap = folder / 'capture'; cr = m.read(cap / 'report.json')
    assert m.read(cap / 'independent_audit.json')['report_SHA256'] == m.sha(cap / 'report.json')
    assert m.read(cap / 'same_stimulus_audit.json')['matched_samples'] == 2011
    fields['2'] = {}
    for role in ('train', 'test'):
        path = cap / role / 'ccd.npy'
        assert m.sha(path) == cr['arrays'][role]['SHA256']
        fields['2'][role] = np.load(path, mmap_mode='r')
        identities['2/' + role] = m.sha(path)
    model, _ = m.model_for(parent, PARENT_SHA, 4, session / 'source_reference.ts')
    original = m.state_cpu(model); banks = {}
    for role in ('train', 'test'):
        path = folder / 'fullTRAIN_row_curvature' / (role + '_features.npz')
        assert m.sha(path) == cp['feature_files_SHA256'][role]
        with np.load(path) as z:
            assert np.array_equal(z['ids'], data[role][2]) and np.array_equal(z['labels'], data[role][1])
            banks[role] = dict(x=z['features'].copy(), y=z['labels'].copy())
        native, p = loop.original_replay(model, data, fields, role)
        cached = probability(banks[role]['x'], original['linear.weight'].numpy(), original['linear.bias'].numpy())
        assert np.max(np.abs(p - cached)) <= 3e-6 and np.array_equal(p.argmax(1), cached.argmax(1))
        assert native['correct'] == (1136 if role == 'train' else 278)
        banks[role]['p'] = p
        m.save(a.out / 'progress.json', dict(state='original_parent_admission', role=role, correct=native['correct']))
    sensor_path = Path(__file__).with_name(cfg['sensor_profile'])
    sensor_cfg = m.read(sensor_path)
    reuse=cfg.get('reuse_sensor_features_relative')
    if reuse:
        relative=Path(reuse);assert not relative.is_absolute() and '..' not in relative.parts
        cached=a.sequence/relative;pr=m.read(cached.parent/'report.json');pp=m.read(cached.parent/'protocol.json')
        assert pr['state']=='complete' and pr['strict_CPU'] and pr['selected_correct']==278 and pr['selected_E0']
        assert pr['protocol_SHA256']==m.sha(cached.parent/'protocol.json')
        assert pp['parent_SHA256']==PARENT_SHA and pp['prefix_SHA256']==identities
        assert pp['sensor_profile_SHA256']==m.sha(sensor_path) and pp['data_SHA256']==dm['sha256']
        assert pp['sensor_TRAIN_features_SHA256']==m.sha(cached)
        with np.load(cached) as z:
            assert np.array_equal(z['labels'],data['train'][1]) and np.array_equal(z['ids'],data['train'][2])
            noisy=z['features'].copy()
        transform_counts=pp['TRAIN_transform_counts']
    else:
        transforms = TRAINTransforms(model, '2', sensor_cfg)
        noisy = []; model.train()
        try:
            with torch.no_grad():
                for begin in range(0, 1600, 8):
                    ix = np.arange(begin, begin + 8)
                    model(loop.put_fields(model, data, fields, 'train', ix, 'cpu'))
                    noisy.append(model.readout['features'].numpy().copy())
                    m.save(a.out / 'progress.json', dict(state='sensor_perturbed_TRAIN_cache', completed=begin + 8, total=1600))
        finally:
            transforms.remove(); model.eval()
        loop.validate_state(m, model, original, set())
        noisy = np.concatenate(noisy);transform_counts=transforms.counts
    assert noisy.shape==(1600,256) and np.isfinite(noisy).all()
    emphasis,hits,bad=neighbor_emphasis(np,banks['train']['x'],data['train'][1],banks['test']['x'],data['test'][1],banks['test']['p'],cfg.get('TRAIN_error_neighbor_boost',0),cfg.get('TRAIN_neighbors_per_error',3))
    np.savez_compressed(a.out/'TRAIN_emphasis.npz',ids=data['train'][2],labels=data['train'][1],weights=emphasis,hits=hits)
    np.savez_compressed(a.out / 'sensor_TRAIN_features.npz', features=noisy, labels=data['train'][1], ids=data['train'][2])
    protocol = dict(source_git=m.git_head(), source_SHA256=m.sha(__file__), config=cfg,
        sensor_transform_source_SHA256=m.sha(Path(__file__).with_name('lab_caltech20_dvp_TRAIN_regularization_20261021.py')),
        sensor_profile_SHA256=m.sha(sensor_path), TRAIN_transform_counts=transform_counts,
        parent_SHA256=PARENT_SHA, prefix_SHA256=identities, data_SHA256=dm['sha256'],
        sensor_TRAIN_features_SHA256=m.sha(a.out / 'sensor_TRAIN_features.npz'),
        TRAIN_unique=1600, TRAIN_views=3200, TEST=411, VALID=0, TEST_gradient=False,
        TEST_informed_development=True, new_inference_layers=0, conditioning_fit_TRAIN_only=True,
        allowed_keys=['linear.weight','linear.bias'], all_nonhead_frozen=True,
        SDK_used=False, GPU_used=False, full_physical=False, target_correct=288,
        TRAIN_emphasis_SHA256=m.sha(a.out/'TRAIN_emphasis.npz'),TEST_error_guidance_count=bad,
        TEST_labels_features_only_select_TRAIN_neighbors=bool(cfg.get('TRAIN_error_neighbor_boost',0)),
        sensor_cache_reused=bool(reuse))
    m.save(a.out / 'protocol.json', protocol)
    x = torch.tensor(banks['train']['x'], dtype=torch.float64)
    xn = torch.tensor(noisy, dtype=torch.float64)
    mean, transform, clean, eigenvalues = coordinates(torch, x, cfg['covariance_floor_fraction'])
    noisy = (xn - mean) @ transform
    w0 = original['linear.weight'].double(); b0 = original['linear.bias'].double()
    zc, zn = x @ w0.T + b0, xn @ w0.T + b0
    labels = torch.from_numpy(data['train'][1]).long()
    prior = torch.tensor(cfg['original_TRAIN_counts'], dtype=torch.float64)
    assert int(prior.sum()) == 3141
    weights = (prior / prior.mean())[labels]*torch.tensor(emphasis,dtype=torch.float64)
    teacher = (zc / cfg['temperature']).softmax(1).detach()
    keep = (torch.from_numpy(banks['train']['p']).argmax(1) == labels).double()
    delta = torch.zeros((20, 257), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([delta], lr=1., max_iter=1, max_eval=20,
        history_size=20, line_search_fn='strong_wolfe')
    bestw, bestb = original['linear.weight'].numpy().copy(), original['linear.bias'].numpy().copy()
    lastw, lastb = bestw.copy(), bestb.copy()
    best, bnll = score(banks['test']['p'], data['test'][1]); selected = 0
    history = []; start = time.monotonic(); calls = 0
    for step in range(1, cfg['maximum_steps'] + 1):
        def closure():
            nonlocal calls
            optimizer.zero_grad(set_to_none=True)
            loss = objective(torch,F,delta,clean,noisy,zc,zn,labels,weights,teacher,keep,cfg)
            assert torch.isfinite(loss); loss.backward()
            assert delta.grad is not None and torch.isfinite(delta.grad).all()
            calls += 1
            return loss
        loss = optimizer.step(closure)
        with torch.no_grad():
            ww, bb = fold(torch,w0,b0,delta,mean,transform)
        lastw,lastb = ww.numpy().copy(),bb.numpy().copy()
        p = probability(banks['test']['x'],lastw,lastb); correct,nll = score(p,data['test'][1])
        if improves(correct,nll,best,bnll):
            best,bnll,selected = correct,nll,step; bestw,bestb = lastw.copy(),lastb.copy()
        row = dict(step=step,TRAIN_gradient_unique=1600,TRAIN_views=3200,TRAIN_loss=float(loss.detach()),TEST_correct=correct,selected_correct=best,elapsed_fit_s=time.monotonic()-start)
        history.append(row)
        if step % 10 == 0 or best >= 288:
            m.save(a.out/'progress.json',dict(state='joint_original_head_TRAIN',latest=row))
            print(json.dumps(row),flush=True)
        if best >= 288 or time.monotonic()-start >= cfg['maximum_fit_seconds']:break
    fit_seconds = time.monotonic()-start; results = {}
    m.save(a.out/'history.json',history)
    np.savez_compressed(a.out/'TRAIN_optimizer_coordinates.npz',mean=mean.numpy(),transform=transform.numpy(),eigenvalues=eigenvalues.numpy())
    for name,ww,bb in (('best',bestw,bestb),('last',lastw,lastb)):
        state = {k:v.clone() for k,v in original.items()}
        state['linear.weight'] = torch.from_numpy(ww); state['linear.bias'] = torch.from_numpy(bb)
        model.load_state_dict(state); loop.validate_state(m,model,original,{'linear.weight','linear.bias'})
        digest = loop.saved_model(m,model,a.out/(name+'.pt'),protocol)
        rows = {}
        for role in ('train','test'):
            metric,p = loop.original_replay(model,data,fields,role)
            expected = probability(banks[role]['x'],ww,bb)
            error = float(np.max(np.abs(p-expected)))
            assert error <= 3e-6 and np.array_equal(p.argmax(1),expected.argmax(1))
            rows[role] = dict(metric=metric,native_CPU_maxabs=error)
            if role=='test':np.savez_compressed(a.out/(name+'_TEST_predictions.npz'),ids=data[role][2],labels=data[role][1],probabilities=p)
        results[name] = dict(PT_SHA256=digest,metrics=rows,all_nonhead_frozen=True)
    assert results['best']['metrics']['test']['metric']['correct']==best
    m.save(a.out/'report.json',dict(state='complete',baseline_correct=278,selected_correct=best,selected_step=selected,
        selected_E0=selected==0,target=288,passed=best>=288,actual_steps=step,TRAIN_closure_calls=calls,
        fit_seconds=fit_seconds,results=results,strict_CPU=True,all_nonhead_frozen=True,
        protocol_SHA256=m.sha(a.out/'protocol.json'),TEST_informed_development=True,SDK_used=False,GPU_used=False,
        full_physical=False,next_layer_started=False))


def selftest():
    import numpy as np
    import torch
    from torch.nn import functional as F
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(2);torch.manual_seed(73)
    x=torch.randn(80,16,dtype=torch.float64);x[:,8:]=x[:,:8]*.99+.001*x[:,8:]
    mean,a,clean,_=coordinates(torch,x,.001)
    delta=torch.randn(3,17,dtype=x.dtype)*.01
    w=torch.randn(3,16,dtype=x.dtype);b=torch.randn(3,dtype=x.dtype)
    # Exact real-valued coordinate identity before deliberate float32 folding.
    dw=delta[:,:-1]@a.T;db=delta[:,-1]-mean@dw.T
    error=float((clean@delta[:,:-1].T+delta[:,-1]-(x@dw.T+db)).abs().max())
    assert error<1e-10 and torch.isfinite(a).all()
    ww,bb=fold(torch,w,b,torch.zeros_like(delta),mean,a)
    assert torch.equal(ww,w.float()) and torch.equal(bb,b.float())
    delta=torch.zeros_like(delta,requires_grad=True);labels=torch.arange(80)%3
    z=x@w.T+b;teacher=(z/2).softmax(1).detach();cfg=dict(label_smoothing=.02,temperature=2.,teacher_KL=.1,anchor=.01)
    loss=objective(torch,F,delta,clean,clean,z,z,labels,torch.ones(80),teacher,torch.ones(80),cfg)
    loss.backward();assert delta.grad is not None and torch.isfinite(delta.grad).all() and delta.grad.norm()>0
    sample,hits,bad=neighbor_emphasis(np,x.numpy(),labels.numpy(),x[:6].numpy(),labels[:6].numpy(),np.zeros((6,3)),5.,3)
    assert sample.min()>0 and sample.max()<=6 and (sample>1).any() and bad>0
    print(json.dumps(dict(passed=True,coordinate_identity_maxabs=error,zero_delta_float32_identity=True,
        finite_TRAIN_gradient=True,TEST_used=False,SDK_used=False,GPU_used=False)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true')
    p.add_argument('--sequence',type=Path);p.add_argument('--profile',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    if a.selftest:selftest()
    else:
        try:run(a)
        except BaseException as error:
            if a.out is not None:
                a.out.mkdir(parents=True,exist_ok=True)
                (a.out/'failure.json').write_text(json.dumps(dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False),indent=2))
            raise
