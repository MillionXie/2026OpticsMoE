"""Independent saved plans, alpha bounds, freeze periods, Adam/EMA and selection."""
import argparse,math,os
from pathlib import Path
import numpy as np
import torch
import caltech20_rich_alpha40_rebalance as m
import audit_caltech20_rich_pair50_alpha as saved
def run(key):
    assert os.environ['CUDA_VISIBLE_DEVICES']=='';torch.set_num_threads(4);cfg=m.config();case=cfg['cases'][key];depth=case['depth'];out=m.OUT/key;dest=out/'science';assert not dest.exists()
    ex=m.s.read(m.OUT/(key+'.producer_exit.json'));assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists()
    meta=m.s.read(out/'metadata.json');fit=m.s.read(out/'fit_result.json');h=m.s.read(out/'history.json');trace=m.s.read(out/'method_trace.json');m.engine.verify_sources(meta['sources'],meta['git_commit'])
    assert meta['sources']==m.sources() and meta['config']==cfg and meta['frozen_controls']==m.old.controls() and meta['case']==case
    q,pt,pred=m.incumbent(depth);assert meta['parent']==q and meta['init_digest']==m.s.digest(m.initialize(depth,case['fresh']).state_dict())
    assert len(h)==case['epochs'] and len(trace)==197*len(h)==fit['new_Adam_updates'] and fit['actual_new_epochs']==case['epochs']
    for name,file in [('best','best.pt'),('last','last.pt'),('history','history.json'),('trace','method_trace.json'),('boundary','optimizer_boundary.pt')]:assert fit[name+'_sha256']==m.s.sha(out/file)
    best=torch.load(out/'best.pt',map_location='cpu',weights_only=False);last=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
    assert best['sources']==last['sources']==meta['sources'] and last['steps']==len(trace)
    model=m.Alpha40MoE(depth);model.load_state_dict(best['model']);assert best['contract']==model.contract() and all(.4001<a<.95 for a in model.alphas())
    gate=m.s.read(out/'numeric_gate.json');assert gate['passed'] and gate['optimizer_steps']==gate['TEST_images_read']==0
    for g in gate['checks']:
        assert g['gradient_max_relative']<=.02 and g['gradient_min_cosine']>=.999 and g['probability_maxabs']<=3e-6 and g['capture_maxabs']<=3e-6 and len(g['gradient_norms'])=={2:12,4:21,6:30}[depth]
        assert all(np.isfinite(a) and a>0 for a in g['gradient_norms'].values())
    candidates=[(0,int(round(m.engine.metrics(pred['test']['labels'],pred['test']['probabilities'])['accuracy']*411)),-m.engine.metrics(pred['val']['labels'],pred['val']['probabilities'])['balanced_nll'])]
    test_checks={};manifest=m.base.manifest()
    for tune,row in enumerate(h,1):
        epoch=tune if case['fresh'] else q['joint_budget_epochs']+tune;a,b,_=saved.plans(epoch,cfg,3141)
        assert row['epoch']==epoch and row['tune_epoch']==tune and row['adam_updates']==197*tune and row['electronic_only']==(tune<=case['electronic_warm_epochs'])
        assert row['order_sha256']==a and row['affine_sha256']==b and np.allclose(row['lr'],m.rates(case,tune),atol=1e-12,rtol=0)
        if row['electronic_only']:assert row['phase_alpha_digest']==meta['frozen_phase_alpha_initial_digest']
        for role in ('train','val'):saved.history_metric(row[role],manifest['supports'][role],.4)
        if tune%cfg['test_interval']==0:
            p=m.load_predictions(out/f'test_epoch{tune:02d}.npz');assert p['ids'].tolist()==manifest['split_ids']['test'];actual=saved.metrics(p['labels'],p['probabilities']);saved.agree(actual,row['test']);candidates.append((tune,int(round(actual['accuracy']*411)),-row['val']['balanced_nll']));test_checks[str(tune)]=m.s.sha(out/f'test_epoch{tune:02d}.npz')
    selected=max(candidates,key=lambda a:(a[1],a[2]))[0];assert selected==fit['selected_tune']==best['tune_epoch']
    if selected:assert m.s.digest(best['model'])==h[selected-1]['ema_digest']
    else:assert m.s.digest(best['model'])==m.s.digest(m.remap(pt['model'],q['alpha_low']))
    assert m.s.digest(last['model'])==h[-1]['raw_digest'] and m.s.digest(last['ema'])==h[-1]['ema_digest']
    for step,row in enumerate(trace,1):
        tune=1+(step-1)//197;batch=(step-1)%197;epoch=tune if case['fresh'] else q['joint_budget_epochs']+tune
        assert row['tune']==tune and row['epoch']==epoch and row['batch']==batch and row['n']==(5 if batch==196 else 16) and row['seed']==m.seed_for(epoch,batch) and row['adam_updates']==step and row['electronic_only']==(tune<=case['electronic_warm_epochs'])
        assert np.isfinite([row['loss'],row['clip_norm']]).all()
        # Reconstruct the independent RNG crop geometry, no model forward/image use.
        rg=np.random.default_rng(np.random.SeedSequence([17,epoch,batch,20261018,40,1]));n=row['n'];area=rg.uniform(cfg['crop_area_min'],1,n);ratio=rg.uniform(.9,1.1,n);w=np.minimum(1,np.sqrt(area*ratio));hh=np.minimum(1,np.sqrt(area/ratio));crop=np.zeros((n,2,3),dtype=np.float32);crop[:,0,0]=w;crop[:,1,1]=hh;crop[:,0,2]=rg.uniform(-1,1,n)*(1-w);crop[:,1,2]=rg.uniform(-1,1,n)*(1-hh);assert row['crop_sha256']==m.s.digest(crop)
    boundary=torch.load(out/'optimizer_boundary.pt',map_location='cpu',weights_only=False);norm=torch.stack([a.double().square().sum() for a in boundary['gradients'].values()]).sum().sqrt().item();assert abs(norm-boundary['clip_norm'])<max(1e-6,1e-5*norm);clip=min(1.,1/(norm+1e-6));maximum=0
    for group,names in zip(boundary['optimizer_before']['param_groups'],boundary['groups']):
        b1,b2=group['betas']
        for pid,n in zip(group['params'],names):
            before=boundary['optimizer_before']['state'][pid];after=last['optimizer']['state'][pid];g=boundary['gradients'][n].double()*clip;t=float(before['step'])+1
            expected_step=197*(case['epochs']-case['electronic_warm_epochs']) if n=='raw_router' or n.startswith('phase_sets.') or n.startswith('fusion_logits.') else 197*case['epochs'];assert float(after['step'])==expected_step
            exp=before['exp_avg'].double()*b1+(1-b1)*g;var=before['exp_avg_sq'].double()*b2+(1-b2)*g.square();raw=boundary['raw_before'][n].double()*(1-group['lr']*group['weight_decay'])-group['lr']*(exp/(1-b1**t))/(var.sqrt()/math.sqrt(1-b2**t)+group['eps'])
            maximum=max(maximum,float((raw-last['model'][n].double()).abs().max()));assert torch.allclose(raw,last['model'][n].double(),atol=2e-6,rtol=2e-5)
            assert torch.allclose(exp,after['exp_avg'].double(),atol=1e-7,rtol=2e-5) and torch.allclose(var,after['exp_avg_sq'].double(),atol=1e-7,rtol=2e-5)
            e=boundary['ema_before'][n].double()*cfg['ema_decay']+last['model'][n].double()*(1-cfg['ema_decay']);assert torch.allclose(e,last['ema'][n].double(),atol=2e-7,rtol=1e-6)
    checks={}
    for role in ('train','val','test'):
        met,rec=saved.verify_predictions(out/('selected_'+role+'.npz'),best['model'],role,manifest);saved.agree(met,best['metrics'][role]);saved.agree(met,fit['metrics'][role]);checks[role]=rec
    assert fit['new_test_model_evaluations']==case['epochs']//cfg['test_interval'] and fit['TEST_used_for_selection']
    dest.mkdir();m.s.save(dest/'audit.json',dict(passed=True,sources=meta['sources'],training_commit=meta['git_commit'],selected_tune=selected,predictions=checks,test_epoch_predictions=test_checks,Adam_maxabs=maximum,phase_and_alpha_frozen_first_epochs=case['electronic_warm_epochs'],model_forward_calls=0,TEST_images_read=0,TEST_used_for_selection=True,independent_generalization=False,frozen_controls=m.old.controls()),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    try:run(a.key)
    except BaseException as e:
        out=m.OUT/a.key;mark=out/'audit_failure.json'
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
