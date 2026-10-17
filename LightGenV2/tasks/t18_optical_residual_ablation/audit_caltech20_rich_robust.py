"""Independent saved prediction/selection/plan/Adam/EMA audit, no forwards."""
import argparse,math,os
from pathlib import Path
import numpy as np
import torch
import caltech20_rich_robust as m
import audit_caltech20_rich_pair50_alpha as saved

def run(key):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(4);cfg=m.config();case=cfg['cases'][key];depth=case['depth'];out=m.OUT/key;dest=out/'science';assert not dest.exists()
    ex=m.s.read(m.OUT/(key+'.producer_exit.json'));assert ex['actual_exit'] and ex['returncode']==0 and not Path('/proc',str(ex['pid']),'cmdline').exists()
    meta=m.s.read(out/'metadata.json');f=m.s.read(out/'fit_result.json');h=m.s.read(out/'history.json');tr=m.s.read(out/'method_trace.json')
    assert meta['sources']==m.sources() and meta['config']==cfg and meta['frozen_controls']==m.controls() and meta['test_used_for_selection'] and meta['optimizer_reset']
    m.engine.verify_sources(meta['sources'],meta['git_commit']);path,parent=m.parent(depth);assert meta['parent_sha256']==m.s.sha(path) and meta['parent_model_digest']==m.s.digest(parent['model']) and meta['remapped_initial_digest']==m.s.digest(m.remap(parent['model']))
    assert len(h)==cfg['review_epochs']==10 and len(tr)==1970 and f['new_Adam_steps']==1970 and f['test_used_for_selection'] and f['new_test_model_evaluations']==5
    for n,name in [('best','best.pt'),('last','last.pt'),('history','history.json'),('trace','method_trace.json'),('boundary','optimizer_boundary.pt')]:assert f[n+'_sha256']==m.s.sha(out/name)
    best=torch.load(out/'best.pt',map_location='cpu',weights_only=False);last=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
    assert best['sources']==last['sources']==meta['sources'] and last['steps']==1970 and last['tune_epoch']==10 and last['scheduler']['last_epoch']==10
    model=m.RobustMoE(depth);model.load_state_dict(best['model'],strict=True);assert model.contract()==best['contract'] and all(.6001<a<=.95 for a in model.alphas())
    assert len(list(model.parameters()))=={2:12,4:21,6:30}[depth]
    gate=m.s.read(out/'numeric_gate.json');assert gate['passed'] and gate['optimizer_steps']==gate['TEST_images_read']==0 and len(gate['checks'])==3
    for r in gate['checks']:
        assert r['gradient_max_relative']<=.02 and r['gradient_min_cosine']>=.999 and r['probability_maxabs']<=3e-6 and r['capture_maxabs']<=3e-6
        assert len(r['gradient_norms'])==len(list(model.parameters())) and all(np.isfinite(v) and v>0 for v in r['gradient_norms'].values())
        if case['sam']:assert r['sam_second_gradient_max_relative']<=.02
    manifest=m.base.manifest();folder,incumbent,pred=m.incumbent_test(depth);inc=m.engine.metrics(pred['labels'],pred['probabilities'])
    incumbent_val=best['metrics']['val'] if best['tune_epoch']==0 else m.s.read(out/'history.json')[0]['val']
    # Recompute initial VAL from saved accepted parent features, never an optical forward.
    pval=(path.parent/'selected_val.npz') if depth!=6 else path.parent/'selected_val.npz'
    with np.load(pval,allow_pickle=False) as z:iv=saved.metrics(z['labels'],z['probabilities'])
    candidates=[(0,int(round(inc['accuracy']*411)),-iv['balanced_nll'])]
    test_checks={}
    for i,row in enumerate(h,1):
        assert row['epoch']==70+i and row['tune_epoch']==i and row['adam_steps']==197*i
        a,b,_=saved.plans(70+i,cfg,3141);assert row['order_sha256']==a and row['affine_sha256']==b
        rates=[cfg['minimum_lr']+(start-cfg['minimum_lr'])*(1+math.cos(math.pi*(i-1)/10))/2 for start in (cfg['phase_lr'],cfg['electronic_lr'])]
        assert np.allclose(row['lr'],rates,atol=1e-12,rtol=0)
        for role in ('train','val'):saved.history_metric(row[role],manifest['supports'][role],.6)
        if i%2==0:
            with np.load(out/f'test_epoch{i:02d}.npz',allow_pickle=False) as z:
                assert z['ids'].tolist()==manifest['split_ids']['test'];met=saved.metrics(z['labels'],z['probabilities'])
            saved.agree(met,row['test']);candidates.append((i,int(round(met['accuracy']*411)),-row['val']['balanced_nll']));test_checks[str(i)]=m.s.sha(out/f'test_epoch{i:02d}.npz')
    selected=max(candidates,key=lambda v:(v[1],v[2]))[0]
    assert selected==f['selected_tune']==best['tune_epoch']
    if selected:assert m.s.digest(best['model'])==h[selected-1]['ema_digest'] and best['epoch']==70+selected
    else:assert m.s.digest(best['model'])==m.s.digest(m.remap(parent['model']))
    assert m.s.digest(last['model'])==h[-1]['raw_digest'] and m.s.digest(last['ema'])==h[-1]['ema_digest']
    for step,r in enumerate(tr,1):
        epoch=71+(step-1)//197;batch=(step-1)%197
        assert r['epoch']==epoch and r['batch']==batch and r['n']==(5 if batch==196 else 16) and r['seed']==m.seed_for(epoch,batch) and r['sam_passes']==(2 if case['sam'] else 1) and r['adam_steps']==step and np.isfinite([r['loss'],r['clip_norm']]).all()
    # Independently replay final AdamW elementwise from saved pre-clip gradients.
    boundary=torch.load(out/'optimizer_boundary.pt',map_location='cpu',weights_only=False);names=list(boundary['raw_before']);groups=[[n for n in names if n=='raw_router' or n.startswith('phase_sets.')],[n for n in names if n!='raw_router' and not n.startswith('phase_sets.')]]
    norm=torch.stack([v.double().square().sum() for v in boundary['gradients'].values()]).sum().sqrt().item();assert abs(norm-boundary['clip_norm'])<max(1e-6,1e-5*norm)
    clip=min(1.,1/(norm+1e-6));maxerr=0.;maxmoment=0.
    for group,named in zip(boundary['optimizer_before']['param_groups'],groups):
        b1,b2=group['betas']
        for pid,n in zip(group['params'],named):
            st=boundary['optimizer_before']['state'][pid];now=last['optimizer']['state'][pid];g=boundary['gradients'][n].double()*clip
            exp=st['exp_avg'].double()*b1+(1-b1)*g;var=st['exp_avg_sq'].double()*b2+(1-b2)*g.square();t=float(st['step'])+1
            raw=boundary['raw_before'][n].double()*(1-group['lr']*group['weight_decay'])-group['lr']*(exp/(1-b1**t))/(var.sqrt()/math.sqrt(1-b2**t)+group['eps'])
            err=float((raw-last['model'][n].double()).abs().max());maxerr=max(maxerr,err);assert torch.allclose(raw,last['model'][n].double(),atol=2e-6,rtol=2e-5),(n,err)
            for expected,actual in [(exp,now['exp_avg'].double()),(var,now['exp_avg_sq'].double())]:
                maxmoment=max(maxmoment,float((expected-actual).abs().max()));assert torch.allclose(expected,actual,atol=1e-7,rtol=2e-5)
            assert float(now['step'])==1970
            expected_ema=boundary['ema_before'][n].double()*cfg['ema_decay']+last['model'][n].double()*(1-cfg['ema_decay'])
            assert torch.allclose(expected_ema,last['ema'][n].double(),atol=2e-7,rtol=1e-6)
    predictions={}
    for role in ('train','val','test'):
        met,receipt=saved.verify_predictions(out/('selected_'+role+'.npz'),best['model'],role,manifest);saved.agree(met,best['metrics'][role]);saved.agree(met,f['metrics'][role]);predictions[role]=receipt
    assert f['fit_seconds']<=cfg['maximum_seconds']+max(r['epoch_seconds'] for r in h)+60
    dest.mkdir();m.s.save(dest/'audit.json',dict(passed=True,sources=meta['sources'],training_commit=meta['git_commit'],test_used_for_selection=True,independent_generalization=False,selected_tune=selected,predictions=predictions,test_epoch_predictions=test_checks,Adam_maxabs=maxerr,Adam_moment_maxabs=maxmoment,model_forward_calls=0,TEST_images_read=0,frozen_controls=m.controls()),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    try:run(a.key)
    except BaseException as e:
        out=m.OUT/a.key;mark=out/'audit_failure.json'
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
