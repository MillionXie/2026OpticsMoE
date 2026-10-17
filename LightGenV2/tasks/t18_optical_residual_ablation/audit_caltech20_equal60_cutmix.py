"""Independent saved objective/restoration/EMA/plan/prediction audit,0forward."""
import argparse,hashlib,math,os
from pathlib import Path
import numpy as np
import torch
import caltech20_equal60_cutmix as m
import audit_caltech20_equal60_logrepair as independent
def plans(epoch,cfg,n):
    order,affine,_=independent.plans(epoch,cfg,n);hashes=[]
    for batch,start in enumerate(range(0,n,16)):
        count=min(16,n-start);rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,batch,20261017,6]));applied=bool(rng.random()<cfg['cutmix_probability']);fraction=float(rng.beta(cfg['cutmix_alpha'],cfg['cutmix_alpha']));perm=rng.permutation(count);width=int(112*np.sqrt(1-fraction));cx,cy=rng.integers(0,112,size=2);x1,x2=max(int(cx)-width//2,0),min(int(cx)+width//2,112);y1,y2=max(int(cy)-width//2,0),min(int(cy)+width//2,112)
        if not applied:x1=x2=y1=y2=0
        lam=1-(x2-x1)*(y2-y1)/(112*112);packed=np.r_[float(applied),lam,(x1,y1,x2,y2),perm.astype(float)].astype('<f8');hashes.append(m.s.digest(packed))
    return order,affine,hashlib.sha256(''.join(hashes).encode()).hexdigest()

def audit(base,out,phase):
    assert base==m.OUT and not out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='';state=m.s.read(base/'status.json');assert not state['active'] and not state['pending'] and not state['failures'];controls=m.admit();cfg=m.config();meta=m.s.read(base/'metadata.json');assert meta['config']==cfg and meta['sources']==m.sources() and meta['frozen_controls']==controls;m.engine.verify_sources(meta['sources'],meta['git_commit']);manifest=m.base.dataset.manifest();checks={};torch.set_num_threads(4)
    if phase=='fit':
        for key,(depth,rho) in m.CASES.items():
            folder=base/key;ex=m.s.read(folder.with_name(key+'.fit_exit.json'));assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists();metadata=m.s.read(folder/'metadata.json');assert metadata['sources']==meta['sources'] and metadata['config']==cfg and metadata['frozen_controls']==controls
            parent=m.BASE/key;old=torch.load(parent/'last.pt',map_location='cpu',weights_only=False);incumbent=torch.load(parent/'best.pt',map_location='cpu',weights_only=False);last=torch.load(folder/'last.pt',map_location='cpu',weights_only=False);best=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);h=m.s.read(folder/'history.json');f=m.s.read(folder/'fit_result.json')
            assert len(h)==f['extra_epochs']==10 and f['actual_joint_epochs']==80 and f['steps']==last['steps']==15760 and f['stop_reason']=='strategy_review_after10'
            assert metadata['parent_last_sha256']==m.s.sha(parent/'last.pt') and metadata['parent_best_sha256']==m.s.sha(parent/'best.pt') and metadata['parent_history_sha256']==m.s.sha(parent/'history.json')
            assert metadata['parent_raw_digest']==m.s.digest(old['model']) and metadata['parent_ema_digest']==m.s.digest(old['ema']) and metadata['restored_optimizer_moment_digest']==m.moments(old['optimizer'])
            for field,file in [('best_sha256','best.pt'),('last_sha256','last.pt'),('history_sha256','history.json')]:assert f[field]==m.s.sha(folder/file)
            assert best['sources']==last['sources']==meta['sources'] and best['config']==last['config']==cfg and last['epoch']==80 and last['scheduler']['last_epoch']==10
            assert all(float(q['step'])==15760 for q in last['optimizer']['state'].values()) and [q['weight_decay'] for q in last['optimizer']['param_groups']]==[0,cfg['electronic_weight_decay']]
            m.engine.seed(17);model=m.base.RegularizedMoE(depth,rho);assert m.s.digest(model.state_dict())==old['initial_digest']==metadata['initial_digest'];model.load_state_dict(last['model'],strict=True);model.load_state_dict(best['model'],strict=True);assert model.contract()==best['contract']
            counts=np.asarray(manifest['supports']['train'],np.float64);expected_weights=np.sqrt(counts.sum()/(20*counts));expected_weights*=counts.sum()/np.dot(counts,expected_weights);assert abs(np.dot(counts,expected_weights)/counts.sum()-1)<1e-12 and np.allclose(metadata['class_weights'],expected_weights,atol=1e-7,rtol=1e-7)
            gate=m.s.read(folder/'numeric_gate.json');assert gate['passed'] and gate['parent_raw_digest']==m.s.digest(old['model']) and np.allclose(gate['class_weights'],expected_weights,atol=1e-7,rtol=1e-7) and len(gate['checks'])==6 and gate['forced_cutmix']
            for check in gate['checks']:
                norms=check['gradient_norms'];assert set(norms)==set(dict(model.named_parameters())) and all(np.isfinite(q) and q>0 for q in norms.values()) and check['gradient_max_relative_l2']<=.02 and check['gradient_min_cosine']>=.999 and check['probability_maxabs']<=2e-6 and check['capture_maxabs']<=2e-6
            for i,row in enumerate(h,1):
                assert row['epoch']==70+i and row['tune_epoch']==i and row['adam_steps']==13790+197*i
                assert tuple(row[k] for k in ('order_sha256','affine_sha256','mix_sha256'))==plans(70+i,cfg,3141) and tuple(row[k] for k in ('flip_sha256','mask_sha256'))==independent.mask_plans(70+i,cfg,3141)
                lr=[cfg['minimum_tune_lr']+(start-cfg['minimum_tune_lr'])*(1+math.cos(math.pi*(i-1)/10))/2 for start in (cfg['phase_lr'],cfg['electronic_lr'])];assert np.allclose(lr,row['lr'],atol=1e-12,rtol=0)
                for role in ('train','val'):independent.history_metric(row[role],manifest['supports'][role])
            candidates=[dict(tune_epoch=0,epoch=incumbent['epoch'],train=incumbent['metrics']['train'],val=incumbent['metrics']['val'],ema_digest=m.s.digest(incumbent['model']))]+h;selected=min(candidates,key=lambda q:q['val']['nll']);assert best['tune_epoch']==f['selected_tune']==selected['tune_epoch'] and best['epoch']==f['selected_epoch']==selected['epoch'] and m.s.digest(best['model'])==selected['ema_digest']
            assert m.s.digest(last['model'])==h[-1]['raw_digest'] and m.s.digest(last['ema'])==h[-1]['ema_digest']
            assert set(last['previous_final_ema'])==set(last['model'])==set(last['ema'])
            for name,previous in last['previous_final_ema'].items():
                expected=previous.numpy().astype(np.float64)*cfg['ema_decay']+last['model'][name].numpy().astype(np.float64)*(1-cfg['ema_decay']);assert np.allclose(expected,last['ema'][name].numpy(),atol=2e-7,rtol=1e-6)
            assert f['fit_seconds']<=cfg['maximum_seconds']+max(q['epoch_seconds'] for q in h)+60
            for role in ('train','val'):
                actual,receipt=independent.verify_predictions(folder/('selected_'+role+'.npz'),best['model'],role,manifest);independent.agree(actual,f['metrics'][role]);independent.agree(actual,best['metrics'][role]);independent.agree(actual,selected[role]);checks[key+'_'+role]=receipt
            del old,incumbent,last,best,model
    else:
        science=m.s.read(base/'audit_completed.json');assert science['actual_audit_returncode']==0 and science['audit_sha256']==m.s.sha(base/'science/audit.json')
        for key in m.CASES:
            folder=base/'once_tests'/key;ex=m.s.read(folder.with_name(key+'.test_exit.json'));assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists();best=torch.load(base/key/'best.pt',map_location='cpu',weights_only=False);started=m.s.read(folder/'started.json');result=m.s.read(folder/'result.json');assert started['sources']==meta['sources'] and started['selected_sha256']==m.s.sha(base/key/'best.pt') and started['model_digest']==m.s.digest(best['model']) and result['selected_epoch']==started['selected_epoch']==best['epoch'] and result['test_used_for_selection'] is False
            if result['test_reused']:
                assert result['test_evaluations']==started['test_evaluations']==0 and m.s.read(m.BASE/'once_tests'/key/'started.json')['model_digest']==started['model_digest'] and started['reused_result_sha256']==m.s.sha(m.BASE/'once_tests'/key/'result.json') and m.s.sha(folder/'predictions.npz')==m.s.sha(m.BASE/'once_tests'/key/'predictions.npz')
            else:assert result['test_evaluations']==started['test_evaluations']==1
            actual,checks[key]=independent.verify_predictions(folder/'predictions.npz',best['model'],'test',manifest);independent.agree(actual,result['metrics'])
    out.mkdir();m.s.save(out/'audit.json',dict(passed=True,phase=phase,sources=meta['sources'],training_commit=meta['git_commit'],checks=checks,models_for_shape_seed=3 if phase=='fit' else 0,model_forward_calls=0,TEST_images_read=0,frozen_controls=controls),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--phase',choices=['fit','test'],default='fit');a=p.parse_args()
    try:audit(a.base,a.out,a.phase)
    except BaseException as e:
        mark=a.base/(a.phase+'_audit_failure.json')
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
