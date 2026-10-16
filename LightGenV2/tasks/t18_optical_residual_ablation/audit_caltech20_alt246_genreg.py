"""Own continuation saved audit; reuses only immutable independent math helpers."""
import argparse,hashlib,os
from pathlib import Path
import numpy as np
import torch
import caltech20_alt246_genreg as study
from caltech20_alt246_genreg_model import RegularizedMoE
from audit_caltech20_alt246_biasfree import plans,verify_predictions,agree,history_metric

def moment_digest(state):
    import hashlib
    h=hashlib.sha256()
    values={str(i)+':'+n:v for i,record in state['state'].items() for n,v in record.items() if isinstance(v,torch.Tensor)}
    for n,v in sorted(values.items()):h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def masks(epoch,cfg,n=1372):
    flips=[];masks=[]
    for batch,start in enumerate(range(0,n,16)):
        size=min(16,n-start);flags=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,batch,20261016,1])).random(size)<.5
        keep=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,batch,20261016,2])).random((size,256))>=cfg['feature_dropout'];drop=keep.astype(np.float32)/(1-cfg['feature_dropout'])
        flips.append(hashlib.sha256(flags.tobytes()).hexdigest());masks.append(hashlib.sha256(drop.tobytes()).hexdigest())
    return hashlib.sha256(''.join(flips).encode()).hexdigest(),hashlib.sha256(''.join(masks).encode()).hexdigest()

def process_exited(pid):
    p=Path('/proc',str(pid),'cmdline');return not p.exists() or study.RUN.encode() not in p.read_bytes()

def audit(base,out,phase):
    s=study.s;assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not out.exists();study.baseline();meta=s.read(base/'metadata.json');assert meta['sources']==study.sources();study.engine.verify_sources(meta['sources'],meta['git_commit'])
    state=s.read(base/'status.json');assert not state['active'] and not state['pending'] and not state['failures'];manifest=study.b.old.manifest(Path(study.b.config()['data']));checks={};torch.set_num_threads(4)
    if phase=='fit':
        for key in meta['cases']:
            folder=base/key;ex=s.read(folder.with_name(key+'.fit_exit.json'));assert ex['returncode']==0 and ex['actual_exit'] and process_exited(ex['pid']);m=s.read(folder/'metadata.json');depth=study.CASES[key][0];cfg=study.config(depth)
            assert m['sources']==meta['sources'] and m['config']==cfg and m['rho']==.3 and m['baseline_source']==study.b.sources();study.engine.verify_sources(m['sources'],m['git_commit'])
            parent=Path(meta['parents'][key]);pLast=parent/key/'last.pt';last_parent=torch.load(pLast,map_location='cpu',weights_only=False)
            assert m['parent_last_sha256']==s.sha(pLast) and m['parent_raw_digest']==s.digest(last_parent['model']) and m['parent_ema_digest']==s.digest(last_parent['ema']) and m['initial_optimizer_moment_digest']==moment_digest(last_parent['optimizer'])
            expected_prior=[] if parent==study.BASE else s.read(parent/key/'history_extra.json');h=s.read(folder/'history_extra.json');assert h[:-10]==expected_prior
            f=s.read(folder/'fit_result.json');assert f['extra_epochs']==len(h) and f['actual_epochs']==100+len(h) and f['steps']==8600+86*len(h)
            assert f['last_sha256']==s.sha(folder/'last.pt') and f['history_sha256']==s.sha(folder/'history_extra.json') and f['best_sha256']==s.sha(f['best_path'])
            last=torch.load(folder/'last.pt',map_location='cpu',weights_only=False);best=torch.load(f['best_path'],map_location='cpu',weights_only=False);model=RegularizedMoE(depth);model.load_state_dict(last['model'],strict=True);model.load_state_dict(best['model'],strict=True)
            assert last['sources']==meta['sources'] and last['config']==cfg and last['epoch']==f['actual_epochs'] and last['steps']==f['steps']
            assert len(last['optimizer']['state'])==len(list(model.parameters())) and all(float(v['step'])==f['steps'] for v in last['optimizer']['state'].values())
            assert [p['weight_decay'] for p in last['optimizer']['param_groups']]==[0,cfg['electronic_weight_decay']]
            for n,v in last['previous_final_ema'].items():
                expected=v.numpy().astype(np.float64)*cfg['ema_decay']+last['model'][n].numpy().astype(np.float64)*(1-cfg['ema_decay']);assert np.allclose(expected,last['ema'][n].numpy(),atol=2e-7,rtol=1e-6)
            baseline_h=s.read(study.BASE/key/'history.json');initial_best=min(baseline_h,key=lambda row:row['val']['balanced_nll']);lr=cfg['lr'];scheduler_best=initial_best['val']['balanced_nll'];bad=0
            for number,row in enumerate(h,101):
                assert row['epoch']==number and row['adam_steps']==86*number
                assert tuple(row[k] for k in ('order_sha256','affine_sha256','mix_sha256'))==plans(number,cfg,1372)
                assert (row['flip_sha256'],row['mask_sha256'])==masks(number,cfg)
                assert np.allclose(row['lr'],[lr,lr],atol=1e-12,rtol=0)
                for role in ('train','val'):history_metric(row[role],manifest['supports'][role])
                if row['val']['balanced_nll']<scheduler_best-.002:scheduler_best=row['val']['balanced_nll'];bad=0
                else:bad+=1
                if bad>3:lr=max(lr*.5,2e-5);bad=0
            assert last['scheduler']['last_epoch']==len(h) and last['scheduler']['num_bad_epochs']==bad and abs(last['scheduler']['best']-scheduler_best)<1e-10
            assert np.allclose([p['lr'] for p in last['optimizer']['param_groups']],[lr,lr],atol=1e-12,rtol=0)
            selected=min(baseline_h+h,key=lambda row:row['val']['balanced_nll']);assert f['selected_epoch']==selected['epoch']==best['epoch'] and s.digest(best['model'])==selected['ema_digest']
            assert s.digest(last['model'])==h[-1]['raw_digest'] and s.digest(last['ema'])==h[-1]['ema_digest']
            for role,path in (('train',f['selected_train_path']),('val',f['selected_val_path'])):
                metrics,receipt=verify_predictions(Path(path),best['model'],role,manifest);agree(metrics,f['metrics'][role]);agree(metrics,selected[role]);checks[key+'_'+role]=receipt
            gate=s.read(folder/'numeric_gate.json');assert gate['passed'] and gate['initial_digest']==m['parent_raw_digest'] and gate['optimizer_updates']==0
            assert {q['samples'] for q in gate['checks']}=={'all20','macro16','tail12'}
            assert len(gate['checks'])==6 and {q['method'] for q in gate['checks']}=={'ordinary','dropout'}
            for check in gate['checks']:assert set(check['gradient_norms'])==set(dict(model.named_parameters())) and all(np.isfinite(v) and v>0 for v in check['gradient_norms'].values())
            assert best['contract']==model.contract() and all(.7<x<=.95 for x in model.alphas());del model,last,best,last_parent
    else:
        receipt=s.read(base/'audit_completed.json');assert receipt['actual_audit_returncode']==0 and receipt['audit_sha256']==s.sha(base/'science/audit.json')
        for key in meta['cases']:
            folder=base/'once_tests'/key;ex=s.read(folder.with_name(key+'.test_exit.json'));assert ex['returncode']==0 and ex['actual_exit'] and process_exited(ex['pid']);f=s.read(base/key/'fit_result.json');result=s.read(folder/'result.json');best=torch.load(f['best_path'],map_location='cpu',weights_only=False)
            assert result['model_digest']==s.digest(best['model']) and result['selected_epoch']==f['selected_epoch'] and result['test_evaluations']==1
            path=Path(result['predictions_path']);assert result['predictions_sha256']==s.sha(path)
            metrics,checks[key]=verify_predictions(path,best['model'],'test',manifest);agree(metrics,result['metrics'])
    out.mkdir();s.save(out/'audit.json',dict(passed=True,phase=phase,sources=meta['sources'],actual_training_commit=meta['git_commit'],checks=checks,model_forward_calls=0,test_images_read=0,cpu_models_for_shape= len(meta['cases']) if phase=='fit' else 0),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--phase',choices=['fit','test'],required=True);a=p.parse_args()
    try:audit(a.base,a.out,a.phase)
    except BaseException as e:
        path=a.base/(a.phase+'_audit_failure.json')
        if not path.exists():study.s.save(path,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
