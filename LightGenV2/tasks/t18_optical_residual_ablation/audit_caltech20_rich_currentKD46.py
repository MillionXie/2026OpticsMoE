"""Original local-epoch scientific chain plus independently reconstructed KD."""
import types,numpy as np,hashlib,torch
import caltech20_rich_currentKD46 as m
import audit_caltech20_rich_L2_retained_soup_finetune as previous
ns=dict(previous.kernel.__globals__);ns['m']=m;kernel=types.FunctionType(previous.kernel.__code__,ns,'same_local_saved_Adam_and_selection_math',previous.kernel.__defaults__,previous.kernel.__closure__)
ns=dict(previous.inherited.__globals__);ns.update(m=m,kernel=kernel);inherited=types.FunctionType(previous.inherited.__code__,ns,'same_semantic_accounting',previous.inherited.__defaults__,previous.inherited.__closure__)
ns=dict(previous.base_run.__globals__);ns.update(m=m,inherited=inherited);base_run=types.FunctionType(previous.base_run.__code__,ns,'same_loss_accounting',previous.base_run.__defaults__,previous.base_run.__closure__)
ns=dict(previous.prototype_run.__globals__);ns.update(m=m,base_run=base_run);prototype_run=types.FunctionType(previous.prototype_run.__code__,ns,'same_TRAIN_prototype_reconstruction',previous.prototype_run.__defaults__,previous.prototype_run.__closure__)
ns=dict(previous.phase_run.__globals__);ns.update(m=m,prototype_run=prototype_run);phase_run=types.FunctionType(previous.phase_run.__code__,ns,'same_phasecode_reconstruction',previous.phase_run.__defaults__,previous.phase_run.__closure__)
ns=dict(previous.retained_run.__globals__);ns.update(m=m,phase_run=phase_run);retained_run=types.FunctionType(previous.retained_run.__code__,ns,'same_complete_retained_science',previous.retained_run.__defaults__,previous.retained_run.__closure__)
def run(key):
    m.select(key);retained_run(key);f=m.OUT/key;cfg=m.config();d=cfg['cases'][key]['depth'];pt=torch.load(m.teacher_source.OUT/'selected_head.pt',map_location='cpu',weights_only=False)
    with np.load(m.teacher_source.OUT/'train_features.npz',allow_pickle=False) as z:x,y,ids=[z[k].copy() for k in ('features','labels','ids')]
    assert ids.tolist()==m.base.manifest()['split_ids']['train'];x=((x-pt['mean'])/pt['std']).astype(np.float32).astype(float);logits=x@pt['model']['weight'].double().numpy().T+pt['model']['bias'].double().numpy();z=logits/cfg['teacher_temperature_by_depth'][str(d)];target=np.exp(z-z.max(1,keepdims=True));target=(target/target.sum(1,keepdims=True)).astype(np.float32);raw=np.exp(logits-logits.max(1,keepdims=True));raw/=raw.sum(1,keepdims=True);valid=((logits.argmax(1)==y)&(raw.max(1)>=.5)).astype(np.float32);trace=m.s.read(f/'method_trace.json')
    for row in trace:
        epoch=row['epoch'];batch=row['batch'];order,_=m.engine.epoch_plan(epoch,3141,cfg);ix=order[batch*16:batch*16+row['n']];rng=np.random.default_rng(np.random.SeedSequence([17,epoch,batch,20261013,4]));lam=float(rng.beta(cfg['mixup_alpha'],cfg['mixup_alpha']));lam=max(lam,1-lam);perm=rng.permutation(len(ix));mass=lam*valid[ix]+(1-lam)*valid[ix][perm];q=(lam*valid[ix,None]*target[ix]+(1-lam)*valid[ix][perm,None]*target[ix][perm])/mass[:,None].clip(1e-12);v=row['view_info'];assert v['teacher_temperature']==cfg['teacher_temperature_by_depth'][str(d)] and v['teacher_q_raw_SHA256']==hashlib.sha256(q.astype('<f4').tobytes()).hexdigest() and v['teacher_mass_raw_SHA256']==hashlib.sha256(mass.astype('<f4').tobytes()).hexdigest()
    fit=m.s.read(f/'fit_result.json');assert fit['local_added_epochs']==5 and fit['not_fresh_initialization'];proof=m.s.read(f/'science/audit.json');proof.update(independent_current_teacher_targets_all_macros=True,teacher_temperature=cfg['teacher_temperature_by_depth'][str(d)],L2_untouched=True);m.s.save(f/'science/audit.json',proof)
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    try:run(a.key)
    except BaseException as e:
        p=m.OUT/a.key/'audit_failure.json'
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
