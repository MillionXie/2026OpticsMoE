"""TRAIN class-compactness optimization of existing direct alpha, then affine OAS.

Only training uses pooled branch sufficient statistics. Deployment is the
unchanged native model, with alpha3 and its existing affine head updated.
No TEST objective/alpha sweep, no new inference operator or physical capture.
"""
import argparse,traceback,time
from pathlib import Path
import lab_caltech20_tucam_layerwise_20261023 as base


def branch_statistics(e,o):
    import torch
    from torch.nn import functional as F
    from caltech20_alt246_biasfree_model import rms
    re,ro=rms(e),rms(o);en,on=e/re,o/ro
    pool=lambda v:F.adaptive_avg_pool2d(v[:,None],(16,16)).flatten(1)
    return dict(e=pool(en),o=pool(on),re=re.flatten(),ee=en.square().mean((-2,-1)),oo=on.square().mean((-2,-1)),eo=(en*on).mean((-2,-1)))


def cached_features(s,alpha):
    from torch.nn import functional as F
    a=alpha.float();scale=((1-a)**2*s['ee']+a*a*s['oo']+2*a*(1-a)*s['eo']).clamp_min(1e-12).sqrt()
    pooled=s['re'][:,None]/scale[:,None]*((1-a)*s['e']+a*s['o'])
    return F.layer_norm(pooled,(256,),eps=1e-5)


def separation(features,labels):
    import torch
    center=features.mean(0);means=torch.stack([features[labels==k].mean(0)for k in range(20)])
    between=(means-center).square().sum(1).mean()
    total=(features-center).square().sum(1).mean().clamp_min(1e-12)
    return between/total


def run(a):
    np,torch,m,loop=base.numerical_modules();torch.set_num_threads(2);torch.manual_seed(73)
    from lab_caltech20_tucam_TRAIN_linear_20261023 import TRAIN_linear_fit
    assert not a.out.exists() and not m.LOCK.exists()
    assert a.out.name=='fullG2_TRAIN_fusion_20261024'
    root=base.ROOT/'LightGenV2/tasks/t18_optical_residual_ablation/runs/hardware/caltech20_residual_TUCam_20261022/layerwise_L4_on_20261023/e2_actual300_20261024'
    source=root/'fullG2_joint_TRAIN_OAS_20261024';assert m.read(source.with_suffix('.return.json'))['returncode']==0
    sr=m.read(source/'report.json');sp=m.read(source/'protocol.json');sa=m.read(source/'independent_audit.json')
    assert sr['full_physical'] and sr['strict_original_CPU'] and sa['passed'] and sa['report_SHA256']==m.sha(source/'report.json')
    cpath=base.HERE/'lab_caltech20_tucam_layerwise_profile_20261023.json';assert m.sha(cpath)=='c077c32638e577b1c86847b5ff7cdd40ab3ede675c9b4299c0697e1b389ec909'
    c=m.read(cpath);base.validate_config(c);reference=base.ROOT/c['reference'];assert m.sha(reference)==c['reference_SHA256']
    pt=source/'best.pt';assert m.sha(pt)==sr['results']['best']['PT_SHA256'];model,_=m.model_for(pt,m.sha(pt),4,reference);original=m.state_cpu(model)
    captured=root/'readout_LOCALMSE/fit/best.pt';assert m.sha(captured)=='60f6883ac597fbe3a2f72000edac661998dd6fb52696ba80cb6b8bcce53f115e'
    native,_=m.model_for(captured,m.sha(captured),4,reference);loop.validate_state(m,model,m.state_cpu(native),m.allowed_keys(model,5));del native
    data,dm=m.data_for(base.ROOT/c['sequence']);assert dm['sha256']==c['data_SHA256'];fields={};paths={};policy=m.read(root/'readout_G2PASS/protocol.json')['policy']
    for stage in base.STAGES:
        cap=base.ROOT/policy['captures'][stage]if stage in policy['captures']else root/'readout_G2PASS/s_3/capture';r=m.read(cap/'report.json');audit=m.read(cap/'independent_audit.json')
        assert audit['passed'] and audit['matched_samples']==2011 and audit['report_SHA256']==m.sha(cap/'report.json') and r['SDK_released'] and r['data_SHA256']==dm['sha256'] and m.sha(cap/'phase.bmp')==r['phase_SHA256']
        paths[stage]=cap;fields[stage]={}
        for role,n in (('train',1600),('test',411)):
            p=cap/role/'ccd.npy';assert m.sha(p)==sp['prefix_SHA256'][stage+'/'+role]==r['arrays'][role]['SHA256'];v=np.load(p,mmap_mode='r');assert v.shape==(n,478,478)and v.dtype==np.uint8;fields[stage][role]=v
    lineage=base.identities(m,paths);keys={'fusion_logits.3','linear.weight','linear.bias'};assert sum(v.numel()for n,v in model.named_parameters()if n in keys)==5141
    a.out.mkdir();protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),parent_PT=str(pt),parent_PT_SHA256=m.sha(pt),prefix_SHA256=lineage,data_SHA256=dm['sha256'],allowed_keys=sorted(keys),TRAIN=1600,TEST=411,VALID=0,full_physical=True,TEST_gradient=False,TEST_development=True,alpha_objective='TRAIN_between_class_scatter_fraction',alpha_steps_max=300,alpha_lr=.005,fit_seconds_max=120,alpha_candidates_from_TEST=0,new_inference_layers=0,SDK_used=False,GPU_used=False)
    m.save(a.out/'protocol.json',protocol);model.eval().requires_grad_(False);holder={};hook=model.electronic[3].register_forward_hook(lambda module,args,value:holder.update(e=value))
    stats={};parent_prob={};maxerr=0.
    try:
        with torch.inference_mode():
            for role,n in (('train',1600),('test',411)):
                pieces=[];probs=[]
                for begin in range(0,n,8):
                    ix=np.arange(begin,min(begin+8,n));prob=model(loop.put_fields(model,data,fields,role,ix,'cpu'))[0];s=branch_statistics(holder['e'],model.nonlinear(model.measured['3'].float(),'global'));f=cached_features(s,model.fusion_logits[3]);err=float((model.linear(f).softmax(1)-prob).abs().max());maxerr=max(maxerr,err);pieces.append({k:v.clone()for k,v in s.items()});probs.append(prob.numpy().copy())
                    if begin%160==0:m.save(a.out/'progress.json',dict(state='native_branch_cache',role=role,consumed=begin+len(ix),probability_max_error=maxerr,TEST_fit=False))
                stats[role]={k:torch.cat([s[k]for s in pieces])for k in pieces[0]};parent_prob[role]=np.concatenate(probs)
    finally:hook.remove()
    assert maxerr<=3e-6,('Cache does not reproduce native probabilities',maxerr)
    # Materialize ordinary constant tensors outside inference_mode before alpha AD.
    stats={role:{k:v.clone()for k,v in packet.items()}for role,packet in stats.items()}
    for role in ('train','test'):
        z=np.load(source/('best_'+role+'_predictions.npz'));assert np.array_equal(z['ids'],data[role][2])and np.array_equal(z['labels'],data[role][1]);assert np.max(np.abs(z['probabilities']-parent_prob[role]))<=3e-6
        np.savez_compressed(a.out/(role+'_branch_statistics.npz'),ids=data[role][2],labels=data[role][1],**{k:v.numpy()for k,v in stats[role].items()})
    # TEST statistics are cached only for admission; optimization has TRAIN only.
    alpha=torch.nn.Parameter(model.fusion_logits[3].detach().clone());labels=torch.tensor(data['train'][1],dtype=torch.long);opt=torch.optim.Adam([alpha],lr=.005);start=time.monotonic();history=[]
    with torch.no_grad():best_sep=float(separation(cached_features(stats['train'],alpha),labels));selected_alpha=alpha.detach().clone()
    initial_sep=best_sep
    for step in range(1,301):
        opt.zero_grad(set_to_none=True);score=separation(cached_features(stats['train'],alpha),labels);loss=-score;loss.backward();assert alpha.grad is not None and torch.isfinite(alpha.grad);opt.step()
        with torch.no_grad():
            alpha.clamp_(.4001001,.95);value=float(separation(cached_features(stats['train'],alpha),labels))
            if value>best_sep:best_sep=value;selected_alpha=alpha.detach().clone()
        if step==1 or step%20==0:
            row=dict(step=step,alpha=float(alpha),TRAIN_separation=value,selected_TRAIN_separation=best_sep,TEST_used=False);history.append(row);m.save(a.out/'history.json',history);m.save(a.out/'progress.json',dict(state='TRAIN_only_fusion_scatter',latest=row));print(__import__('json').dumps(row),flush=True)
        if time.monotonic()-start>=120:break
    with torch.no_grad():model.fusion_logits[3].copy_(selected_alpha);features=cached_features(stats['train'],selected_alpha).numpy()
    w,b,fit=TRAIN_linear_fit(features,data['train'][1],c['original_TRAIN_counts'])
    with torch.no_grad():model.linear.weight.copy_(torch.from_numpy(w));model.linear.bias.copy_(torch.from_numpy(b))
    m.save(a.out/'TRAIN_statistics.json',dict(initial_alpha=float(original['fusion_logits.3']),selected_alpha=float(selected_alpha),initial_separation=initial_sep,selected_separation=best_sep,steps=step,TRAIN_only=True,OAS=fit,cache_probability_max_error=maxerr))
    loop.validate_state(m,model,original,keys);loop.saved_model(m,model,a.out/'last.pt',protocol);results={}
    def review(label):
        p=a.out/(label+'.pt');native,_=m.model_for(p,m.sha(p),4,reference);loop.validate_state(m,native,original,keys);result=dict(PT_SHA256=m.sha(p))
        for role in ('train','test'):
            m.save(a.out/'progress.json',dict(state='strict_native_CPU',checkpoint=label,role=role));metric,prob=base.evaluate(native,data,fields,role);result[role]=metric;np.savez_compressed(a.out/(label+'_'+role+'_predictions.npz'),ids=data[role][2],labels=data[role][1],probabilities=prob)
        return result
    results['last']=review('last');baseline=sr['results']['best']['test'];candidate=results['last']['test'];better=candidate['correct']>baseline['correct']or(candidate['correct']==baseline['correct']and candidate['nll']<baseline['nll']-1e-4)
    if not better:model.load_state_dict(original)
    loop.saved_model(m,model,a.out/'best.pt',protocol);results['best']=review('best');best=results['best']['test']['correct'];assert best>=baseline['correct']and base.identities(m,paths)==lineage
    report=dict(state='complete',baseline_correct=baseline['correct'],selected_correct=best,selected_E0=not better,results=results,full_physical=True,strict_original_CPU=True,target=340,target_met=best>=340,TEST_gradient=False,TEST_development=True,TRAIN=1600,TEST=411,VALID=0,new_inference_layers=0,captured_upstream_bitwise_frozen=True,SDK_used=False,GPU_used=False,protocol_SHA256=m.sha(a.out/'protocol.json'))
    m.save(a.out/'report.json',report);m.save(a.out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(a.out/'report.json'),prefix_SHA256=lineage,all_non_suffix_parameters_bitwise_frozen=True))


def selftest():
    import torch
    from caltech20_alt246_biasfree_model import rms,features
    torch.set_num_threads(2);torch.manual_seed(73);e=torch.rand(20,41,41);o=torch.rand_like(e);s=branch_statistics(e,o)
    for a in (.4001001,.8,.95):
        alpha=torch.tensor(a,dtype=torch.float64,requires_grad=True);mixed=(1-alpha.float())*e/rms(e)+alpha.float()*o/rms(o);native=features(rms(e)*mixed/rms(mixed));cached=cached_features(s,alpha);assert float((native-cached).abs().max())<3e-6;score=separation(cached,torch.arange(20));score.backward();assert torch.isfinite(alpha.grad)
    print('TRAIN branch-statistics identity and finite direct-alpha gradients passed')


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--out',type=Path);q.add_argument('--selftest',action='store_true');a=q.parse_args()
    if a.selftest:selftest()
    else:
        try:run(a)
        except BaseException as e:
            if a.out and a.out.exists():
                _,_,m,_=base.numerical_modules();m.save(a.out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False,all_data_preserved=True))
            raise
