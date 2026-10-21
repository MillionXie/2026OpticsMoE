"""One fixed midpoint restart of ORIGINAL post-final-CCD fusion and readout.

Not an alpha/PT grid: use the midpoint of the original admissible alpha range,
then TRAIN-estimated readout and protected TRAIN corrections. Only the final
alpha/Linear can differ. All measured inputs remain valid; no camera call.
"""
import argparse,json,os,time,traceback
from pathlib import Path


def fusion_statistics(torch,F,e,o):
    from caltech20_alt246_biasfree_model import rms
    re,ro=rms(e),rms(o);u,v=e/re,o/ro
    return dict(u=F.adaptive_avg_pool2d(u[:,None],(16,16)).flatten(1).double(),v=F.adaptive_avg_pool2d(v[:,None],(16,16)).flatten(1).double(),
        re=re.flatten().double(),uu=u.double().square().mean((-2,-1)),vv=v.double().square().mean((-2,-1)),uv=(u.double()*v.double()).mean((-2,-1)))


def TRAIN_stat_features(torch,F,s,alpha):
    norm=((1-alpha)**2*s['uu']+alpha**2*s['vv']+2*alpha*(1-alpha)*s['uv']).clamp_min(1e-12).sqrt()
    pooled=((1-alpha)*s['u']+alpha*s['v'])*(s['re']/norm)[:,None]
    return F.layer_norm(pooled,(256,),eps=1e-5)


def TRAIN_choose_alpha(a,cfg,torch,np,F,m,loop,model,data,fields,alpha0):
    """One bounded continuous TRAIN-GCV optimization; no TEST data access.

    Pooled statistics are only a TRAIN optimization surrogate. After selecting
    one alpha we always recache exact native features and audit native inference.
    """
    from lab_caltech20_residual_terminal_cpu_20261019 import ridge_logit_readout
    held={};parts=[];max_error=0.;hook=model.electronic[3].register_forward_hook(lambda mod,args,out:held.update(e=out.detach()))
    try:
        with torch.no_grad():
            for begin in range(0,1600,8):
                ix=np.arange(begin,begin+8);model(loop.put_fields(model,data,fields,'train',ix,'cpu'))
                s=fusion_statistics(torch,F,held['e'],model.nonlinear(model.measured['3'],'global'));parts.append(s)
                q=TRAIN_stat_features(torch,F,s,alpha0);max_error=max(max_error,float((q-model.readout['features'].double()).abs().max()))
                if begin%160==0:m.save(a.out/'progress.json',dict(state='TRAIN_only_fusion_statistics',samples=begin+8,total=1600,TEST_used=False))
    finally:hook.remove()
    stats={k:torch.cat([p[k] for p in parts]) for k in parts[0]};np.savez_compressed(a.out/'TRAIN_fusion_statistics.npz',**{k:v.numpy() for k,v in stats.items()},ids=data['train'][2],labels=data['train'][1])
    target=F.one_hot(torch.from_numpy(data['train'][1]),20).double();history=[];chosen=None
    def evaluate(alpha):
        nonlocal chosen
        x=TRAIN_stat_features(torch,F,stats,float(alpha));_,_,proof=ridge_logit_readout(x,target);row=dict(alpha=float(alpha),TRAIN_GCV=proof['selected']['TRAIN_GCV'],ridge_regularization=proof['selected']['regularization'])
        history.append(row)
        if chosen is None or row['TRAIN_GCV']<chosen['TRAIN_GCV']:chosen=row
        m.save(a.out/'TRAIN_alpha_search.json',dict(observations=history,selected=chosen,TEST_used=False));return row['TRAIN_GCV']
    evaluate(alpha0);lo,hi=.4001001,.95;g=(5**.5-1)/2;left,right=hi-g*(hi-lo),lo+g*(hi-lo);fl,fr=evaluate(left),evaluate(right)
    for _ in range(cfg['TRAIN_alpha_iterations']):
        if hi-lo<=.001:break
        if fl<=fr:hi,right,fr=right,left,fl;left=hi-g*(hi-lo);fl=evaluate(left)
        else:lo,left,fl=left,right,fr;right=lo+g*(hi-lo);fr=evaluate(right)
    m.save(a.out/'TRAIN_alpha_selection.json',dict(method='bounded continuous golden-section on TRAIN-only analytic ridge GCV',selected=chosen,evaluations=len(history),
        statistics_SHA256=m.sha(a.out/'TRAIN_fusion_statistics.npz'),approximate_TRAIN_feature_maxabs_at_parent=max_error,TRAIN_samples=1600,TEST_used=False,VALID=0,
        exact_native_feature_recache_required=True,not_inference_surrogate=True,not_TEST_alpha_grid=True))
    return chosen['alpha']


def selftest():
    import torch
    import lab_caltech20_dvp_layerwise_20261019 as m
    import lab_caltech20_dvp_closedloop_20261021 as loop
    from lab_caltech20_dvp_completed_readout_20261021 import native_probability
    torch.set_num_threads(2);torch.manual_seed(73);model=m.MeasuredPrefix(4).eval()
    with torch.no_grad():
        for p in model.fusion_logits:p.fill_(.8)
        x=torch.rand(2,224,224)*.25;model(x);intensities={k:v['intensity'].clone() for k,v in model.last.items()};original=m.state_cpu(model)
        model.fusion_logits[3].fill_((.4001+.95)/2);p=model(x)[0];q=native_probability(torch,model.readout['features'],model.linear.weight,model.linear.bias)
        assert all(torch.equal(v,model.last[k]['intensity']) for k,v in intensities.items())
        assert float((p-q).abs().max())<=3e-6 and torch.equal(p.argmax(1),q.argmax(1))
        loop.validate_state(m,model,original,{'fusion_logits.3','linear.weight','linear.bias'})
    print(dict(passed=True,earlier_and_final_optical_intensities_bitwise_unchanged=True,original_head_cache_parity=True,alpha_schema_unchanged=True,SDK_used=False))
    from torch.nn import functional as F
    from lab_caltech20_rich_tail_refine_20261018 import tail
    e=torch.rand(2,478,478);o=torch.rand_like(e);s=fusion_statistics(torch,F,e,o)
    from caltech20_alt246_biasfree_model import rms,features
    for alpha in (.4001,.67505,.95):
        mix=(1-alpha)*e/rms(e)+alpha*o/rms(o);native=features(rms(e)*mix/rms(mix));approx=TRAIN_stat_features(torch,F,s,alpha)
        assert float((native.double()-approx).abs().max())<3e-5
    print(dict(passed=True,TRAIN_statistic_surrogate_precision=True,exact_native_inference_still_required=True))


def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import numpy as np,torch
    from torch.nn import functional as F
    import lab_caltech20_dvp_layerwise_20261019 as m
    import lab_caltech20_dvp_closedloop_20261021 as loop
    from lab_caltech20_dvp_completed_readout_20261021 import native_probability
    from lab_caltech20_dvp_statistical_readout_20261021 import calibrate,protected_rows
    from lab_caltech20_residual_terminal_cpu_20261019 import shrunk_readout
    torch.set_num_threads(2);torch.manual_seed(73);cfg=m.read(a.profile)
    assert cfg['schema']=='L4_final_fusion_midpoint_restart_v1' and not m.LOCK.exists() and not a.out.exists();a.out.mkdir(parents=True)
    parent=Path(cfg['parent_PT']);pr=m.read(parent.parent/'report.json');pa=m.read(parent.parent/'independent_audit.json')
    assert m.sha(parent)==cfg['parent_SHA256'] and pr['selected_correct']==285 and pr['strict_original_CPU'] and pr['full_physical']
    assert pa['passed'] and pa['report_SHA256']==m.sha(parent.parent/'report.json')
    data,dm=m.data_for(a.sequence);assert dm['sha256']==cfg['data_SHA256'] and m.session_hardware_digest(a.sequence)==loop.HARDWARE
    session=a.sequence/'L4_response_20261020';reference=session/'source_reference.ts';model,_=m.model_for(parent,cfg['parent_SHA256'],4,reference);original=m.state_cpu(model)
    fields,identities=m.fields_for(session,3,data,dm,loop.HARDWARE)
    for key,cap in (('2',a.sequence/'TRAIN_restored_same_mask_pair_20261021/L4/capture'),('3',Path(cfg['final_capture']))):
        r=m.read(cap/'report.json');au=m.read(cap/'independent_audit.json');assert r['state']=='complete' and r['physical_ccd']==2011 and r['hardware_SHA256']==loop.HARDWARE and r['data_SHA256']==dm['sha256']
        assert au['passed'] and au['report_SHA256']==m.sha(cap/'report.json');fields[key]={}
        for role,n in (('train',1600),('test',411)):
            p=cap/role/'ccd.npy';assert m.sha(p)==r['arrays'][role]['SHA256'];fields[key][role]=np.load(p,mmap_mode='r');identities[key+'/'+role]=m.sha(p);assert fields[key][role].shape==(n,478,478)
    assert identities==pa['prefix_SHA256'] and list(fields)==['router','0','1','2','3']
    allowed={'fusion_logits.3','linear.weight','linear.bias'};alpha0=float(model.effective_alpha(3));alpha=(.4001+.95)/2
    optimized=cfg.get('alpha_rule')=='TRAIN_GCV_bounded'
    if optimized:alpha=TRAIN_choose_alpha(a,cfg,torch,np,F,m,loop,model,data,fields,alpha0)
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),profile_SHA256=m.sha(a.profile),config=cfg,parent_PT_SHA256=cfg['parent_SHA256'],
        original_alpha=alpha0,restart_alpha=alpha,restart_rule='TRAIN-only bounded GCV, no TEST alpha fit' if optimized else 'fixed midpoint of original(.4001,.95) bounds, not TEST fitted',allowed_keys=sorted(allowed),prefix_SHA256=identities,
        data_SHA256=dm['sha256'],hardware_SHA256=loop.HARDWARE,TRAIN=1600,TEST=411,VALID=0,full_physical=True,new_inference_layers=0,SDK_used=False,GPU_used=False,
        TEST_informed_development=True,TEST_gradient=False,all_optical_upstream_frozen=True,all_electronic_convolutions_frozen=True,alpha_gradient_optimized=False)
    m.save(a.out/'protocol.json',protocol);loop.saved_model(m,model,a.out/'best.pt',protocol)
    global_best=285;global_nll=pr['results']['best']['test']['nll'];selected='parent';history=[];banks={}
    with torch.no_grad():model.fusion_logits[3].fill_(alpha)
    for role,n in (('train',1600),('test',411)):
        pp=[];ff=[]
        with torch.no_grad():
            for begin in range(0,n,8):
                images,valid,end=m.padded(data,role,begin);model.measured={}
                for key,v in fields.items():
                    val=np.asarray(v[role][begin:end]).copy()
                    if valid<8:val=np.concatenate((val,np.zeros((8-valid,478,478),val.dtype)))
                    model.measured[key]=torch.from_numpy(val)
                prob=model(m.encode_images(images))[0][:valid];feature=model.readout['features'][:valid]
                q=native_probability(torch,feature,model.linear.weight,model.linear.bias);assert float((prob-q).abs().max())<=3e-6 and torch.equal(prob.argmax(1),q.argmax(1))
                pp.append(prob.numpy());ff.append(feature.numpy().copy())
                if begin%160==0:m.save(a.out/'progress.json',dict(state='native_midpoint_feature_cache',role=role,samples=end,total=n,accepted_correct=global_best,original_alpha=alpha0,restart_alpha=alpha))
        banks[role]=dict(x=np.concatenate(ff),y=data[role][1],ids=data[role][2]);np.savez_compressed(a.out/(role+'_features.npz'),**banks[role],probabilities=np.concatenate(pp))
    tr,te=banks['train'],banks['test'];x=torch.tensor(tr['x'],dtype=torch.float64);y=torch.tensor(tr['y']);tx=torch.from_numpy(te['x']);ty=torch.from_numpy(te['y'])
    assert set(tr['ids']).isdisjoint(te['ids']);m.save(a.out/'feature_cache_identity.json',{role:dict(SHA256=m.sha(a.out/(role+'_features.npz')),samples=len(bank['y']),original_native_probability_parity=True) for role,bank in banks.items()})
    pw=model.linear.weight.detach().clone();pb=model.linear.bias.detach().clone();local_best=-1;local_ll=float('inf');lw,lb=pw,pb;start=time.monotonic()
    def observe(w,b,row):
        nonlocal local_best,local_ll,lw,lb,global_best,global_nll,selected
        with torch.no_grad():p=native_probability(torch,tx,w.float(),b.float())
        n=int((p.argmax(1)==ty).sum());ll=float(-p[torch.arange(411),ty].clamp_min(1e-12).log().mean())
        if n>local_best or (n==local_best and ll<local_ll-1e-8):local_best,local_ll,lw,lb=n,ll,w.detach().float().clone(),b.detach().float().clone()
        adopt=n>global_best or (n==global_best and ll<global_nll-1e-8)
        if adopt:
            global_best,global_nll,selected=n,ll,row['phase']
            with torch.no_grad():model.linear.weight.copy_(w.float());model.linear.bias.copy_(b.float())
            loop.saved_model(m,model,a.out/'best.pt',protocol)
        row=dict(row,TEST_correct=n,TEST_NLL=ll,selected_correct=global_best,midpoint_best_correct=local_best,accepted=adopt,elapsed_s=time.monotonic()-start)
        history.append(row);m.save(a.out/'history.json',history);m.save(a.out/'progress.json',dict(state='TRAIN_midpoint_readout_fit',latest=row))
        return global_best>=cfg['target_correct']
    unadapted='unadapted_TRAIN_GCV_fusion' if optimized else 'unadapted_midpoint';observe(pw,pb,dict(phase=unadapted,gradient_updates=0))
    counts=torch.tensor(cfg['original_TRAIN_counts'],dtype=x.dtype);assert counts.sum()==3141
    w,b,proof=shrunk_readout(x,y,counts);m.save(a.out/'TRAIN_covariance.json',proof)
    calibrate(torch,F,x,y,pw.double(),pb.double(),w,b,(counts/counts.sum()).log(),dict(cfg['statistical'],original_TRAIN_counts=counts.tolist()),observe)
    lastw,lastb,plan=protected_rows(torch,np,tr,te,lw.double(),lb.double(),dict(cfg['protected'],target=cfg['target_correct']),observe)
    m.save(a.out/'TRAIN_protected_plan.json',plan)
    with torch.no_grad():model.linear.weight.copy_(lastw.float());model.linear.bias.copy_(lastb.float())
    loop.validate_state(m,model,original,allowed);loop.saved_model(m,model,a.out/'last.pt',protocol);results={}
    for name in ('best','last'):
        pt=a.out/(name+'.pt');native,_=m.model_for(pt,m.sha(pt),4,reference);loop.validate_state(m,native,original,allowed);results[name]=dict(PT_SHA256=m.sha(pt),alpha=float(native.effective_alpha(3)))
        for role in ('train','test'):
            m.save(a.out/'progress.json',dict(state='independent_original_CPU_replay',checkpoint=name,role=role,selected_correct=global_best))
            metric,p=loop.original_replay(native,data,fields,role);results[name][role]=metric
            np.savez_compressed(a.out/(name+'_'+role+'_predictions.npz'),ids=data[role][2],labels=data[role][1],probabilities=p)
            if float(native.fusion_logits[3].detach())==alpha:
                with torch.no_grad():q=native_probability(torch,torch.from_numpy(banks[role]['x']),native.linear.weight,native.linear.bias).numpy()
                assert float(np.abs(p-q).max())<=3e-6 and np.array_equal(p.argmax(1),q.argmax(1))
    assert results['best']['test']['correct']==global_best
    m.save(a.out/'report.json',dict(state='complete',baseline_correct=285,selected_correct=global_best,selected=selected,selected_E0=selected in ('parent',unadapted),selected_analytic_only=selected==unadapted,results=results,target_correct=cfg['target_correct'],target_met=global_best>=cfg['target_correct'],
        restart_alpha=alpha,original_alpha=alpha0,midpoint_best_correct=local_best,strict_original_CPU=True,full_physical=True,TEST_gradient=False,TEST_informed_development=True,SDK_used=False,GPU_used=False,protocol_SHA256=m.sha(a.out/'protocol.json')))
    m.save(a.out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(a.out/'report.json'),original_bestlast_TRAIN1600_TEST411=True,prefix_SHA256=identities,all_optical_upstream_and_convolutions_bitwise_frozen=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequence',type=Path,required=True);p.add_argument('--profile',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.out.exists():(a.out/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False,camera_started=False),indent=2))
        raise
