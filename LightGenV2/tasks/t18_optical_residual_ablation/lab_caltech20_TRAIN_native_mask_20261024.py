"""Native-activation-admitted response for TRAIN mask learning; native inference.

The original physical E2 CPU gate remains the only permission for a new G2.
"""
import argparse,json,shutil,time,traceback
from pathlib import Path
import lab_caltech20_TRAIN_photometric_mask_20261024 as parent
base=parent.base

RECIPE=dict(epochs=24,seconds=1200,phase_lr=.0002,electronic_lr=.0002,alpha_lr=.0002,head_lr=.0005,weight_decay=.01,label_smoothing=.03)


def make_predictor(model,gain,bg,sigma):
    np,torch,m,_=base.numerical_modules()
    from lab_caltech20_TRAIN_native_response_fit_20261024 import response
    class TRAINNativePrefix(m.MeasuredPrefix):
        def __init__(self):
            super().__init__(4);self.load_state_dict(m.state_cpu(model),strict=True)
            self._gain=gain.tolist();self._background=bg.tolist();self._sigma=float(sigma)
            assert 0<=self._sigma<=12 and tuple(gain.shape)==tuple(bg.shape)==(8,8)
        def capture(self,amplitude,stage,weights=None,callback=None,ids=None):
            if stage!='2' or stage in self.measured or callback is not None:return super().capture(amplitude,stage,weights,callback,ids)
            field,amp,phase=self.physical_fields(amplitude,stage,weights);full=self.propagator(field).abs().square()
            value=response(full,full.new_tensor(self._gain),full.new_tensor(self._background),full.new_tensor(self._sigma))
            self.last[stage]=dict(amplitude=amp.detach(),phase_gray=(phase.detach()*256).to(torch.uint8),intensity=value.detach());return value
    rng=torch.get_rng_state();work=TRAINNativePrefix().cpu();torch.set_rng_state(rng);work.propagator.transfer.copy_(model.propagator.transfer.detach().cpu())
    assert set(work.state_dict())==set(model.state_dict());return work


def score(model,gain,bg,sigma,data,fields,role):
    _,torch,_,_=base.numerical_modules();cpu=make_predictor(model,gain,bg,sigma).eval()
    with torch.inference_mode():return base.evaluate(cpu,data,fields,role)


def selftest():
    np,torch,m,_=base.numerical_modules();torch.set_num_threads(2);torch.manual_seed(73)
    native=m.MeasuredPrefix(4).eval();work=make_predictor(native,torch.ones(8,8),torch.zeros(8,8),0.).eval()
    x=torch.rand(2,224,224)*.25;shared={k:torch.randint(0,255,(2,478,478),dtype=torch.uint8) for k in ('router','0','1')};native.measured=shared;work.measured=shared
    with torch.no_grad():expected=native(x)[0];got=work(x)[0]
    assert torch.allclose(expected,got,atol=3e-6,rtol=3e-6)
    work=make_predictor(native,torch.ones(8,8)*.7,torch.ones(8,8)*.4,4.11);amp=torch.rand(2,224,224,requires_grad=True);weights=torch.ones(2,4)*.5
    value=work.capture(amp,'2',weights);(value*torch.linspace(0,1,478)[None,None,:]).mean().backward();norms=[float(work.phase_sets[i].grad.norm()) for i in range(5,9)]
    assert all(v>0 for v in norms) and amp.grad is not None and torch.isfinite(amp.grad).all()
    native.load_state_dict(work.state_dict(),strict=True);assert all(torch.equal(v,work.state_dict()[k]) for k,v in native.state_dict().items())
    real=torch.randint(0,255,(2,478,478),dtype=torch.uint8);work.measured={'2':real};assert torch.equal(work.capture(amp,'2',weights),real.float())
    print(json.dumps(dict(passed=True,nominal_probability_maxabs=float((got-expected).abs().max()),phase_gradient_norms=norms,input_gradient_finite=True,native_state_reload_exact=True,real_CCD_bypass_exact=True,new_inference_layers=0,SDK_used=False,GPU_used=False)))


def fit(model,data,dm,fields,paths,parent_pt,reference,g,b,sigma,c,out):
    np,torch,m,loop=base.numerical_modules()
    from torch.nn import functional as F
    from lab_caltech20_dvp_TRAIN_regularization_20261021 import TRAINTransforms
    assert list(fields)==['router','0','1'] and not out.exists() and not m.LOCK.exists();out.mkdir();original=m.state_cpu(model);keys=m.allowed_keys(model,3);lineage=base.identities(m,paths)
    metric,prob=score(model,g,b,sigma,data,fields,'test');initial_train,_=score(model,g,b,sigma,data,fields,'train')
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),recipe=RECIPE,parent_PT_SHA256=m.sha(parent_pt),data_SHA256=dm['sha256'],prefix_SHA256=lineage,allowed_keys=sorted(keys),
        TRAIN=1600,TEST=411,VALID=0,TEST_gradient=False,TEST_development=True,selection_not_physical_accuracy=True,new_inference_layers=0,native_response_fit_report_SHA256='d883bd75f4cbfb822aeb2375b71bad4e266b3ee370935ddd18563e1b3a02afc6')
    m.save(out/'protocol.json',protocol);m.save(out/'baseline.json',dict(calibrated_TEST=metric,TRAIN=initial_train,not_physical_accuracy=True));np.savez_compressed(out/'baseline_TEST_predictions.npz',ids=data['test'][2],labels=data['test'][1],probabilities=prob)
    best,bnll,selected=metric['correct'],metric['nll'],0;loop.saved_model(m,model,out/'best.pt',protocol);loop.gpu_admission(m,c);work=make_predictor(model,g,b,sigma).cuda()
    for n,v in work.named_parameters():v.requires_grad_(n in keys)
    groups=[]
    for prefix,lr in [('phase_sets.',.0002),('electronic.',.0002),('fusion_logits.',.0002),('linear.',.0005)]:
        ps=[v for n,v in work.named_parameters() if n in keys and n.startswith(prefix)]
        if ps:groups.append(dict(params=ps,lr=lr,base_lr=lr,weight_decay=0 if prefix=='fusion_logits.' else .01))
    opt=torch.optim.AdamW(groups);held={};hook=work.linear.register_forward_hook(lambda module,args,value:held.update(logits=value));transforms=TRAINTransforms(work,'1',{**c,'CCD_gain_sigma':.02,'CCD_additive_sigma_DN':.5,'CCD_pixel_dropout_p':.002})
    prior=np.asarray(c['original_TRAIN_counts'],float);weights=torch.tensor(.5+.5*np.sqrt(prior/prior.mean()),dtype=torch.float32,device='cuda');rng=np.random.default_rng(73);start=time.monotonic();history=[];updates=0
    try:
        for epoch in range(1,25):
            work.cuda().train();order=rng.permutation(1600);total=used=0;scale=.1+.9*.5*(1+np.cos(np.pi*(epoch-1)/23))
            for group in opt.param_groups:group['lr']=group['base_lr']*scale
            for begin in range(0,1600,8):
                ix=order[begin:begin+8];x=loop.put_fields(work,data,fields,'train',ix,'cuda');y=torch.from_numpy(data['train'][1][ix].astype(np.int64)).cuda();opt.zero_grad(set_to_none=True);work(x);loss=F.cross_entropy(held['logits'],y,weight=weights,label_smoothing=.03)
                assert torch.isfinite(loss);loss.backward();ps=[v for v in work.parameters() if v.requires_grad];assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in ps);torch.nn.utils.clip_grad_norm_(ps,1.);opt.step();updates+=1
                with torch.no_grad():
                    for v in work.fusion_logits:
                        if v.requires_grad:v.clamp_(.4001001,.95)
                used+=len(ix);total+=float(loss.detach())*len(ix)
                if begin%160==0:m.save(out/'progress.json',dict(state='native_response_TRAIN_mask',epoch=epoch,TRAIN_consumed=used,TRAIN_loss=total/used,optimizer_updates=updates,selected_predictive_correct=best,not_physical_accuracy=True))
            assert used==1600
            if epoch==1 or epoch%2==0 or epoch==24 or time.monotonic()-start>=1200:
                metric,_=score(work,g,b,sigma,data,fields,'test')
                if metric['correct']>best or (metric['correct']==best and metric['nll']<bnll-1e-4):best,bnll,selected=metric['correct'],metric['nll'],epoch;loop.saved_model(m,work,out/'best.pt',protocol)
                row=dict(epoch=epoch,TRAIN_consumed=1600,TRAIN_loss=total/1600,calibrated_TEST_predictive_correct=metric['correct'],selected_predictive_correct=best,selected_epoch=selected,not_physical_accuracy=True,elapsed_s=time.monotonic()-start);history.append(row);m.save(out/'history.json',history);m.save(out/'progress.json',row);print(json.dumps(row),flush=True)
            if (best>=305 and selected>0) or time.monotonic()-start>=1200:break
        loop.validate_state(m,work,original,keys);loop.saved_model(m,work,out/'last.pt',protocol)
    finally:transforms.remove();hook.remove()
    results={}
    for label in ('best','last'):
        pt=out/(label+'.pt');native,_=m.model_for(pt,m.sha(pt),4,reference);loop.validate_state(m,native,original,keys);results[label]=dict(PT_SHA256=m.sha(pt),original_native_state_load=True)
        for role in ('train','test'):
            metric,prob=score(native,g,b,sigma,data,fields,role);results[label]['calibrated_'+role]=metric;np.savez_compressed(out/(label+'_'+role+'_predictions.npz'),ids=data[role][2],labels=data[role][1],probabilities=prob)
    assert results['best']['calibrated_test']['correct']==best and base.identities(m,paths)==lineage
    report=dict(state='complete',selected_predictive_correct=best,selected_epoch=selected,selected_E0=selected==0,actual_epochs=epoch,optimizer_updates=updates,results=results,initial_calibrated_TRAIN=initial_train,
        strict_original_state_and_CPU=True,selection_not_physical_accuracy=True,full_physical=False,TEST_development=True,TEST_gradient=False,new_inference_layers=0,upstream_bitwise_frozen=True,TRAIN_transform_counts=transforms.counts,protocol_SHA256=m.sha(out/'protocol.json'))
    m.save(out/'report.json',report);m.save(out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(out/'report.json'),prefix_SHA256=lineage,all_measured_ancestors_bitwise_frozen=True))
    files=[v for v in out.iterdir() if v.is_file()];m.save(out/'completed_files_SHA.json',dict(files=[dict(name=v.name,SHA256=m.sha(v),bytes=v.stat().st_size) for v in sorted(files)],not_physical_accuracy=True))
    chosen,_=m.model_for(out/'best.pt',m.sha(out/'best.pt'),4,reference);del work,opt;torch.cuda.empty_cache();return chosen,out/'best.pt',report


def run(a):
    np,torch,m,loop=base.numerical_modules()
    import lab_caltech20_TRAIN_native_response_fit_20261024 as response
    import lab_caltech20_tucam_capture_20261023 as capture
    from lab_caltech20_tucam_E2_mask_retake_20261023 import phase_probe
    p=m.read(a.profile);parent.validate_policy(p)
    assert m.sha(parent.__file__)=='dfb6ab85b3344301fd88f83fe3fbd4e774c969b5ac8115172e9bffbe69ae3607'
    assert m.sha(response.__file__)==pinned_response_sha() and m.sha(base.__file__)==p['suffix_engine_SHA256'] and m.sha(capture.__file__)==p['capture_engine_SHA256']
    assert m.sha(base.HERE/'lab_caltech20_tucam_E2_mask_retake_20261023.py')==p['phase_probe_engine_SHA256']
    assert m.sha(base.HERE/p['base_profile'])==p['base_profile_SHA256'];c=m.read(base.HERE/p['base_profile']);base.validate_config(c);c['intermediate_target']=305
    folder=base.ROOT/p['response_fit'];folder=folder.parent/'native_feature_fit';rr=m.read(folder/'report.json');audit=m.read(folder/'independent_audit.json')
    assert m.sha(folder/'report.json')==audit['report_SHA256']=='d883bd75f4cbfb822aeb2375b71bad4e266b3ee370935ddd18563e1b3a02afc6' and audit['passed'] and rr['native_response_admitted']
    assert not rr['TEST_used'] and rr['actual_updates']==200 and m.read(folder.with_suffix('.return.json'))['returncode']==0
    assert m.sha(folder/'TRAIN_native_response.npz')==rr['map_SHA256']=='2e8a1aa53f0acc4df02f4ef82609462fa2fba06e14709cce1168d9abb3c47e96'
    with np.load(folder/'TRAIN_native_response.npz',allow_pickle=False) as z:g=torch.from_numpy(z['gain'].copy());b=torch.from_numpy(z['background'].copy());sigma=float(z['sigma_pixels']);assert len(z['fit_TRAIN_ids'])==20
    proof=m.read(a.out.with_suffix('.operator_selftest.json'));assert proof['passed'] and proof['source_entry_SHA256']==m.sha(__file__) and not proof['SDK_used'] and not proof['GPU_used']
    assert not a.out.exists() and not m.LOCK.exists();a.out.mkdir();torch.set_num_threads(2);torch.manual_seed(73)
    m.save(a.out/'protocol.json',dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),recipe=RECIPE,policy_SHA256=m.sha(a.profile),native_response_report_SHA256=m.sha(folder/'report.json'),native_response_map_SHA256=m.sha(folder/'TRAIN_native_response.npz'),scheduler=False,automatic_retry=False,TEST_gradient=False))
    model,data,dm,fields,paths,pt,reference,_,_,old_phase=parent.admit(p,c,a.out);original=m.state_cpu(model)
    model,pt,rr=fit(model,data,dm,fields,paths,pt,reference,g,b,sigma,c,a.out/'mask_fit')
    new_phase=phase_probe(model,data,fields,'2',a.out/'candidate_E2_phase.bmp');changed=[k for k in ('phase_sets.5','phase_sets.6','phase_sets.7','phase_sets.8') if not torch.equal(original[k],model.state_dict()[k])]
    allowed=rr['selected_predictive_correct']>=305 and not rr['selected_E0'] and bool(changed) and new_phase!=old_phase and rr['results']['best']['calibrated_train']['nll']<rr['initial_calibrated_TRAIN']['nll']
    m.save(a.out/'mask_deployment_audit.json',dict(passed=allowed,selected_predictive_correct=rr['selected_predictive_correct'],not_physical_accuracy=True,changed_E2_phase_keys=changed,new_phase_SHA256=new_phase,old_phase_SHA256=old_phase))
    if not allowed:m.save(a.out/'report.json',dict(state='native_response_mask_proposal_unqualified',new_CCD=0,G2_not_started=True,SDK_released=not m.LOCK.exists(),source_actual_correct_still273=True));return
    actual={**c,'phase_lr':.0002,'electronic_lr':.001,'alpha_lr':.001,'head_lr':.002}
    for stage,target,recipe in [('2',305,p['observed_recipe']),('3',340,p['final_recipe'])]:
        out=a.out/('s_'+stage);out.mkdir();hardware=base.ROOT/c['hardware_by_stage'][stage]['path'];assert m.sha(hardware)==c['hardware_by_stage'][stage]['SHA256'] and shutil.disk_usage(base.ROOT).free>=120*1024**3
        if stage=='3':assert result['strict_original_CPU'] and result['results']['best']['test']['correct']>=305 and sum(v.numel() for n,v in model.named_parameters() if n in m.allowed_keys(model,5))==5150
        fields=capture.acquire(model,data,dm,fields,paths,hardware,stage,out/'capture',pt,interstage_fit=True);paths[stage]=out/'capture'
        if stage=='2':assert m.read(paths[stage]/'report.json')['phase_SHA256']==new_phase
        else:assert sum(v.numel() for n,v in model.named_parameters() if n in m.allowed_keys(model,5))==5150
        model,pt,result=base.fit_suffix(model,data,dm,fields,paths,pt,reference,actual,recipe,out/'fit',target)
        m.save(out/'endpoint_report.json',dict(actual_correct=result['selected_correct'],strict_original_CPU=result['strict_original_CPU'],full_physical=stage=='3',PT_SHA256=m.sha(pt)))
        if stage=='2' and result['selected_correct']<305:
            m.save(a.out/'report.json',dict(state='actual_E2_gate_unmet',actual_E2_correct=result['selected_correct'],G2_not_started=True,strict_original_CPU=result['strict_original_CPU'],SDK_released=not m.LOCK.exists(),full_physical=False));return
        assert result['strict_original_CPU'] and (stage=='3' or result['results']['best']['test']['correct']>=305)
    m.save(a.out/'report.json',dict(state='complete',full_physical=True,full_physical_correct=result['selected_correct'],target_met=result['selected_correct']>=340,strict_original_CPU=True,TEST_development=True,SDK_released=not m.LOCK.exists(),PT_SHA256=m.sha(pt),new_inference_layers=0,response_map_not_deployed=True,prefix_SHA256=base.identities(m,paths)))


def pinned_response_sha():return 'e79673be1ccc079768880877e7f42a05c28f295f5caebdb36e010f9e199fa8c8'


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--profile',type=Path,required=True);q.add_argument('--out',type=Path,required=True);a=q.parse_args()
    try:run(a)
    except BaseException as e:
        if a.out.exists():
            _,_,m,_=base.numerical_modules();m.save(a.out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False,all_data_preserved=True))
        raise
