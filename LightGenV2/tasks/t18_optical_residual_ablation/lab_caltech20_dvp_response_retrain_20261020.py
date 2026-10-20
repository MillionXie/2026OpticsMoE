"""Retrain original uncaptured masks with a TRAIN-fitted observation surrogate.

The response branch exists only during simulation training/proxy evaluation.
Exported state is the original architecture, with original native CPU evaluation.
The passed-prefix native gate is never replaced with a surrogate score.
"""
import argparse
import os
from pathlib import Path
import time


def response_model(base,parameters,spatial=None,only_stage=None):
    import torch
    from lab_caltech20_rich_layerwise_20261018 import MeasuredPrefix,state_cpu
    from lab_caltech20_dvp_TRAIN_response_fit_20261020 import response_intensity
    class TrainingResponse(MeasuredPrefix):
        def __init__(self):
            super().__init__(base.depth)
            self.load_state_dict(state_cpu(base));self.propagator.transfer.copy_(base.propagator.transfer)
            self.register_buffer('observation_parameters',torch.tensor(parameters),persistent=False)
            self.only_response_stage=only_stage
            self.spatial_response=spatial is not None
            if spatial is not None:
                self.register_buffer('response_curve',torch.tensor(spatial['phase_curve_radians']),persistent=False)
                self.register_buffer('response_coefficients',torch.tensor(spatial['pupil_coefficients_radians']),persistent=False)
                self.register_buffer('response_sigma',torch.tensor(spatial['sigma_pixels']),persistent=False)
                self.register_buffer('response_background',torch.tensor(spatial['relative_background']),persistent=False)
            self.response_enabled=False

        def capture(self,amplitude,stage,weights=None,callback=None,ids=None):
            if not self.response_enabled or stage in self.measured or (self.only_response_stage is not None and stage!=self.only_response_stage):
                return super().capture(amplitude,stage,weights,callback,ids)
            assert callback is None,'Response surrogate cannot open physical acquisition'
            _,amp,_=self.physical_fields(amplitude,stage,weights)
            turns=torch.remainder(torch.angle(self.plane(stage)),2*torch.pi)/(2*torch.pi)
            exact=torch.floor(turns*256).clamp(0,255)/256
            ste=turns+(exact-turns).detach()
            if self.spatial_response:
                from lab_caltech20_dvp_TRAIN_spatial_response_20261020 import spatial_intensity
                return spatial_intensity(amp,ste[20:498,20:498],self.propagator,self.response_curve,self.response_sigma,self.response_background,self.response_coefficients)
            return response_intensity(amp,ste[20:498,20:498],self.propagator,self.observation_parameters)
    model=TrainingResponse();model.requires_grad_(False)
    assert set(model.state_dict())==set(base.state_dict())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items())
    return model


def selftest():
    import torch
    import lab_caltech20_dvp_layerwise_20261019 as m
    torch.set_num_threads(2);torch.manual_seed(73)
    base=m.MeasuredPrefix(4).eval();base.requires_grad_(False)
    # Direct alpha coordinates, as required by the actual residual model.
    with torch.no_grad():
        for v in base.fusion_logits:v.fill_(.8)
    model=response_model(base,[.79,.76,8.8,2.44]).eval()
    x=torch.rand(2,224,224)*.25
    with torch.no_grad():
        expected=base(x)[0];got=model(x)[0]
    assert torch.equal(expected,got),'Disabled surrogate must be byte-identical to original inference'
    original=m.state_cpu(model);allowed=m.allowed_keys(model,3)
    for key,value in model.named_parameters():value.requires_grad_(key in allowed)
    model.response_enabled=True
    p=model(x)[0];loss=-p[:,0].clamp_min(1e-12).log().mean();loss.backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for key,value in model.named_parameters() if key in allowed)
    torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-5).step()
    assert all(torch.equal(value,m.state_cpu(model)[key]) for key,value in original.items() if key not in allowed)
    original_model=m.MeasuredPrefix(4);original_model.load_state_dict(m.state_cpu(model))
    original_model.propagator.transfer.copy_(model.propagator.transfer);model.response_enabled=False
    with torch.no_grad():assert torch.equal(model(x)[0],original_model(x)[0])
    spatial=dict(phase_curve_radians=torch.linspace(0,2*torch.pi,17).tolist(),pupil_coefficients_radians=[0.]*8,sigma_pixels=2.,relative_background=.1)
    current=response_model(base,[1.,1.,0.,0.],spatial,'2').eval()
    with torch.no_grad():assert torch.equal(current(x)[0],base(x)[0])
    keys=m.allowed_keys(current,3)-{'linear.weight','linear.bias'};before=m.state_cpu(current)
    for name,value in current.named_parameters():value.requires_grad_(name in keys)
    current.response_enabled=True;value=-current(x)[0][:,0].clamp_min(1e-12).log().mean();value.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for name,p in current.named_parameters() if name in keys)
    torch.optim.AdamW([p for p in current.parameters() if p.requires_grad],lr=1e-5).step()
    assert all(torch.equal(v,m.state_cpu(current)[k]) for k,v in before.items() if k not in keys)
    anchored=response_model(base,[1.,1.,0.,0.],spatial,'2').eval();before=m.state_cpu(anchored)
    keys=m.allowed_keys(anchored,3)
    for name,value in anchored.named_parameters():value.requires_grad_(name in keys)
    with torch.no_grad():teacher=anchored(x)[0].detach()
    native=anchored(x)[0];lp=native.clamp_min(1e-12).log()
    kl=torch.nn.functional.kl_div(lp,teacher,reduction='batchmean')
    anchored.response_enabled=True;response=anchored(x)[0]
    loss=-.75*lp[:,0].mean()-.25*response[:,0].clamp_min(1e-12).log().mean()+kl;loss.backward()
    assert torch.isfinite(kl) and abs(float(kl))<1e-6
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for name,p in anchored.named_parameters() if name in keys)
    torch.optim.AdamW([p for p in anchored.parameters() if p.requires_grad],lr=1e-5).step()
    assert all(torch.equal(v,m.state_cpu(anchored)[k]) for k,v in before.items() if k not in keys)
    print(dict(passed=True,original_inference_byte_identical=True,export_original_structure=True,
               finite_mask_gradients=True,all_unallowed_frozen=True,spatial_single_stage=True,spatial_original_head_frozen=True,native_TRAIN_anchor_gradients=True,SDK_used=False))


def response_assess(model,data,fields,values,spatial=None,only_stage=None):
    import torch
    import lab_caltech20_dvp_layerwise_20261019 as m
    if next(model.parameters()).device.type!='cpu':
        # core.assess intentionally reconstructs a NATIVE CPU model for GPU
        # inputs. A response proxy must instead be an explicit CPU wrapper.
        rng=torch.get_rng_state();base=m.MeasuredPrefix(model.depth)
        base.load_state_dict(m.state_cpu(model));base.propagator.transfer.copy_(model.propagator.transfer.detach().cpu())
        current=response_model(base,values,spatial,only_stage);torch.set_rng_state(rng)
    else:current=model
    current.response_enabled=True
    return m.assess(current,data,fields,'test')


def recipe_name(depth,spatial,anchor):
    assert not anchor or spatial is not None
    return ('1' if depth==4 else 'router')+('_spatial_native_anchor_20261021' if anchor else '_spatial_response_retrain_20261020' if spatial is not None else '_response_retrain_20261020')


def TRAIN_teacher(model,data,fields,out):
    """Cache the unchanged passed parent's native TRAIN probabilities only."""
    import torch
    import numpy as np
    import lab_caltech20_dvp_layerwise_20261019 as m
    teacher=torch.empty((1600,20),device='cuda');model.eval();model.response_enabled=False
    with torch.no_grad():
        for begin in range(0,1600,8):
            ix=np.arange(begin,min(1600,begin+8))
            model.measured={k:torch.from_numpy(np.asarray(v['train'][ix]).copy()).cuda() for k,v in fields.items()}
            teacher[ix]=model(m.encode_images(data['train'][0][ix].cuda()))[0]
    assert torch.isfinite(teacher).all() and not teacher.requires_grad
    assert torch.allclose(teacher.sum(1),torch.ones(1600,device='cuda'),atol=1e-5)
    np.save(out/'native_teacher_TRAIN.npy',teacher.cpu().numpy())
    return teacher


def run(a):
    import numpy as np
    import torch
    from torch.nn import functional as F
    import lab_caltech20_dvp_layerwise_20261019 as m
    from lab_caltech20_dvp_strict_retake_20261020 import check_score,required_correct
    from lab_caltech20_rich_tail_refine_20261018 import windows_process_names
    torch.set_num_threads(2);torch.manual_seed(73)
    gpu='GPU-fb43b3de-2747-6835-a21f-77fc2c3532c9'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==gpu and torch.cuda.is_available()
    assert 'A100' not in torch.cuda.get_device_name(0) and not m.LOCK.exists()
    names=windows_process_names();own={os.getpid(),os.getppid()}
    desktop=m.read(Path(__file__).with_name('lab_caltech20_dvp_layerwise_profile_20261019.json'))['desktop_graphics_process_allowlist']
    import subprocess
    for row in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True).splitlines():
        uuid,pid,exe=row.split(',',2);pid=int(pid)
        if uuid.strip()==gpu:assert pid in own or names.get(pid,Path(exe.strip()).name.lower()) in desktop,('Foreign GPU owner',pid,names.get(pid,exe))
    torch.cuda.set_per_process_memory_fraction(.65,0)
    session=m.session_for(a);prefix={4:3,6:1}[a.depth]
    parent=session/'fits'/({4:'1_binary_repaired_20261020',6:'router_binary_repaired_20261020'}[a.depth])
    pr,pp=m.read(parent/'report.json'),m.read(parent/'protocol.json');pa=m.read(parent/'independent_audit.json')
    digest=check_score(pr,a.depth,prefix)
    assert pa['passed'] and pa['report_SHA256']==m.sha(parent/'report.json') and pp['prefix']==prefix
    response=m.read(a.response/'report.json')
    assert response['response_fit_admitted'] and not response['TEST_used'] and response['observation_model_only']
    assert response['source_SHA256']==m.sha(Path(__file__).with_name('lab_caltech20_dvp_TRAIN_response_fit_20261020.py'))
    values=[response['parameters'][k] for k in ('phase_scale','phase_gamma','sigma_pixels','relative_background')]
    spatial=None;only_stage=None
    if a.spatial_response is not None:
        spatial=m.read(a.spatial_response/'report.json')
        assert spatial['spatial_response_admitted'] and not spatial['TEST_used'] and not spatial['labels_used']
        assert spatial['source_SHA256']==m.sha(Path(__file__).with_name('lab_caltech20_dvp_TRAIN_spatial_response_20261020.py'))
        assert spatial['prior_report_SHA256']==m.sha(a.response/'report.json')
        only_stage={4:'2',6:'0'}[a.depth]
    data,dm=m.data_for(a.sequence);base,_,fields,identities=m.fitting_model(a,data,dm,parent/'best.pt',digest,prefix)
    assert identities==pp['prefix_SHA256'] and m.sha(parent/'best.pt')==digest
    model=response_model(base,values,spatial,only_stage).eval();original=m.state_cpu(model);allowed=m.allowed_keys(model,prefix)
    if spatial is not None and not a.native_anchor:allowed=allowed-{'linear.weight','linear.bias'}
    name=recipe_name(a.depth,spatial,a.native_anchor);out=session/'fits'/name
    assert not out.exists();out.mkdir()
    model.response_enabled=False;baseline,native_predictions=m.assess(model,data,fields,'test')
    proxy,_=response_assess(model,data,fields,values,spatial,only_stage)
    phase_lr=.0002 if a.native_anchor else .0006 if spatial is not None else .0002
    electronic_lr=.0002 if a.native_anchor else .0006
    head_lr=.0001 if a.native_anchor else 0 if spatial is not None else .0006
    protocol=dict(depth=a.depth,prefix=prefix,parent_PT_SHA256=digest,prefix_SHA256=identities,data_SHA256=dm['sha256'],
        hardware_SHA256=m.session_hardware_digest(a.sequence),allowed_keys=sorted(allowed),TRAIN=1600,TEST=411,VALID=0,
        TEST_gradient=False,TEST_informed_development=True,response_parameters=values,response_report_SHA256=m.sha(a.response/'report.json'),
        response_source_SHA256=response['source_SHA256'],source_git=m.git_head(),source_SHA256=m.sha(__file__),
        epochs=12,fit_seconds=1200,batch=8,seed=73,phase_lr=phase_lr,electronic_lr=electronic_lr,alpha_lr=electronic_lr,head_lr=head_lr,
        TRAIN_loss=('0.75 native TRAIN CE + 0.25 response TRAIN CE + 1.0 KL(native passed-parent TRAIN teacher || current native TRAIN); label smoothing .01' if a.native_anchor else '0.5 original-native TRAIN CE + 0.5 frozen-response TRAIN CE; label smoothing .01, no TEST gradients'),
        deployment_structure_unchanged=True,response_branch_deployment=False,baseline_TEST=baseline,baseline_response_proxy=proxy,
        selection='highest response-proxy correct then NLL among candidates that still pass original-native passed-prefix gate; E0 retained otherwise',
        prerequisite_native_correct=required_correct(a.depth,prefix),next_measured_stage_target=required_correct(a.depth,prefix+1),
        no_next_capture_without_independent_review=True,old_failed_downstream_CCD_used=False)
    protocol.update(spatial_response_report_SHA256=None if spatial is None else m.sha(a.spatial_response/'report.json'),
        response_only_physical_stage=only_stage,original_readout_frozen=spatial is not None and not a.native_anchor,
        native_anchor=a.native_anchor,teacher_TEST_used=False,teacher_weight=1. if a.native_anchor else 0.,
        unverified_future_response_not_applied=spatial is not None)
    counts=[348,160,638,640,102,78,68,72,98,85,68,79,80,70,68,91,69,68,68,191]
    assert sum(counts)==3141 and dm['source_cache_SHA256']==m.SOURCE_CACHE
    weights=torch.tensor(counts,dtype=torch.float32,device='cuda');weights/=weights.mean()
    protocol['original_TRAIN_class_counts']=counts if spatial is not None else None
    m.save(out/'protocol.json',protocol);np.savez_compressed(out/'baseline_TEST_predictions.npz',ids=data['test'][2],labels=data['test'][1],probabilities=native_predictions)
    def snapshot(name,epoch):
        state=m.state_cpu(model);assert all(torch.equal(v,state[k]) for k,v in original.items() if k not in allowed)
        torch.save(dict(model=state,protocol=protocol,epoch=epoch),out/(name+'.pt'))
    snapshot('best',0);selected=0;best_proxy=proxy.copy();model.cuda();groups=[]
    teacher=TRAIN_teacher(model,data,fields,out) if a.native_anchor else None
    if teacher is not None:
        protocol['teacher_TRAIN_SHA256']=m.sha(out/'native_teacher_TRAIN.npy');m.save(out/'protocol.json',protocol)
    for key,lr in (('phase_sets.',phase_lr),('electronic.',electronic_lr),('fusion_logits.',electronic_lr),('linear.',head_lr)):
        params=[p for n,p in model.named_parameters() if n in allowed and n.startswith(key)]
        if params:groups.append(dict(params=params,lr=lr,weight_decay=0 if key=='fusion_logits.' else .0001))
    for key,param in model.named_parameters():param.requires_grad_(key in allowed)
    opt=torch.optim.AdamW(groups);history=[];started=time.monotonic()
    for epoch in range(1,13):
        model.train();order=torch.randperm(1600).numpy();loss_sum=0.;native_ce=0.;response_ce=0.;kl_sum=0.
        for begin in range(0,1600,8):
            ix=order[begin:begin+8];model.measured={k:torch.from_numpy(np.asarray(v['train'][ix]).copy()).cuda() for k,v in fields.items()}
            x=m.encode_images(data['train'][0][ix].cuda());y=torch.from_numpy(data['train'][1][ix].astype(np.int64)).cuda();opt.zero_grad(set_to_none=True)
            losses=[];kl_value=0.
            for enabled in (False,True):
                model.response_enabled=enabled;p=model(x)[0];lp=p.clamp_min(1e-12).log()
                per=.99*F.nll_loss(lp,y,reduction='none')-.01*lp.mean(1)
                loss=(per*weights[y]).sum()/weights[y].sum() if spatial is not None else per.mean()
                weight=(.25 if enabled else .75) if a.native_anchor else .5
                term=weight*loss
                if teacher is not None and not enabled:
                    kl=F.kl_div(lp,teacher[ix],reduction='batchmean');term=term+kl;kl_value=float(kl.detach())
                term.backward();losses.append(float(loss.detach()))
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for key,v in model.named_parameters() if key in allowed)
            torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1.);opt.step()
            with torch.no_grad():
                for key,param in model.named_parameters():
                    if key in allowed and key.startswith('fusion_logits.'):param.clamp_(.4001001,.95)
            native_ce+=losses[0]*len(ix);response_ce+=losses[1]*len(ix);kl_sum+=kl_value*len(ix)
            loss_sum+=((.75*losses[0]+.25*losses[1]+kl_value) if a.native_anchor else sum(losses)*.5)*len(ix)
        model.eval();model.response_enabled=False;native,_=m.assess(model,data,fields,'test')
        proxy,_=response_assess(model,data,fields,values,spatial,only_stage)
        eligible=native['correct']>=required_correct(a.depth,prefix)
        if eligible and m.improves(proxy['correct'],proxy['nll'],best_proxy['correct'],best_proxy['nll']):
            best_proxy=proxy.copy();selected=epoch;snapshot('best',epoch)
        row=dict(epoch=epoch,TRAIN_samples=1600,TRAIN_loss=loss_sum/1600,native_TRAIN_CE=native_ce/1600,response_TRAIN_CE=response_ce/1600,
            native_TEST=native,response_TEST_proxy=proxy,native_gate_eligible=eligible,selected_epoch=selected,selected_proxy=best_proxy,
            native_TRAIN_parent_KL=kl_sum/1600,elapsed_s=time.monotonic()-started)
        history.append(row);m.save(out/'history.json',history);m.save(out/'progress.json',row)
        print(m.json.dumps({k:v for k,v in row.items() if k not in ('native_TEST','response_TEST_proxy','selected_proxy')}),flush=True)
        if time.monotonic()-started>=1200:break
    model.cpu();snapshot('last',epoch);results={};proxy_results={}
    for name in ('best','last'):
        path=out/(name+'.pt');model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
        model.response_enabled=False;score,p=m.assess(model,data,fields,'test')
        np.savez_compressed(out/(name+'_TEST_predictions.npz'),ids=data['test'][2],labels=data['test'][1],probabilities=p)
        results[name]=dict(TEST_development=score,PT_SHA256=m.sha(path),strict_CPU=True,predictions_SHA256=m.sha(out/(name+'_TEST_predictions.npz')))
        model.response_enabled=True;proxy_results[name]=m.assess(model,data,fields,'test')[0]
    selected_correct=results['best']['TEST_development']['correct']
    m.save(out/'report.json',dict(state='complete',results=results,response_proxy_results=proxy_results,selected_correct=selected_correct,
        selected_epoch=selected,selected_e0=selected==0,actual_epochs=epoch,target_correct=required_correct(a.depth,prefix),
        within8pp=selected_correct>=required_correct(a.depth,prefix),frozen_parameters_unchanged=True,intermediate=True,
        native_CPU=True,TEST_gradients=False,SDK_used=False,GPU_released=True,deployment_structure_unchanged=True,
        response_branch_deployment=False,elapsed_s=time.monotonic()-started))
    torch.cuda.empty_cache()


def audit(a):
    """Independently replay the exported ORIGINAL model, never its proxy class."""
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and a.spatial_response is not None
    import torch
    import numpy as np
    import lab_caltech20_dvp_layerwise_20261019 as m
    from lab_caltech20_dvp_strict_retake_20261020 import required_correct
    torch.set_num_threads(2);session=m.session_for(a);prefix={4:3,6:1}[a.depth]
    out=session/'fits'/recipe_name(a.depth,True,a.native_anchor)
    target=out/'independent_audit.json';assert not target.exists()
    p,r=m.read(out/'protocol.json'),m.read(out/'report.json')
    assert r['state']=='complete' and p['original_readout_frozen']==(not a.native_anchor) and p['response_only_physical_stage']=={4:'2',6:'0'}[a.depth]
    assert p.get('native_anchor',False)==a.native_anchor
    if a.native_anchor:assert p['teacher_TRAIN_SHA256']==m.sha(out/'native_teacher_TRAIN.npy') and not p['teacher_TEST_used']
    assert p['spatial_response_report_SHA256']==m.sha(a.spatial_response/'report.json')
    data,dm=m.data_for(a.sequence);fields,identities=m.fields_for(session,prefix,data,dm,m.session_hardware_digest(a.sequence))
    assert identities==p['prefix_SHA256'] and dm['sha256']==p['data_SHA256']
    parent=session/'fits'/({4:'1_binary_repaired_20261020',6:'router_binary_repaired_20261020'}[a.depth])
    assert m.read(parent/'independent_audit.json')['passed'] and m.sha(parent/'best.pt')==p['parent_PT_SHA256']
    origin=torch.load(parent/'best.pt',map_location='cpu',weights_only=False)['model'];results={}
    for role in ('best','last'):
        path=out/(role+'.pt');model,_=m.model_for(path,r['results'][role]['PT_SHA256'],a.depth,session/'source_reference.ts')
        state=m.state_cpu(model);allowed=m.allowed_keys(model,prefix)
        if not a.native_anchor:allowed=allowed-{'linear.weight','linear.bias'}
        assert set(p['allowed_keys'])==allowed and set(state)==set(origin)
        assert all(torch.equal(v,state[k]) and v.dtype==state[k].dtype for k,v in origin.items() if k not in allowed)
        score,probabilities=m.assess(model,data,fields,'test');saved=r['results'][role]['TEST_development']
        assert score['correct']==saved['correct'] and abs(score['nll']-saved['nll'])<=3e-6 and score['samples']==411
        with np.load(out/(role+'_TEST_predictions.npz'),allow_pickle=False) as z:
            assert np.array_equal(z['ids'],data['test'][2]) and np.array_equal(z['labels'],data['test'][1])
            assert np.max(np.abs(z['probabilities']-probabilities))<=3e-6 and np.array_equal(z['probabilities'].argmax(1),probabilities.argmax(1))
        assert m.sha(out/(role+'_TEST_predictions.npz'))==r['results'][role]['predictions_SHA256']
        if role=='best' and r['selected_e0']:assert all(torch.equal(v,state[k]) for k,v in origin.items())
        results[role]=dict(correct=score['correct'],NLL=score['nll'],PT_SHA256=m.sha(path),original_structure=True)
    q=dict(passed=True,depth=a.depth,prefix=prefix,selected_correct=results['best']['correct'],selected_e0=r['selected_e0'],
        within8pp=results['best']['correct']>=required_correct(a.depth,prefix),TRAIN=1600,TEST=411,VALID=0,
        report_SHA256=m.sha(out/'report.json'),protocol_SHA256=m.sha(out/'protocol.json'),parent_PT_SHA256=p['parent_PT_SHA256'],
        original_readout_frozen=not a.native_anchor,all_unallowed_and_measured_upstream_frozen=True,results=results,SDK_used=False,GPU_used=False,
        source_git=m.git_head(),source_SHA256=m.sha(__file__),proxy_not_used_for_native_replay=True)
    m.save(target,q);print(m.json.dumps(q),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true')
    p.add_argument('--audit',action='store_true')
    p.add_argument('--native-anchor',action='store_true')
    p.add_argument('--sequence',type=Path);p.add_argument('--response',type=Path);p.add_argument('--depth',type=int,choices=(4,6));p.add_argument('--spatial-response',type=Path)
    a=p.parse_args();a.attempt='strict8pp_20261020'
    if a.selftest:selftest()
    else:
        assert a.sequence and a.response and a.depth
        assert not a.native_anchor or a.spatial_response is not None
        if a.audit:audit(a)
        else:run(a)
