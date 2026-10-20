"""Retrain original uncaptured masks with a TRAIN-fitted observation surrogate.

The response branch exists only during simulation training/proxy evaluation.
Exported state is the original architecture, with original native CPU evaluation.
The passed-prefix native gate is never replaced with a surrogate score.
"""
import argparse
import os
from pathlib import Path
import time


def response_model(base,parameters):
    import torch
    from lab_caltech20_rich_layerwise_20261018 import MeasuredPrefix,state_cpu
    from lab_caltech20_dvp_TRAIN_response_fit_20261020 import response_intensity
    class TrainingResponse(MeasuredPrefix):
        def __init__(self):
            super().__init__(base.depth)
            self.load_state_dict(state_cpu(base));self.propagator.transfer.copy_(base.propagator.transfer)
            self.register_buffer('observation_parameters',torch.tensor(parameters),persistent=False)
            self.response_enabled=False

        def capture(self,amplitude,stage,weights=None,callback=None,ids=None):
            if not self.response_enabled or stage in self.measured:
                return super().capture(amplitude,stage,weights,callback,ids)
            assert callback is None,'Response surrogate cannot open physical acquisition'
            _,amp,_=self.physical_fields(amplitude,stage,weights)
            turns=torch.remainder(torch.angle(self.plane(stage)),2*torch.pi)/(2*torch.pi)
            exact=torch.floor(turns*256).clamp(0,255)/256
            ste=turns+(exact-turns).detach()
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
    print(dict(passed=True,original_inference_byte_identical=True,export_original_structure=True,
               finite_mask_gradients=True,all_unallowed_frozen=True,SDK_used=False))


def response_assess(model,data,fields,values):
    import torch
    import lab_caltech20_dvp_layerwise_20261019 as m
    if next(model.parameters()).device.type!='cpu':
        # core.assess intentionally reconstructs a NATIVE CPU model for GPU
        # inputs. A response proxy must instead be an explicit CPU wrapper.
        rng=torch.get_rng_state();base=m.MeasuredPrefix(model.depth)
        base.load_state_dict(m.state_cpu(model));base.propagator.transfer.copy_(model.propagator.transfer.detach().cpu())
        current=response_model(base,values);torch.set_rng_state(rng)
    else:current=model
    current.response_enabled=True
    return m.assess(current,data,fields,'test')


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
    data,dm=m.data_for(a.sequence);base,_,fields,identities=m.fitting_model(a,data,dm,parent/'best.pt',digest,prefix)
    assert identities==pp['prefix_SHA256'] and m.sha(parent/'best.pt')==digest
    model=response_model(base,values).eval();original=m.state_cpu(model);allowed=m.allowed_keys(model,prefix)
    name=('1' if a.depth==4 else 'router')+'_response_retrain_20261020';out=session/'fits'/name
    assert not out.exists();out.mkdir()
    model.response_enabled=False;baseline,native_predictions=m.assess(model,data,fields,'test')
    proxy,_=response_assess(model,data,fields,values)
    protocol=dict(depth=a.depth,prefix=prefix,parent_PT_SHA256=digest,prefix_SHA256=identities,data_SHA256=dm['sha256'],
        hardware_SHA256=m.session_hardware_digest(a.sequence),allowed_keys=sorted(allowed),TRAIN=1600,TEST=411,VALID=0,
        TEST_gradient=False,TEST_informed_development=True,response_parameters=values,response_report_SHA256=m.sha(a.response/'report.json'),
        response_source_SHA256=response['source_SHA256'],source_git=m.git_head(),source_SHA256=m.sha(__file__),
        epochs=12,fit_seconds=1200,batch=8,seed=73,phase_lr=.0002,electronic_lr=.0006,alpha_lr=.0006,head_lr=.0006,
        TRAIN_loss='0.5 original-native TRAIN CE + 0.5 frozen-response TRAIN CE; label smoothing .01, no TEST gradients',
        deployment_structure_unchanged=True,response_branch_deployment=False,baseline_TEST=baseline,baseline_response_proxy=proxy,
        selection='highest response-proxy correct then NLL among candidates that still pass original-native passed-prefix gate; E0 retained otherwise',
        prerequisite_native_correct=required_correct(a.depth,prefix),next_measured_stage_target=required_correct(a.depth,prefix+1),
        no_next_capture_without_independent_review=True,old_failed_downstream_CCD_used=False)
    m.save(out/'protocol.json',protocol);np.savez_compressed(out/'baseline_TEST_predictions.npz',ids=data['test'][2],labels=data['test'][1],probabilities=native_predictions)
    def snapshot(name,epoch):
        state=m.state_cpu(model);assert all(torch.equal(v,state[k]) for k,v in original.items() if k not in allowed)
        torch.save(dict(model=state,protocol=protocol,epoch=epoch),out/(name+'.pt'))
    snapshot('best',0);selected=0;best_proxy=proxy.copy();model.cuda();groups=[]
    for key,lr in (('phase_sets.',.0002),('electronic.',.0006),('fusion_logits.',.0006),('linear.',.0006)):
        params=[p for n,p in model.named_parameters() if n in allowed and n.startswith(key)]
        if params:groups.append(dict(params=params,lr=lr,weight_decay=0 if key=='fusion_logits.' else .0001))
    for key,param in model.named_parameters():param.requires_grad_(key in allowed)
    opt=torch.optim.AdamW(groups);history=[];started=time.monotonic()
    for epoch in range(1,13):
        model.train();order=torch.randperm(1600).numpy();loss_sum=0.;native_ce=0.;response_ce=0.
        for begin in range(0,1600,8):
            ix=order[begin:begin+8];model.measured={k:torch.from_numpy(np.asarray(v['train'][ix]).copy()).cuda() for k,v in fields.items()}
            x=m.encode_images(data['train'][0][ix].cuda());y=torch.from_numpy(data['train'][1][ix].astype(np.int64)).cuda();opt.zero_grad(set_to_none=True)
            losses=[]
            for enabled in (False,True):
                model.response_enabled=enabled;p=model(x)[0];lp=p.clamp_min(1e-12).log()
                loss=.99*F.nll_loss(lp,y)-.01*lp.mean();(.5*loss).backward();losses.append(float(loss.detach()))
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for key,v in model.named_parameters() if key in allowed)
            torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1.);opt.step()
            with torch.no_grad():
                for key,param in model.named_parameters():
                    if key in allowed and key.startswith('fusion_logits.'):param.clamp_(.4001001,.95)
            native_ce+=losses[0]*len(ix);response_ce+=losses[1]*len(ix);loss_sum+=sum(losses)*.5*len(ix)
        model.eval();model.response_enabled=False;native,_=m.assess(model,data,fields,'test')
        proxy,_=response_assess(model,data,fields,values)
        eligible=native['correct']>=required_correct(a.depth,prefix)
        if eligible and m.improves(proxy['correct'],proxy['nll'],best_proxy['correct'],best_proxy['nll']):
            best_proxy=proxy.copy();selected=epoch;snapshot('best',epoch)
        row=dict(epoch=epoch,TRAIN_samples=1600,TRAIN_loss=loss_sum/1600,native_TRAIN_CE=native_ce/1600,response_TRAIN_CE=response_ce/1600,
            native_TEST=native,response_TEST_proxy=proxy,native_gate_eligible=eligible,selected_epoch=selected,selected_proxy=best_proxy,
            elapsed_s=time.monotonic()-started)
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


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true')
    p.add_argument('--sequence',type=Path);p.add_argument('--response',type=Path);p.add_argument('--depth',type=int,choices=(4,6))
    a=p.parse_args();a.attempt='strict8pp_20261020'
    if a.selftest:selftest()
    else:
        assert a.sequence and a.response and a.depth
        run(a)
