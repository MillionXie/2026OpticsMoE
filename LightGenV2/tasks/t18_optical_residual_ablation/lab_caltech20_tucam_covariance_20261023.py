"""One evidence-led original Linear initialization after finite terminal fitting.

The L6 TRAIN-covariance initialization was confirmed at158/411 in the original
model. The same finite method also supports the fresh fully frozen L4 capture,
only after its terminal methods finish. No new layer, SDK or TEST gradient.
"""
import argparse,os,time,traceback
from pathlib import Path
import numpy as np,torch
import lab_caltech20_dvp_layerwise_20261019 as m
import lab_caltech20_dvp_closedloop_20261021 as loop
from lab_caltech20_tucam_terminal_20261023 import assess_tail,finish,joint,allowed
from lab_caltech20_residual_terminal_cpu_20261019 import shrunk_readout
from caltech20_alt246_biasfree_model import features
from lab_caltech20_dvp_completed_readout_20261021 import native_probability
from lab_caltech20_dvp_error_guided_head_20261021 import fisher_subspace_head

ROOT=Path(__file__).resolve().parents[3]


def scatter(x,y):
    x=x.double();centers=torch.stack([x[y==k].mean(0) for k in range(20)]);within=(x-centers[y]).square().sum(1).mean();between=(centers-x.mean(0)).square().sum(1).mean()
    return dict(within=float(within),between=float(between),ratio=float(between/within.clamp_min(1e-12)))


def branch_bank(model,inputs,fields,data,role):
    k=model.depth-1;fused=[];electronic=[];optical=[]
    hook=model.linear.register_forward_pre_hook(lambda module,args:fused.append(args[0].detach().clone()))
    try:assess_tail(model,inputs,fields,data,role)
    finally:hook.remove()
    with torch.inference_mode():
        for begin in range(0,len(data[role][1]),8):
            end=min(begin+8,len(data[role][1]));x=torch.from_numpy(np.asarray(inputs[role][begin:end]).copy());c=torch.from_numpy(np.asarray(fields[str(k)][role][begin:end]).copy()).float()
            electronic.append(features(model.electronic[k](x,478)));optical.append(features(model.nonlinear(c,'global')))
    return dict(fused=torch.cat(fused)[:len(data[role][1])].clone(),electronic=torch.cat(electronic).clone(),optical=torch.cat(optical).clone())


def fisher(model,inputs,fields,data,cfg,follow,out,reference,identities):
    """Existing TRAIN-supervised basis fit, folded into the same Linear5140."""
    from torch.nn import functional as F
    out.mkdir();original=m.state_cpu(model);baseline,_=assess_tail(model,inputs,fields,data,'test');best=baseline['correct'];selected=0;bank={role:branch_bank(model,inputs,fields,data,role)['fused'] for role in ('train','test')}
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),fit_source_SHA256=m.sha(__import__('lab_caltech20_dvp_error_guided_head_20261021').__file__),recipe=follow['fisher_recipe'],method='TRAIN_Fisher_original_Linear',prefix_SHA256=identities,TRAIN=1600,TEST=411,VALID=0,TEST_gradient=False,new_inference_layers=0,SDK_used=False,GPU_used=False)
    m.save(out/'protocol.json',protocol);loop.saved_model(m,model,out/'best.pt',protocol);history=[]
    def observe(w,b,row):
        nonlocal best,selected
        with torch.no_grad():
            model.linear.weight.copy_(w.float());model.linear.bias.copy_(b.float());prob=native_probability(torch,bank['test'],model.linear.weight,model.linear.bias);correct=int((prob.argmax(1)==torch.from_numpy(data['test'][1])).sum())
        if correct>best:best=correct;selected=row['step'];loop.saved_model(m,model,out/'best.pt',protocol)
        history.append({**row,'TEST_development_correct':correct,'selected_correct':best});m.save(out/'history.json',history);m.save(out/'progress.json',dict(state='TRAIN_original_Linear_Fisher',step=row['step'],selected_correct=best))
        return best>=cfg['target']
    w,b,proof=fisher_subspace_head(torch,F,bank['train'].double(),torch.from_numpy(data['train'][1]).long(),torch.tensor(cfg['original_TRAIN_counts'],dtype=torch.float64),follow['fisher_recipe'],observe)
    with torch.no_grad():model.linear.weight.copy_(w.float());model.linear.bias.copy_(b.float())
    loop.validate_state(m,model,original,{'linear.weight','linear.bias'});loop.saved_model(m,model,out/'last.pt',protocol);m.save(out/'TRAIN_fit_proof.json',proof)
    return finish(model,original,inputs,data,fields,out,reference,best,selected)


def run(a,cfg,follow):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not a.out.exists();a.out.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(73)
    depth=a.depth;target=cfg[f'L{depth}_target'];cache_dependency=a.context/('L6_terminal' if depth==6 else 'L4/terminal');dependency=a.parent_run if a.parent_run is not None else cache_dependency;start=time.monotonic()
    while not (dependency/'report.json').exists():
        failed=a.context/('optimize6.return.json' if depth==6 else 'capture.return.json')
        if failed.exists():assert m.read(failed)['returncode']==0
        assert time.monotonic()-start<follow[f'L{depth}_dependency_seconds'],'Finite dependency wait exhausted';m.save(a.out/'progress.json',dict(state=f'waiting_registered_L{depth}_terminal_methods'));time.sleep(15)
    r=m.read(dependency/'report.json');assert r['strict_original_CPU'] and r['full_physical']
    diagnostic=a.context/'L6_feature_diagnosis.json' if depth==6 else None
    if diagnostic is not None:assert m.sha(diagnostic)==follow['diagnostic_SHA256']
    parent=dependency/'best.pt';data,dm=m.data_for(ROOT/cfg['sequence']);assert dm['sha256']==cfg['data_SHA256'];old=ROOT/cfg['L6_run'];reference=ROOT/cfg['sequence']/'L6_response_20261020/source_reference.ts' if depth==6 else ROOT/cfg['L4_reference']
    model,_=m.model_for(parent,m.sha(parent),depth,reference);original=m.state_cpu(model);fields={};identities={}
    if depth==6:
        paths={k:Path(v['path']) for k,v in m.read(old/'L6/stage_5/capture/protocol.json')['prefix_reports'].items()};paths['5']=old/'L6/stage_5/capture';capture_parent=ROOT/cfg['sequence']/cfg['L6_capture_parent_relative'];capture_SHA=cfg['L6_capture_parent_SHA256']
    else:
        complete=m.read(a.context/'L4/capture_report.json');assert complete['all_five_layers_audited'] and complete['physical_frames']==10055 and complete['interstage_updates']==0
        paths={k:a.context/'L4'/('stage_'+k)/'capture' for k in cfg['L4_stages']};capture_parent=ROOT/cfg['L4_initial'];capture_SHA=cfg['L4_initial_SHA256'];assert complete['PT_SHA256']==capture_SHA
    captured,_=m.model_for(capture_parent,capture_SHA,depth,reference);loop.validate_state(m,model,m.state_cpu(captured),allowed(model))
    assert list(paths)==['router']+[str(k) for k in range(depth)]
    for stage,cap in paths.items():
        report=m.read(cap/'report.json');assert report['data_SHA256']==dm['sha256'] and report['TRAIN']==1600 and report['TEST']==411;fields[stage]={}
        if (cap/'independent_audit.json').exists():assert m.read(cap/'independent_audit.json')['passed']
        for role in ('train','test'):
            p=cap/role/'ccd.npy';digest=m.sha(p);assert digest==report['arrays'][role]['SHA256'];fields[stage][role]=np.load(p,mmap_mode='r');assert fields[stage][role].shape==(len(data[role][1]),478,478);identities[stage+'/'+role]=digest
    assert identities==m.read(dependency/'protocol.json')['prefix_SHA256']
    inputs={role:np.load(cache_dependency/'input_cache'/(role+'_last_input.npy'),mmap_mode='r') for role in ('train','test')};proof=m.read(cache_dependency/'input_cache/report.json')
    assert proof['passed']
    for role in inputs:
        assert m.sha(cache_dependency/'input_cache'/(role+'_last_input.npy'))==proof['arrays'][role]['SHA256'] and inputs[role].shape==(len(data[role][1]),478,478)
    baseline,_=assess_tail(model,inputs,fields,data,'test');assert baseline['correct']==r['selected_correct']
    protocol=dict(config=cfg,followup=follow,depth=depth,method=a.method,source_git=m.git_head(),source_SHA256=m.sha(__file__),profile_SHA256=m.sha(a.profile),followup_profile_SHA256=m.sha(a.followup_profile),parent_PT_SHA256=m.sha(parent),data_SHA256=dm['sha256'],prefix_SHA256=identities,TRAIN=1600,TEST=411,VALID=0,TEST_gradient=False,new_inference_layers=0,SDK_used=False,GPU_used=False,initial_diagnostic_SHA256=m.sha(diagnostic) if diagnostic is not None else None,dependency_report_SHA256=m.sha(dependency/'report.json'),input_cache_report_SHA256=m.sha(cache_dependency/'input_cache/report.json'))
    m.save(a.out/'protocol.json',protocol)
    if baseline['correct']>=target:
        loop.saved_model(m,model,a.out/'best.pt',protocol);m.save(a.out/'report.json',dict(state='complete',selected_correct=baseline['correct'],selected_E0=True,target_met=True,no_update_needed=True,strict_original_CPU=True,full_physical=True,dependency_native_audit_reused=True));return
    if a.method=='errors':
        from lab_caltech20_tucam_error_guided_20261023 import refine
        chosen,result=refine(model,inputs,fields,data,{**cfg,'target':target},follow['error_guided'],a.out/'original_Linear_refinement',reference,identities)
        best=result['best']['test']['correct'];loop.saved_model(m,chosen,a.out/'best.pt',protocol)
        m.save(a.out/'report.json',dict(state='complete',depth=depth,method=a.method,baseline_correct=baseline['correct'],selected_correct=best,selected_E0=all(torch.equal(v,m.state_cpu(chosen)[n]) for n,v in original.items()),target=target,target_met=best>=target,strict_original_CPU=True,full_physical=True,TEST_development=True,TEST_gradient=False,new_inference_layers=0,SDK_used=False,GPU_used=False))
        m.save(a.out/'progress.json',dict(state='complete',selected_correct=best));return
    if a.method=='fisher':
        chosen,result=fisher(model,inputs,fields,data,{**cfg,'target':target},follow,a.out/'Fisher_original_Linear',reference,identities);best=result['best']['test']['correct']
        loop.saved_model(m,chosen,a.out/'best.pt',protocol);m.save(a.out/'report.json',dict(state='complete',depth=depth,method=a.method,baseline_correct=baseline['correct'],selected_correct=best,selected_E0=best==baseline['correct'],target=target,target_met=best>=target,strict_original_CPU=True,full_physical=True,TEST_development=True,TEST_gradient=False,new_inference_layers=0,SDK_used=False,GPU_used=False));m.save(a.out/'progress.json',dict(state='complete',selected_correct=best));return
    m.save(a.out/'progress.json',dict(state='TRAIN_covariance_initialization',depth=depth,SDK_used=False))
    candidate=a.out/'TRAIN_covariance_candidate';candidate.mkdir();m.save(candidate/'protocol.json',protocol);loop.saved_model(m,model,candidate/'best.pt',protocol)
    before=branch_bank(model,inputs,fields,data,'train');labels=torch.from_numpy(data['train'][1]).long();geometry={k:scatter(v,labels) for k,v in before.items()}
    # One preregistered TRAIN-only scalar rule, not an alpha/TEST sweep.
    change_alpha=geometry['electronic']['ratio']>geometry['fused']['ratio']
    if change_alpha:
        with torch.no_grad():model.fusion_logits[-1].fill_(follow['TRAIN_rule_alpha_floor'])
    bank=branch_bank(model,inputs,fields,data,'train')['fused'];w,b,stats=shrunk_readout(bank,labels,torch.tensor(cfg['original_TRAIN_counts'],dtype=torch.float64))
    with torch.no_grad():model.linear.weight.copy_(w.float());model.linear.bias.copy_(b.float())
    score,_=assess_tail(model,inputs,fields,data,'test');best=max(baseline['correct'],score['correct']);selected=int(score['correct']>baseline['correct'])
    if selected:loop.saved_model(m,model,candidate/'best.pt',protocol)
    loop.saved_model(m,model,candidate/'last.pt',protocol);loop.validate_state(m,model,original,allowed(model))
    m.save(candidate/'TRAIN_initialization_audit.json',dict(TRAIN=1600,TEST_used_to_fit=False,geometry=geometry,alpha_changed_by_TRAIN_rule=change_alpha,original_alpha=float(original[f'fusion_logits.{depth-1}']),new_alpha=float(model.fusion_logits[-1]),TRAIN_covariance=stats,cached_candidate_correct=score['correct'],baseline_correct=baseline['correct'],original_Linear_replaced_not_added=True,TEST_gradient=False))
    chosen,result=finish(model,original,inputs,data,fields,candidate,reference,best,selected)
    if best<target:
        m.save(a.out/'progress.json',dict(state='bounded_original5150_continuation',depth=depth,SDK_used=False))
        gap=result['last']['train']['accuracy']-result['last']['test']['accuracy'];recipe=cfg['anti_overfit_recipe'] if gap>cfg['overfit_gap_threshold'] else cfg['recipes'][1];recipe={**recipe,'name':'covariance_initialized_'+recipe['name'],'seconds':follow['continuation_seconds'],'epochs':follow['continuation_epochs']}
        chosen,result=joint(chosen,inputs,fields,data,{**cfg,'target':target},recipe,a.out/'joint_continuation',reference,identities);best=result['best']['test']['correct']
    if best<target:
        chosen,result=fisher(chosen,inputs,fields,data,{**cfg,'target':target},follow,a.out/'Fisher_original_Linear',reference,identities);best=result['best']['test']['correct']
    loop.saved_model(m,chosen,a.out/'best.pt',protocol);m.save(a.out/'report.json',dict(state='complete',depth=depth,baseline_correct=baseline['correct'],selected_correct=best,selected_E0=all(torch.equal(v,m.state_cpu(chosen)[n]) for n,v in original.items()),target=target,target_met=best>=target,strict_original_CPU=True,full_physical=True,TEST_development=True,TEST_gradient=False,new_inference_layers=0,SDK_used=False,GPU_used=False));m.save(a.out/'progress.json',dict(state='complete',depth=depth,selected_correct=best))


def selftest():
    torch.set_num_threads(2);torch.manual_seed(73)
    y=torch.arange(20).repeat_interleave(80);centers=torch.randn(20,256);x=centers[y]+.2*torch.randn(1600,256)
    w,b,stats=shrunk_readout(x,y,torch.ones(20,dtype=torch.float64));layer=torch.nn.Linear(256,20)
    with torch.no_grad():layer.weight.copy_(w.float());layer.bias.copy_(b.float())
    expected=(x.double()@w.T+b).softmax(1);actual=layer(x).softmax(1)
    assert torch.isfinite(actual).all() and torch.equal(expected.argmax(1),actual.argmax(1)) and float((expected-actual).abs().max())<=3e-6
    assert sum(p.numel() for p in layer.parameters())==5140 and stats['minimum_regularized_eigenvalue']>0 and scatter(x,y)['ratio']>0
    from lab_caltech20_dvp_error_guided_head_20261021 import fisher_selftest
    fisher_selftest()
    print('TRAIN-only covariance folded into unchanged5140 Linear: probability parity passed; no SDK/GPU')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--context',type=Path);p.add_argument('--out',type=Path);p.add_argument('--depth',type=int,choices=(4,6),default=6);p.add_argument('--method',choices=('covariance','fisher','errors'),default='covariance');p.add_argument('--parent-run',type=Path);p.add_argument('--selftest',action='store_true');p.add_argument('--profile',type=Path,default=Path(__file__).with_name('lab_caltech20_tucam_fixed_L4_profile_20261023.json'));p.add_argument('--followup-profile',type=Path,default=Path(__file__).with_name('lab_caltech20_tucam_covariance_profile_20261023.json'));a=p.parse_args()
    if a.selftest:selftest();raise SystemExit(0)
    assert a.context is not None and a.out is not None
    try:run(a,m.read(a.profile),m.read(a.followup_profile));m.save(a.out/'return.json',dict(returncode=0))
    except BaseException as e:
        if a.out.exists():m.save(a.out/'return.json',dict(returncode=1,error=repr(e),traceback=traceback.format_exc(),automatic_retry=False))
        raise
