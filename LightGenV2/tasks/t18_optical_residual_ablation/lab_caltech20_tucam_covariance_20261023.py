"""One evidence-led original Linear initialization after finite terminal fitting.

The initial L6 TRAIN-covariance probe improved cached TEST90->156, without a
new layer. Confirm natively, then make one bounded continuation from retained
best. No physical operation, source mutation, random grid or TEST gradient.
"""
import argparse,os,time,traceback
from pathlib import Path
import numpy as np,torch
import lab_caltech20_dvp_layerwise_20261019 as m
import lab_caltech20_dvp_closedloop_20261021 as loop
from lab_caltech20_tucam_terminal_20261023 import assess_tail,finish,joint,allowed
from lab_caltech20_residual_terminal_cpu_20261019 import shrunk_readout
from caltech20_alt246_biasfree_model import features

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


def run(a,cfg,follow):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not a.out.exists();a.out.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(73)
    dependency=a.context/'L6_terminal';start=time.monotonic()
    while not (dependency/'report.json').exists():
        failed=a.context/'optimize6.return.json'
        if failed.exists():assert m.read(failed)['returncode']==0
        assert time.monotonic()-start<7200,'Finite dependency wait exhausted';m.save(a.out/'progress.json',dict(state='waiting_registered_L6_terminal_methods'));time.sleep(15)
    r=m.read(dependency/'report.json');assert r['strict_original_CPU'] and r['full_physical']
    diagnostic=a.context/'L6_feature_diagnosis.json';assert m.sha(diagnostic)==follow['diagnostic_SHA256']
    parent=dependency/'best.pt';data,dm=m.data_for(ROOT/cfg['sequence']);assert dm['sha256']==cfg['data_SHA256'];old=ROOT/cfg['L6_run'];reference=ROOT/cfg['sequence']/'L6_response_20261020/source_reference.ts'
    model,_=m.model_for(parent,m.sha(parent),6,reference);original=m.state_cpu(model);paths={k:Path(v['path']) for k,v in m.read(old/'L6/stage_5/capture/protocol.json')['prefix_reports'].items()};paths['5']=old/'L6/stage_5/capture';fields={};identities={}
    captured,_=m.model_for(ROOT/cfg['sequence']/cfg['L6_capture_parent_relative'],cfg['L6_capture_parent_SHA256'],6,reference);loop.validate_state(m,model,m.state_cpu(captured),allowed(model))
    assert list(paths)==['router','0','1','2','3','4','5']
    for stage,cap in paths.items():
        report=m.read(cap/'report.json');assert report['data_SHA256']==dm['sha256'] and report['TRAIN']==1600 and report['TEST']==411;fields[stage]={}
        if (cap/'independent_audit.json').exists():assert m.read(cap/'independent_audit.json')['passed']
        for role in ('train','test'):
            p=cap/role/'ccd.npy';digest=m.sha(p);assert digest==report['arrays'][role]['SHA256'];fields[stage][role]=np.load(p,mmap_mode='r');assert fields[stage][role].shape==(len(data[role][1]),478,478);identities[stage+'/'+role]=digest
    assert identities==m.read(dependency/'protocol.json')['prefix_SHA256']
    inputs={role:np.load(dependency/'input_cache'/(role+'_last_input.npy'),mmap_mode='r') for role in ('train','test')};proof=m.read(dependency/'input_cache/report.json')
    assert proof['passed']
    for role in inputs:
        assert m.sha(dependency/'input_cache'/(role+'_last_input.npy'))==proof['arrays'][role]['SHA256'] and inputs[role].shape==(len(data[role][1]),478,478)
    baseline,_=assess_tail(model,inputs,fields,data,'test');assert baseline['correct']==r['selected_correct']
    protocol=dict(config=cfg,followup=follow,source_git=m.git_head(),source_SHA256=m.sha(__file__),parent_PT_SHA256=m.sha(parent),data_SHA256=dm['sha256'],prefix_SHA256=identities,TRAIN=1600,TEST=411,VALID=0,TEST_gradient=False,new_inference_layers=0,SDK_used=False,GPU_used=False,initial_diagnostic_SHA256=m.sha(diagnostic),dependency_report_SHA256=m.sha(dependency/'report.json'),input_cache_report_SHA256=m.sha(dependency/'input_cache/report.json'))
    m.save(a.out/'protocol.json',protocol)
    if baseline['correct']>=cfg['L6_target']:
        loop.saved_model(m,model,a.out/'best.pt',protocol);m.save(a.out/'report.json',dict(state='complete',selected_correct=baseline['correct'],selected_E0=True,target_met=True,no_update_needed=True,strict_original_CPU=True,full_physical=True,dependency_native_audit_reused=True));return
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
    m.save(candidate/'TRAIN_initialization_audit.json',dict(TRAIN=1600,TEST_used_to_fit=False,geometry=geometry,alpha_changed_by_TRAIN_rule=change_alpha,original_alpha=float(original['fusion_logits.5']),new_alpha=float(model.fusion_logits[-1]),TRAIN_covariance=stats,cached_candidate_correct=score['correct'],baseline_correct=baseline['correct'],original_Linear_replaced_not_added=True,TEST_gradient=False))
    chosen,result=finish(model,original,inputs,data,fields,candidate,reference,best,selected)
    if best<cfg['L6_target']:
        gap=result['last']['train']['accuracy']-result['last']['test']['accuracy'];recipe=cfg['anti_overfit_recipe'] if gap>cfg['overfit_gap_threshold'] else cfg['recipes'][1];recipe={**recipe,'name':'covariance_initialized_'+recipe['name'],'seconds':follow['continuation_seconds'],'epochs':follow['continuation_epochs']}
        chosen,result=joint(chosen,inputs,fields,data,{**cfg,'target':cfg['L6_target']},recipe,a.out/'joint_continuation',reference,identities);best=result['best']['test']['correct']
    loop.saved_model(m,chosen,a.out/'best.pt',protocol);m.save(a.out/'report.json',dict(state='complete',baseline_correct=baseline['correct'],selected_correct=best,selected_E0=all(torch.equal(v,m.state_cpu(chosen)[n]) for n,v in original.items()),target=cfg['L6_target'],target_met=best>=cfg['L6_target'],strict_original_CPU=True,full_physical=True,TEST_development=True,TEST_gradient=False,new_inference_layers=0,SDK_used=False,GPU_used=False))


def selftest():
    torch.set_num_threads(2);torch.manual_seed(73)
    y=torch.arange(20).repeat_interleave(80);centers=torch.randn(20,256);x=centers[y]+.2*torch.randn(1600,256)
    w,b,stats=shrunk_readout(x,y,torch.ones(20,dtype=torch.float64));layer=torch.nn.Linear(256,20)
    with torch.no_grad():layer.weight.copy_(w.float());layer.bias.copy_(b.float())
    expected=(x.double()@w.T+b).softmax(1);actual=layer(x).softmax(1)
    assert torch.isfinite(actual).all() and torch.equal(expected.argmax(1),actual.argmax(1)) and float((expected-actual).abs().max())<=3e-6
    assert sum(p.numel() for p in layer.parameters())==5140 and stats['minimum_regularized_eigenvalue']>0 and scatter(x,y)['ratio']>0
    print('TRAIN-only covariance folded into unchanged5140 Linear: probability parity passed; no SDK/GPU')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--context',type=Path);p.add_argument('--out',type=Path);p.add_argument('--selftest',action='store_true');p.add_argument('--profile',type=Path,default=Path(__file__).with_name('lab_caltech20_tucam_fixed_L4_profile_20261023.json'));p.add_argument('--followup-profile',type=Path,default=Path(__file__).with_name('lab_caltech20_tucam_covariance_profile_20261023.json'));a=p.parse_args()
    if a.selftest:selftest();raise SystemExit(0)
    assert a.context is not None and a.out is not None
    try:run(a,m.read(a.profile),m.read(a.followup_profile));m.save(a.out/'return.json',dict(returncode=0))
    except BaseException as e:
        if a.out.exists():m.save(a.out/'return.json',dict(returncode=1,error=repr(e),traceback=traceback.format_exc(),automatic_retry=False))
        raise
