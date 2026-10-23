"""Finite TRAIN-only capacity check on the newly learned photometric mask.

Only the existing 5140 affine readout changes. The response is a TRAIN forward
model, never a deployed correction; every score here is predictive, not physical.
"""
import argparse,json,time,traceback
from pathlib import Path
import lab_caltech20_TRAIN_photometric_mask_20261024 as mask
base=mask.base


def validate_policy(p):
    assert p['schema']=='TUCam_L4_photometric_frozen_mask_readout_v1'
    assert p['mask_fit'].endswith('/e2_photometric_20261024/mask_fit')
    assert p['mask_report_SHA256']=='beda3b62ab53073b89c061e96ddb8840712d04d3fd4fe14abbd522577a751ccc'
    assert p['mask_PT_SHA256']=='ffb3c5e2323d78131207e1fdae7815d19e4d10eb630594202848cba9b7c065c0'
    assert p['mask_correct']==258 and p['mask_TRAIN_correct']==1178
    assert p['recipe']==dict(analytical_fits=1,outer_steps=20,seconds=900,LBFGS_max_iter=5,LBFGS_max_eval=8,label_smoothing=.03,L2_parent_TRAIN_NLL_fraction=.1)
    assert p['TRAIN']==1600 and p['TEST']==411 and p['VALID']==0
    assert p['TEST_gradient'] is False and p['new_inference_layers']==0 and p['SDK_used'] is False and p['GPU_used'] is False
    assert p['intermediate_target']==305 and p['final_target']==340 and p['scheduler'] is False


def fit(model,data,dm,fields,paths,parent,reference,gain,bg,c,p,out):
    np,torch,m,loop=base.numerical_modules()
    from torch.nn import functional as F
    from lab_caltech20_tucam_TRAIN_linear_20261023 import TRAIN_linear_fit,TRAIN_affine_objective
    assert list(fields)==['router','0','1'] and not out.exists() and not m.LOCK.exists();out.mkdir()
    original=m.state_cpu(model);keys={'linear.weight','linear.bias'};lineage=base.identities(m,paths)
    source=base.ROOT/p['mask_fit'];source_report=m.read(source/'report.json');cpu=mask.make_predictor(model,gain,bg).eval()
    features={};errors={}
    with torch.inference_mode():
        for role,n in (('train',1600),('test',411)):
            chunks=[]
            for begin in range(0,n,8):
                ix=np.arange(begin,min(begin+8,n));cpu(loop.put_fields(cpu,data,fields,role,ix,'cpu'));chunks.append(cpu.readout['features'].cpu().numpy().copy())
                if begin%80==0:m.save(out/'progress.json',dict(state='fresh_calibrated_frozen_mask_features',role=role,consumed=begin+len(ix),not_physical_accuracy=True,TEST_fit=False))
            values=np.concatenate(chunks);assert values.shape==(n,256) and np.isfinite(values).all();features[role]=torch.from_numpy(values)
            cache=out/(role+'_features.npz');np.savez_compressed(cache,ids=data[role][2],labels=data[role][1],features=values)
            with np.load(source/('best_'+role+'_predictions.npz'),allow_pickle=False) as z:
                assert np.array_equal(z['ids'],data[role][2]) and np.array_equal(z['labels'],data[role][1])
                errors[role]=float(np.max(np.abs(cpu.linear(features[role]).softmax(1).numpy()-z['probabilities'])))
            assert errors[role]<=3e-6
    cfg=p['recipe'];head=torch.nn.Linear(256,20).double();head.load_state_dict(model.linear.state_dict());labels=torch.from_numpy(data['train'][1].astype(np.int64));train=features['train'].double()
    prior=np.asarray(c['original_TRAIN_counts'],float);weights=torch.tensor(.5+.5*np.sqrt(prior/prior.mean()),dtype=torch.float64)
    regularization=2*cfg['L2_parent_TRAIN_NLL_fraction']*source_report['results']['best']['calibrated_train']['nll']/float(head.weight.detach().square().sum())
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),policy=p,parent_PT_SHA256=m.sha(parent),data_SHA256=dm['sha256'],prefix_SHA256=lineage,
        allowed_keys=sorted(keys),trainable_parameters=5140,new_inference_layers=0,TRAIN=1600,TEST=411,VALID=0,TEST_gradient=False,TEST_development=True,
        all_masks_electronics_alpha_frozen=True,SDK_used=False,GPU_used=False,selection_not_physical_accuracy=True,response_map_not_deployed=True,
        feature_probability_max_errors=errors,feature_cache_SHA256={r:m.sha(out/(r+'_features.npz')) for r in features},TRAIN_only_L2=regularization)
    m.save(out/'protocol.json',protocol)
    baseline=source_report['results']['best']['calibrated_test'];best=baseline['correct'];bnll=baseline['nll'];selected='parent_joint_mask';best_head={k:v.detach().clone() for k,v in head.state_dict().items()};history=[];calls=0;started=time.monotonic()
    def cached_metrics(role):
        with torch.inference_mode():
            logits=F.linear(features[role],head.weight.float(),head.bias.float());y=torch.from_numpy(data[role][1].astype(np.int64))
            return dict(correct=int((logits.argmax(1)==y).sum()),samples=len(y),nll=float(F.cross_entropy(logits,y)))
    def review(method):
        nonlocal best,bnll,selected,best_head
        row=dict(method=method,TRAIN=cached_metrics('train'),calibrated_TEST=cached_metrics('test'),not_physical_accuracy=True,full_TRAIN_gradient_evaluations=calls,elapsed_s=time.monotonic()-started)
        v=row['calibrated_TEST']
        if v['correct']>best or (v['correct']==best and v['nll']<bnll-1e-4):
            best,bnll,selected=v['correct'],v['nll'],method;best_head={k:x.detach().clone() for k,x in head.state_dict().items()}
        row.update(selected_predictive_correct=best,selected_method=selected);history.append(row);m.save(out/'history.json',history);m.save(out/'progress.json',row);print(json.dumps(row),flush=True)
    review('parent_joint_mask')
    initial_objective=float(TRAIN_affine_objective(head.weight,head.bias,train,labels,weights,.03,regularization)[0].detach())
    w,b,stats=TRAIN_linear_fit(train.numpy(),labels.numpy(),prior)
    with torch.no_grad():head.weight.copy_(torch.from_numpy(w));head.bias.copy_(torch.from_numpy(b))
    review('TRAIN_OAS_single_fit')
    analytical_objective=float(TRAIN_affine_objective(head.weight,head.bias,train,labels,weights,.03,regularization)[0].detach())
    use_analytical=analytical_objective<initial_objective
    if not use_analytical:head.load_state_dict({k:v.double() for k,v in model.linear.state_dict().items()})
    m.save(out/'TRAIN_statistics.json',dict(OAS=stats,parent_TRAIN_objective=initial_objective,OAS_TRAIN_objective=analytical_objective,LBFGS_start_from_OAS=use_analytical,initialization_chosen_by_TRAIN_only=True))
    opt=torch.optim.LBFGS(head.parameters(),lr=1,max_iter=5,max_eval=8,history_size=20,line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-10)
    outer=0
    for outer in range(1,cfg['outer_steps']+1):
        def closure():
            nonlocal calls
            opt.zero_grad(set_to_none=True);loss,_,_=TRAIN_affine_objective(head.weight,head.bias,train,labels,weights,.03,regularization)
            assert torch.isfinite(loss);loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in head.parameters());calls+=1;return loss
        opt.step(closure);review('TRAIN_LBFGS_outer_'+str(outer))
        if best>=305 or time.monotonic()-started>=cfg['seconds']:break
    model.linear.load_state_dict(head.float().state_dict());loop.validate_state(m,model,original,keys);loop.saved_model(m,model,out/'last.pt',protocol)
    model.linear.load_state_dict({k:v.float() for k,v in best_head.items()});loop.validate_state(m,model,original,keys);loop.saved_model(m,model,out/'best.pt',protocol)
    results={}
    for label in ('best','last'):
        pt=out/(label+'.pt');native,_=m.model_for(pt,m.sha(pt),4,reference);loop.validate_state(m,native,original,keys);results[label]=dict(PT_SHA256=m.sha(pt),original_native_state_load=True)
        for role in ('train','test'):
            m.save(out/'progress.json',dict(state='strict_original_state_calibrated_CPU',checkpoint=label,role=role,not_physical_accuracy=True))
            metric,prob=mask.predictive_score(native,gain,bg,data,fields,role);results[label]['calibrated_'+role]=metric
            np.savez_compressed(out/(label+'_'+role+'_predictions.npz'),ids=data[role][2],labels=data[role][1],probabilities=prob)
    assert results['best']['calibrated_test']['correct']==best and base.identities(m,paths)==lineage
    report=dict(state='complete',selected_predictive_correct=best,selected_method=selected,selected_parent=selected=='parent_joint_mask',results=results,
        strict_original_state_and_CPU=True,selection_not_physical_accuracy=True,full_physical=False,TEST_development=True,TEST_gradient=False,new_inference_layers=0,
        all_non_readout_bitwise_frozen=True,SDK_used=False,GPU_used=False,analytical_fits=1,outer_steps=outer,full_TRAIN_gradient_evaluations=calls,
        feature_probability_max_errors=errors,protocol_SHA256=m.sha(out/'protocol.json'),mask_report_SHA256=p['mask_report_SHA256'])
    m.save(out/'report.json',report);m.save(out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(out/'report.json'),prefix_SHA256=lineage,all_non_readout_bitwise_frozen=True))
    files=[v for v in out.iterdir() if v.is_file()];m.save(out/'completed_files_SHA.json',dict(files=[dict(name=v.name,SHA256=m.sha(v),bytes=v.stat().st_size) for v in sorted(files)],not_physical_accuracy=True))
    return report


def run(a):
    np,torch,m,loop=base.numerical_modules();p=m.read(a.profile);validate_policy(p)
    import lab_caltech20_tucam_TRAIN_linear_20261023 as linear
    assert m.sha(mask.__file__)==p['mask_engine_SHA256'] and m.sha(linear.__file__)==p['linear_engine_SHA256']
    bp=base.HERE/p['mask_policy'];assert m.sha(bp)==p['mask_policy_SHA256'];old_policy=m.read(bp);mask.validate_policy(old_policy)
    for key,name in [('suffix_engine_SHA256','lab_caltech20_tucam_layerwise_20261023.py'),('operator_SHA256','lab_caltech20_TRAIN_response_operator_20261024.py'),('flat_engine_SHA256','lab_caltech20_TRAIN_flat_response_20261024.py')]:assert m.sha(base.HERE/name)==old_policy[key]
    assert m.sha(base.HERE/old_policy['base_profile'])==old_policy['base_profile_SHA256']
    c=m.read(base.HERE/old_policy['base_profile']);base.validate_config(c)
    source=base.ROOT/p['mask_fit'];r=m.read(source/'report.json');audit=m.read(source/'independent_audit.json');parent=source/'best.pt'
    assert m.read(source.parent.with_suffix('.return.json'))['returncode']==0 and not m.LOCK.exists()
    assert r['state']=='complete' and r['strict_original_state_and_CPU'] and not r['selected_E0'] and r['actual_epochs']==16
    assert m.sha(source/'report.json')==p['mask_report_SHA256']==audit['report_SHA256'] and audit['passed']
    assert m.sha(parent)==p['mask_PT_SHA256']==r['results']['best']['PT_SHA256']
    assert r['results']['best']['calibrated_train']['correct']==1178 and r['results']['best']['calibrated_test']['correct']==258
    assert m.read(source/'local_SHA_backup.json')['all_local_remote_SHA_matched']
    assert not a.out.exists();a.out.mkdir();torch.set_num_threads(2);torch.manual_seed(73)
    old,data,dm,fields,paths,_,reference,gain,bg,_=mask.admit(old_policy,c,a.out)
    model,_=m.model_for(parent,p['mask_PT_SHA256'],4,reference);loop.validate_state(m,model,m.state_cpu(old),m.allowed_keys(model,3))
    report=fit(model,data,dm,fields,paths,parent,reference,gain,bg,c,p,a.out/'readout_fit')
    m.save(a.out/'report.json',dict(state='offline_readout_complete',selected_predictive_correct=report['selected_predictive_correct'],not_physical_accuracy=True,
        actual_source_correct_still273=True,new_CCD=0,G2_not_started=True,SDK_released=not m.LOCK.exists(),next_requires_original_physical_E2_gate305=True))


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--profile',type=Path,required=True);q.add_argument('--out',type=Path,required=True);a=q.parse_args()
    try:run(a)
    except BaseException as e:
        if a.out.exists():
            _,_,m,_=base.numerical_modules();m.save(a.out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False,all_data_preserved=True))
        raise
