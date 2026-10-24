"""Disclosed teacher adaptation, then TRAIN-only distillation; no independent TEST.

Select a real training checkpoint in a human-requested interval. Preserve all
existing higher checkpoints and record observed peak; never modify predictions.
"""
import argparse,json,traceback
from pathlib import Path
import lab_caltech20_tucam_layerwise_20261023 as base
from lab_caltech20_readout_transfer_20261024 import optimize_pool
from lab_caltech20_TEST_supervised_readout_20261024 import fit_head


def interval_for(simulation_correct,gap_pp,samples=411):
    import math
    return [math.ceil(simulation_correct-gap_pp*samples/100-1e-10),simulation_correct-1]


def validate(c):
    assert c['schema']=='T18_disclosed_interval_transfer_v1' and c['depth']in(4,6)
    assert c['selection_interval']==interval_for(c['simulation_correct'],c['maximum_gap_pp'])
    assert c['TRAIN']==1600 and c['former_TEST']==411 and c['VALID']==0
    assert c['teacher_seen_TEST']and not c['student_direct_TEST_gradient']and c['independent_test_samples']==0
    assert c['trainable_parameters']==5140 and not any(c[k]for k in('SDK_used','GPU_used','new_capture','replace_original_development_best'))
    assert c['outer_steps']==(120 if c['depth']==4 else 600)and c['fit_seconds_per_recipe']==120 and c['LBFGS_inner_iterations']==1 and c['LBFGS_lr']==.25
    assert c['temperature']==2 and c['label_smoothing']==.01 and c['preconditioned_delta_L2']==.0005
    assert c['gradient_balance_multipliers']=={'KL_TRAIN_interval':2.}and c['ratio_bounds']==[1/19,19.]


def run(a):
    np,torch,m,loop=base.numerical_modules();torch.set_num_threads(2)
    from lab_caltech20_tucam_TRAIN_linear_20261023 import canonical_affine_logits
    c=m.read(a.profile);validate(c);assert not a.out.exists()and not m.LOCK.exists()
    root=base.ROOT;task=base.HERE;depth=c['depth'];keys={'linear.weight','linear.bias'}
    pt=root/c['parent_PT'];protected=root/c['protected_manifest'];reference=root/c['reference']
    assert m.sha(pt)==c['parent_PT_SHA256']and m.sha(protected)==c['protected_manifest_SHA256']and m.sha(reference)==c['reference_SHA256']
    model,_=m.model_for(pt,m.sha(pt),depth,reference);original=m.state_cpu(model);assert sum(original[k].numel()for k in keys)==5140
    data,dm=m.data_for(root/c['sequence']);assert dm['sha256']==c['data_SHA256']
    if depth==4:
        parent_protocol=m.read(root/c['parent_protocol']);policy=m.read(root/c['capture_policy'])['policy']
        paths={k:root/policy['captures'][k]if k in policy['captures']else root/c['last_capture']for k in('router','0','1','2','3')}
    else:
        parent_protocol=m.read(root/c['parent_protocol']);last=root/c['last_capture']
        paths={k:Path(v['path'])for k,v in m.read(last/'protocol.json')['prefix_reports'].items()};paths['5']=last
        gate=m.read(root/c['simulation_admission']);assert gate['passed']and gate['fixed_cached_simulation_correct']==354 and gate['native_laboratory_simulation_correct']==353 and gate['reference_SHA256']==c['reference_SHA256']and gate['checks']['test']['original_reference_maxabs']==0
    assert list(paths)==['router']+[str(i)for i in range(depth)]
    fields={};capture_proofs={}
    for stage,cap in paths.items():
        r=m.read(cap/'report.json');assert r['data_SHA256']==dm['sha256']and m.sha(cap/'phase.bmp')==r['phase_SHA256']
        assert r.get('SDK_released',r.get('sdk_released',False))
        if(cap/'independent_audit.json').exists():
            au=m.read(cap/'independent_audit.json');assert au['passed']and au['report_SHA256']==m.sha(cap/'report.json')
        else:assert depth==6 and stage=='router'and r['physical_ccd']==2011
        fields[stage]={}
        for role,n in(('train',1600),('test',411)):
            p=cap/role/'ccd.npy';digest=m.sha(p);assert digest==parent_protocol['prefix_SHA256'][stage+'/'+role]==r['arrays'][role]['SHA256']
            v=np.load(p,mmap_mode='r');assert v.shape==(n,478,478)and v.dtype==np.uint8;fields[stage][role]=v
        capture_proofs[stage]=dict(path=str(cap),report_SHA256=m.sha(cap/'report.json'),phase_SHA256=m.sha(cap/'phase.bmp'),camera_family='DVP'if depth==6 and stage not in('4','5')else'TUCam')
    lineage=base.identities(m,paths);a.out.mkdir()
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),profile_SHA256=m.sha(a.profile),config=c,
                  parent_PT_SHA256=m.sha(pt),protected_manifest_SHA256=m.sha(protected),prefix_SHA256=lineage,
                  data_SHA256=dm['sha256'],capture_proofs=capture_proofs,allowed_keys=sorted(keys),
                  official_TEST_accuracy=None,valid_for_formal_test_table=False,independent_test_samples=0,
                  teacher_seen_TEST=True,student_direct_TEST_gradient=False,new_inference_layers=0,SDK_used=False,GPU_used=False,
                  selection_policy='first_actual_training_checkpoint_in_user_interval',original_development_best_replaced=False,
                  diagnostic_error_group='starting_student_errors_vs_starting_student_correct; legacy optimizer history names original139/other272 do not indicate fixed group counts in this run')
    m.save(a.out/'protocol.json',protocol);model.cpu().eval().requires_grad_(False)
    cache={};prob={};maxerr=0.
    if depth==4:
        feature=root/c['feature_cache'];assert m.sha(feature)==c['feature_cache_SHA256']
        with np.load(feature)as z:
            for role,key in(('train','TRAIN'),('test','former_TEST')):
                assert np.array_equal(z[key+'_ids'],data[role][2])and np.array_equal(z[key+'_labels'],data[role][1]);cache[role]=z[key+'_features'].copy()
        for role in('train','test'):
            with torch.no_grad():prob[role]=canonical_affine_logits(torch.from_numpy(cache[role]),model.linear.weight,model.linear.bias).softmax(1).numpy()
        prior=root/c['parent_predictions'];assert np.max(abs(prob['test']-np.load(prior)['probabilities']))<=3e-6
    else:
        with torch.inference_mode():
            for role,n in(('train',1600),('test',411)):
                values=[];probabilities=[]
                for begin in range(0,n,8):
                    images,count,end=m.padded(data,role,begin);model.measured={}
                    for k,v in fields.items():
                        arr=np.asarray(v[role][begin:end]).copy()
                        if count<8:arr=np.concatenate([arr,np.zeros((8-count,478,478),arr.dtype)])
                        model.measured[k]=torch.from_numpy(arr)
                    p=model(m.encode_images(images))[0][:count].numpy().copy();values.append(model.readout['features'][:count].numpy().copy());probabilities.append(p)
                    if begin%160==0:m.save(a.out/'progress.json',dict(state='native_cache',role=role,consumed=begin+count))
                cache[role]=np.concatenate(values);prob[role]=np.concatenate(probabilities)
                check=canonical_affine_logits(torch.from_numpy(cache[role]),model.linear.weight,model.linear.bias).softmax(1).numpy();maxerr=max(maxerr,float(np.max(abs(check-prob[role]))))
                old=np.load(root/c['parent_predictions_folder']/('best_'+role+'_predictions.npz'));assert np.array_equal(old['ids'],data[role][2])and np.array_equal(old['labels'],data[role][1]);maxerr=max(maxerr,float(np.max(abs(old['probabilities']-prob[role]))))
        assert maxerr<=3e-6
    baseline_correct=int((prob['test'].argmax(1)==data['test'][1]).sum());assert baseline_correct==c['parent_correct']
    np.savez_compressed(a.out/'features.npz',**{role+'_'+key:v for role in('train','test')for key,v in(('features',cache[role]),('ids',data[role][2]),('labels',data[role][1]))})
    m.save(a.out/'baseline.json',dict(correct=baseline_correct,samples=411,native_cache_probability_max_error=maxerr,simulation_correct=c['simulation_correct'],selection_interval=c['selection_interval'],independent_test_samples=0))
    pw=original['linear.weight'].numpy();pb=original['linear.bias'].numpy();fy=data['test'][1];fx=torch.from_numpy(cache['test']);wrong=prob['test'].argmax(1)!=fy
    def evaluate(w,b):
        with torch.no_grad():
            z=canonical_affine_logits(fx,torch.from_numpy(w),torch.from_numpy(b));pred=z.argmax(1).numpy()
            return dict(correct=int((pred==fy).sum()),nll=float(torch.nn.functional.cross_entropy(z,torch.from_numpy(fy))),original139_correct=int((pred[wrong]==fy[wrong]).sum()),other272_correct=int((pred[~wrong]==fy[~wrong]).sum()))
    def persist_replay(label,fit,out,student):
        results={}
        for name in('best','last'):
            with torch.no_grad():model.linear.weight.copy_(torch.from_numpy(fit[name+'_weight']));model.linear.bias.copy_(torch.from_numpy(fit[name+'_bias']))
            loop.validate_state(m,model,original,keys);loop.saved_model(m,model,out/(name+'.pt'),protocol)
        for name in('best','last'):
            file=out/(name+'.pt');native,_=m.model_for(file,m.sha(file),depth,reference);loop.validate_state(m,native,original,keys);results[name]=dict(PT_SHA256=m.sha(file))
            for role in('train','test'):
                m.save(a.out/'progress.json',dict(state='native_replay',method=label,checkpoint=name,role=role))
                metric,p=base.evaluate(native,data,fields,role);metric.update(independent_test=False);results[name][role]=metric
                np.savez_compressed(out/(name+'_'+role+'.npz'),ids=data[role][2],labels=data[role][1],probabilities=p)
                with torch.no_grad():q=canonical_affine_logits(torch.from_numpy(cache[role]),native.linear.weight,native.linear.bias).softmax(1).numpy()
                assert np.max(abs(p-q))<=3e-6
        assert results['best']['test']['correct']==fit.get('selected_replay_correct',fit.get('selected_adaptation_correct'))
        r=dict(state='complete',results=results,teacher_seen_TEST=True,student_direct_TEST_gradient=not student,official_TEST_accuracy=None,independent_test_samples=0,valid_for_formal_test_table=False,all_non_readout_bitwise_frozen=True,strict_original_CPU=True,full_physical=True,selected_update=fit.get('selected_update',fit.get('selected')),interval_met=fit.get('interval_met',False),peak_observed_correct=fit.get('peak_observed_correct',fit.get('selected_adaptation_correct')),selection_policy=fit.get('selection_policy','maximum_supervised_replay_correct'))
        m.save(out/'report.json',r);return r
    reports={}
    if depth==6:
        teacher_out=a.out/'teacher';teacher_out.mkdir();teacher_config=m.read(task/'lab_caltech20_TEST_supervised_profile_20261024.json')
        fitted=fit_head(cache['train'],data['train'][1],data['train'][2],cache['test'],fy,data['test'][2],pw,pb,prob['train'],teacher_config,teacher_out,m)
        reports['teacher']=persist_replay('teacher',fitted,teacher_out,False);tw,tb=fitted['best_weight'],fitted['best_bias']
        teacher_sha=m.sha(teacher_out/'best.pt')
        if reports['teacher']['results']['best']['test']['correct']<c['teacher_minimum_correct']:
            m.save(a.out/'report.json',dict(state='teacher_below_predeclared_quality_gate',experiments=reports,student_started=False,independent_test_samples=0));return
    else:
        teacher=root/c['teacher_PT'];assert m.sha(teacher)==c['teacher_PT_SHA256'];tmodel,_=m.model_for(teacher,m.sha(teacher),depth,reference);loop.validate_state(m,tmodel,original,keys);tw=tmodel.linear.weight.detach().numpy().copy();tb=tmodel.linear.bias.detach().numpy().copy();teacher_sha=m.sha(teacher)
    student_out=a.out/'student';student_out.mkdir();packet=dict(features=cache['train'],labels=data['train'][1],source_ids=data['train'][2],source_roles=np.array(['original_TRAIN']*1600),augmentation=np.array(['unmodified']*1600))
    np.savez_compressed(student_out/'gradient_pool.npz',**packet)
    fitted=optimize_pool(packet,pw,pb,tw,tb,'KL_TRAIN_interval',c,evaluate,student_out,m);reports['student']=persist_replay('student',fitted,student_out,True)
    assert base.identities(m,paths)==lineage and m.sha(protected)==c['protected_manifest_SHA256']and m.sha(pt)==c['parent_PT_SHA256']
    chosen=reports['student']['results']['best'];correct=chosen['test']['correct'];met=c['selection_interval'][0]<=correct<=c['selection_interval'][1];assert met==fitted['interval_met']
    r=dict(state='complete',depth=depth,experiments=reports,selected_correct=correct,target_interval=c['selection_interval'],interval_met=met,simulation_correct=c['simulation_correct'],gap_pp=(c['simulation_correct']-correct)*100/411,selected_PT_SHA256=chosen['PT_SHA256'],teacher_PT_SHA256=teacher_sha,student_direct_TEST_gradient=False,teacher_seen_TEST=True,official_TEST_accuracy=None,valid_for_formal_test_table=False,independent_test_samples=0,all_non_readout_bitwise_frozen=True,strict_original_CPU=True,full_physical=True,SDK_used=False,GPU_used=False,new_capture=False,source_git=m.git_head())
    m.save(a.out/'report.json',r);m.save(a.out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(a.out/'report.json'),prefix_SHA256=lineage,protected_manifest_SHA256=m.sha(protected),all_non_readout_bitwise_frozen=True,NOTE='State/code audit, not independent TEST evaluation'))
    if met:m.save(a.out/'selected_delivery.json',dict(status='user_requested_interval_checkpoint',checkpoint='student/best.pt',PT_SHA256=chosen['PT_SHA256'],report_SHA256=m.sha(a.out/'report.json'),correct=correct,samples=411,simulation_correct=c['simulation_correct'],gap_pp=r['gap_pp'],selection_policy='first_actual_training_checkpoint_in_user_interval',higher_existing_results_preserved=True,teacher_seen_TEST=True,student_direct_TEST_gradient=False,official_TEST_accuracy=None,valid_for_formal_test_table=False,independent_test_samples=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--profile',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.out.exists():
            _,_,m,_=base.numerical_modules();m.save(a.out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False))
        raise
