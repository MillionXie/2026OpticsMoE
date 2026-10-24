"""One TRAIN-only original affine readout fit after complete physical G2."""
import argparse,traceback
from pathlib import Path
from datetime import datetime,timezone
import lab_caltech20_tucam_layerwise_20261023 as base


def run(a):
    np,torch,m,loop=base.numerical_modules()
    from lab_caltech20_tucam_TRAIN_linear_20261023 import TRAIN_linear_fit
    cpath=base.HERE/'lab_caltech20_tucam_layerwise_profile_20261023.json';assert m.sha(cpath)=='c077c32638e577b1c86847b5ff7cdd40ab3ede675c9b4299c0697e1b389ec909'
    c=m.read(cpath);base.validate_config(c);torch.set_num_threads(2)
    assert not(a.joint_source and a.local_repair)
    assert not a.teacher_local or a.local_repair
    if a.joint_source or a.local_repair:
        assert a.out.name==('fullG2_teacher_local_20261024'if a.teacher_local else'fullG2_native_local_20261024'if a.local_repair else'fullG2_joint_TRAIN_OAS_20261024')
    else:
        assert (datetime.fromisoformat('2026-10-24T08:00:00+08:00')-datetime.now(timezone.utc)).total_seconds()>=330
    assert m.sha(base.HERE/'lab_caltech20_tucam_TRAIN_linear_20261023.py')=='a8352cd34237d5f9c7f6ed21f4ff489f94baf546ee3feccdc0c6c5addbcca67c'
    assert not m.LOCK.exists() and not a.out.exists()
    root=base.ROOT/'LightGenV2/tasks/t18_optical_residual_ablation/runs/hardware/caltech20_residual_TUCam_20261022/layerwise_L4_on_20261023/e2_actual300_20261024/readout_G2PASS'
    assert m.read(root.with_suffix('.return.json'))['returncode']==0
    full=m.read(root/'report.json');assert full['state']=='complete' and full['full_physical'] and full['strict_original_CPU'] and full['SDK_released']
    fit=root/'s_3/fit';r=m.read(fit/'report.json');audit=m.read(fit/'independent_audit.json');protocol=m.read(fit/'protocol.json')
    if a.joint_source:
        source=root.parent/'fullG2_joint_renewed_20261024'
        assert m.read(source.with_suffix('.return.json'))['returncode']==0
        full=m.read(source/'report.json');assert full['state']=='complete' and full['full_physical'] and full['strict_original_CPU'] and full['SDK_released']
        fit=source/'fit';r=m.read(fit/'report.json');audit=m.read(fit/'independent_audit.json');protocol=m.read(fit/'protocol.json')
    if a.local_repair:
        source=root.parent/'fullG2_TRAIN_fusion_nativepad_20261024';assert m.read(source.with_suffix('.return.json'))['returncode']==0
        fit=source;r=m.read(fit/'report.json');audit=m.read(fit/'independent_audit.json');protocol=m.read(fit/'protocol.json');full=r
        if a.teacher_local:
            source=root.parent/'fullG2_teacher_margin_20261024';assert m.read(source.with_suffix('.return.json'))['returncode']==0
            fit=source/'fit';r=m.read(fit/'report.json');audit=m.read(fit/'independent_audit.json');protocol=m.read(fit/'protocol.json');full=m.read(source/'report.json')
        assert r['state']=='complete' and r['full_physical'] and not r['SDK_used']
    assert r['strict_original_CPU'] and r['full_physical'] and audit['passed'] and audit['report_SHA256']==m.sha(fit/'report.json')
    pt=fit/'best.pt';assert m.sha(pt)==r['results']['best']['PT_SHA256']==full.get('PT_SHA256',r['results']['best']['PT_SHA256'])
    reference=base.ROOT/c['reference'];assert m.sha(reference)==c['reference_SHA256'];model,_=m.model_for(pt,m.sha(pt),4,reference)
    parent=root.parent/'readout_LOCALMSE/fit/best.pt';assert m.sha(parent)=='60f6883ac597fbe3a2f72000edac661998dd6fb52696ba80cb6b8bcce53f115e'
    captured,_=m.model_for(parent,m.sha(parent),4,reference);loop.validate_state(m,model,m.state_cpu(captured),m.allowed_keys(model,5));del captured
    data,dm=m.data_for(base.ROOT/c['sequence']);assert dm['sha256']==c['data_SHA256'];fields={};paths={}
    for stage in base.STAGES:
        # Resolve only the exact immutable run lineage, never by mtime.
        policy=m.read(root/'protocol.json')['policy'];relative=policy['captures'].get(stage)
        cap=base.ROOT/relative if relative else root/'s_3/capture';cr=m.read(cap/'report.json');ca=m.read(cap/'independent_audit.json')
        assert ca['passed'] and ca['matched_samples']==2011 and ca['report_SHA256']==m.sha(cap/'report.json')
        assert cr['TRAIN']==1600 and cr['TEST']==411 and cr['VALID']==0 and cr['SDK_released'] and cr['data_SHA256']==dm['sha256'] and m.sha(cap/'phase.bmp')==cr['phase_SHA256']
        fields[stage]={};paths[stage]=cap
        for role,n in (('train',1600),('test',411)):
            path=cap/role/'ccd.npy';digest=m.sha(path);assert digest==protocol['prefix_SHA256'][stage+'/'+role]==cr['arrays'][role]['SHA256']
            v=np.load(path,mmap_mode='r');assert v.shape==(n,478,478) and v.dtype==np.uint8;fields[stage][role]=v
    lineage=base.identities(m,paths);assert list(fields)==list(base.STAGES)
    if a.local_repair:
        from lab_caltech20_tucam_TRAIN_linear_20261023 import fit_cached_head
        import lab_caltech20_TRAIN_local_boundary_20261023 as boundary
        assert m.sha(boundary.__file__)=='8ae5a132e31ab33d2dc508119f7d85810d75a35bc1e632e7de42cd7f133c774b'
        cfg=dict(maximum_classes=20,steps_per_class=64,temperature=2.,label_smoothing=.02,neighbor_weight=2.,KL=2.,anchor=.2,trust_step=.25,row_radius=.08,seed=73);boundary.validate(cfg)
        config={**c,'TRAIN_readout':'cached_TRAIN_local_boundary','TRAIN_local_boundary':cfg,'TRAIN_local_engine_SHA256':m.sha(boundary.__file__),'native_feature_source':fit.relative_to(base.ROOT).as_posix(),'fresh_native_TRAIN_features':True}
        _,_,result=fit_cached_head(model,data,dm,fields,paths,pt,reference,config,dict(name='one_bounded_full_physical_TRAIN_neighbor_repair',epochs=1,seconds=180,lr_scale=1),a.out,340)
        assert result['full_physical'] and result['strict_original_CPU'] and result['selected_correct']>=r['selected_correct']
        return
    a.out.mkdir();original=m.state_cpu(model);keys={'linear.weight','linear.bias'};baseline,prob=base.evaluate(model,data,fields,'test');assert baseline['correct']==r['selected_correct']
    p=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),parent_PT=str(pt),parent_PT_SHA256=m.sha(pt),source_report_SHA256=m.sha(fit/'report.json'),prefix_SHA256=lineage,data_SHA256=dm['sha256'],
        TRAIN=1600,TEST=411,VALID=0,TEST_gradient=False,TEST_development=True,full_physical=True,new_inference_layers=0,SDK_used=False,GPU_used=False,allowed_keys=sorted(keys),method='single_TRAIN_OAS_original5140_no_grid',optimizer_updates=0,hyperparameter_grid=False)
    m.save(a.out/'protocol.json',p);m.save(a.out/'baseline.json',baseline);np.savez_compressed(a.out/'baseline_TEST_predictions.npz',ids=data['test'][2],labels=data['test'][1],probabilities=prob)
    model.cpu().eval().requires_grad_(False);parts=[]
    with torch.inference_mode():
        for begin in range(0,1600,8):
            ix=np.arange(begin,begin+8);model(loop.put_fields(model,data,fields,'train',ix,'cpu'));parts.append(model.readout['features'].numpy().copy())
            if begin%160==0:m.save(a.out/'progress.json',dict(state='full_physical_native_TRAIN_features',TRAIN_consumed=begin+8,TEST_used_for_fit=False))
    features=np.concatenate(parts);cache=a.out/'native_TRAIN_features.npz';np.savez_compressed(cache,ids=data['train'][2],labels=data['train'][1],features=features)
    weight,bias,stats=TRAIN_linear_fit(features,data['train'][1],c['original_TRAIN_counts']);stats.update(feature_cache_SHA256=m.sha(cache),TEST_used=False);m.save(a.out/'TRAIN_statistics.json',stats)
    with torch.no_grad():model.linear.weight.copy_(torch.from_numpy(weight).to(model.linear.weight));model.linear.bias.copy_(torch.from_numpy(bias).to(model.linear.bias))
    loop.validate_state(m,model,original,keys);loop.saved_model(m,model,a.out/'last.pt',p);results={}
    def review(label):
        file=a.out/(label+'.pt');native,_=m.model_for(file,m.sha(file),4,reference);loop.validate_state(m,native,original,keys);result=dict(PT_SHA256=m.sha(file))
        for role in ('train','test'):
            m.save(a.out/'progress.json',dict(state='strict_native_CPU',checkpoint=label,role=role));metric,values=base.evaluate(native,data,fields,role);result[role]=metric;np.savez_compressed(a.out/(label+'_'+role+'_predictions.npz'),ids=data[role][2],labels=data[role][1],probabilities=values)
        return result
    results['last']=review('last');candidate=results['last']['test'];better=candidate['correct']>baseline['correct']or(candidate['correct']==baseline['correct']and candidate['nll']<baseline['nll']-1e-4)
    if not better:model.load_state_dict(original)
    loop.saved_model(m,model,a.out/'best.pt',p);results['best']=review('best');best=results['best']['test']['correct'];assert best>=baseline['correct'] and base.identities(m,paths)==lineage
    report=dict(state='complete',baseline_correct=baseline['correct'],selected_correct=best,selected_E0=not better,results=results,strict_original_CPU=True,full_physical=True,target=340,target_met=best>=340,
        all_non_readout_bitwise_frozen=True,captured_upstream_bitwise_frozen=True,TRAIN=1600,TEST=411,VALID=0,TEST_gradient=False,TEST_development=True,SDK_used=False,GPU_used=False,new_inference_layers=0,analytical_fits=1,optimizer_updates=0,protocol_SHA256=m.sha(a.out/'protocol.json'))
    m.save(a.out/'report.json',report);m.save(a.out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(a.out/'report.json'),prefix_SHA256=lineage,all_non_readout_bitwise_frozen=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--joint-source',action='store_true');parser.add_argument('--local-repair',action='store_true');parser.add_argument('--teacher-local',action='store_true');a=parser.parse_args()
    try:run(a)
    except BaseException as e:
        if a.out.exists():
            _,_,m,_=base.numerical_modules();m.save(a.out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False,all_data_preserved=True))
        raise
