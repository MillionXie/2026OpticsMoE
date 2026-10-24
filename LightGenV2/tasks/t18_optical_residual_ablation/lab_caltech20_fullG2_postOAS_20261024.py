"""Finite original5150 continuation after TRAIN covariance readout correction."""
import argparse,traceback
from pathlib import Path
from datetime import datetime,timezone
import lab_caltech20_tucam_layerwise_20261023 as base


def run(a):
    np,torch,m,loop=base.numerical_modules();torch.set_num_threads(2)
    if a.renewed_offline:
        assert a.out.name=='fullG2_joint_renewed_20261024'
    else:
        assert (datetime.fromisoformat('2026-10-24T08:00:00+08:00')-datetime.now(timezone.utc)).total_seconds()>=240
    assert not a.out.exists() and not m.LOCK.exists()
    prefix=base.ROOT/'LightGenV2/tasks/t18_optical_residual_ablation/runs/hardware/caltech20_residual_TUCam_20261022/layerwise_L4_on_20261023/e2_actual300_20261024';source=prefix/'fullG2_readout_OAS';capture_run=prefix/'readout_G2PASS'
    assert m.read(source.with_suffix('.return.json'))['returncode']==0
    sr=m.read(source/'report.json');sa=m.read(source/'independent_audit.json');sp=m.read(source/'protocol.json');assert sr['full_physical'] and sr['strict_original_CPU'] and sa['passed'] and sa['report_SHA256']==m.sha(source/'report.json')
    pt=source/'best.pt';assert m.sha(pt)==sr['results']['best']['PT_SHA256'] and sr['selected_correct']>=sr['baseline_correct']
    cp=base.HERE/'lab_caltech20_tucam_layerwise_profile_20261023.json';assert m.sha(cp)=='c077c32638e577b1c86847b5ff7cdd40ab3ede675c9b4299c0697e1b389ec909';c=m.read(cp);base.validate_config(c)
    reference=base.ROOT/c['reference'];assert m.sha(reference)==c['reference_SHA256'];model,_=m.model_for(pt,m.sha(pt),4,reference)
    captured=Path(m.read(capture_run/'s_3/fit/protocol.json')['parent_PT']);assert m.sha(captured)=='60f6883ac597fbe3a2f72000edac661998dd6fb52696ba80cb6b8bcce53f115e';original,_=m.model_for(captured,m.sha(captured),4,reference);loop.validate_state(m,model,m.state_cpu(original),m.allowed_keys(model,5));del original
    data,dm=m.data_for(base.ROOT/c['sequence']);assert dm['sha256']==c['data_SHA256'];policy=m.read(capture_run/'protocol.json')['policy'];fields={};paths={}
    for stage in base.STAGES:
        cap=base.ROOT/policy['captures'][stage] if stage in policy['captures'] else capture_run/'s_3/capture';r=m.read(cap/'report.json');audit=m.read(cap/'independent_audit.json')
        assert audit['passed'] and audit['matched_samples']==2011 and audit['report_SHA256']==m.sha(cap/'report.json') and r['data_SHA256']==dm['sha256'] and r['SDK_released']
        assert m.sha(cap/'phase.bmp')==r['phase_SHA256'];fields[stage]={};paths[stage]=cap
        for role,n in (('train',1600),('test',411)):
            f=cap/role/'ccd.npy';assert m.sha(f)==sp['prefix_SHA256'][stage+'/'+role]==r['arrays'][role]['SHA256'];v=np.load(f,mmap_mode='r');assert v.shape==(n,478,478) and v.dtype==np.uint8;fields[stage][role]=v
    assert sum(v.numel()for name,v in model.named_parameters()if name in m.allowed_keys(model,5))==5150
    if not a.renewed_offline:
        assert (datetime.fromisoformat('2026-10-24T08:00:00+08:00')-datetime.now(timezone.utc)).total_seconds()>=225
    a.out.mkdir();m.save(a.out/'admission.json',dict(passed=True,source_git=m.git_head(),source_SHA256=m.sha(__file__),parent_PT_SHA256=m.sha(pt),parent_report_SHA256=m.sha(source/'report.json'),source_correct=sr['selected_correct'],full_physical=True,prefix_SHA256=base.identities(m,paths),SDK_used=False,new_inference_layers=0))
    # Original first-fit TRAIN514/1600 demonstrated underfit. Recalibrated head
    # supplies a new initialization; accelerate only lawful final fusion/local.
    config={**c,'head_lr':.0001,'electronic_lr':.0003,'alpha_lr':.002}
    recipe=dict(name='full_physical_post_TRAIN_OAS_original5150_finite_continue',epochs=12,seconds=120,lr_scale=1,TRAIN_class_emphasis=.75)
    if a.renewed_offline:
        recipe=dict(name='human_renewed_offline_joint5150',epochs=80,seconds=600,lr_scale=1,TRAIN_class_emphasis=.75)
        m.save(a.out/'current_authorization.json',dict(previous_window_preserved=True,renewed_user_request='2026-10-24 offline final electronics and fusion only',epochs_max=80,fit_seconds_max=600,recurring=False,SDK_used=False,recapture=False,parameter_count=5150,gradient_data='TRAIN only',TEST_development=True))
    model,pt,result=base.fit_suffix(model,data,dm,fields,paths,pt,reference,config,recipe,a.out/'fit',340)
    m.save(a.out/'report.json',dict(state='complete',full_physical=True,selected_correct=result['selected_correct'],source_correct=sr['selected_correct'],target_met=result['target_met'],strict_original_CPU=True,PT_SHA256=m.sha(pt),SDK_released=not m.LOCK.exists(),TEST_development=True,TEST_gradient=False,new_inference_layers=0))


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--out',type=Path,required=True);q.add_argument('--renewed-offline',action='store_true');a=q.parse_args()
    try:run(a)
    except BaseException as e:
        if a.out.exists():
            _,_,m,_=base.numerical_modules();m.save(a.out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False,all_data_preserved=True))
        raise
