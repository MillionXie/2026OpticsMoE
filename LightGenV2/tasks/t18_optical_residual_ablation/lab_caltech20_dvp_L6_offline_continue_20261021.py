"""Camera-paused L6: optimize only the unmeasured suffix after current E2.

Reuses the tested suffix trainer, never its capture entry point. Every existing
measured optical prefix stays byte-exact. New suffix PT requires fresh downstream
CCD if the human later resumes hardware; all metrics here remain mixed.
"""
import argparse,json,os,traceback
from pathlib import Path


def run(a):
    import torch,numpy as np
    import lab_caltech20_dvp_layerwise_20261019 as m
    import lab_caltech20_dvp_closedloop_20261021 as loop
    from lab_caltech20_dvp_suffix_stable_20261021 import load_prefix
    import lab_caltech20_dvp_L6_resume_G2_20261021 as engine
    torch.set_num_threads(2);torch.manual_seed(73);cfg=m.read(a.profile)
    assert cfg['schema']=='L6_camera_paused_extended_suffix_v1' and cfg['camera_paused'] and os.environ.get('CUDA_VISIBLE_DEVICES')==cfg['GPU_UUID']
    assert not a.out.exists() and m.sha(engine.__file__)==cfg['suffix_engine_SHA256']
    base=m.read(Path(__file__).with_name(cfg['prefix_profile']));data,dm,fields,identities,old,reference,_=load_prefix(a,m,np,base)
    parent=a.sequence/cfg['parent_relative'];report=m.read(parent.parent/'report.json');audit=m.read(parent.parent/'independent_audit.json')
    assert m.sha(parent)==cfg['parent_SHA256']==report['results']['best']['PT_SHA256'] and report['selected_correct']==205 and not report['full_physical']
    assert audit['passed'] and audit['report_SHA256']==m.sha(parent.parent/'report.json') and identities==audit['prefix_SHA256']
    model,_=m.model_for(parent,cfg['parent_SHA256'],6,reference);loop.validate_state(m,model,m.state_cpu(old),m.allowed_keys(old,4))
    case=a.sequence/base['source_run'];session=a.sequence/'L6_response_20261020'
    paths={'router':session/'capture/router','0':a.sequence/'TRAIN_restored_same_mask_pair_20261021/L6/capture','1':case/'L6/stage_1/capture','2':case/'L6/stage_2/capture'}
    chosen,best=engine.suffix(m,loop,data,dm,fields,model,dict(cfg,_paths={k:str(v) for k,v in paths.items()}),'2',a.out,reference)
    report=m.read(a.out/'report.json');assert report['strict_original_CPU'] and not report['full_physical']
    assert m.read(a.out/'independent_audit.json')['passed']
    m.save(a.out/'offline_scope_audit.json',dict(wrapper_git=m.git_head(),wrapper_SHA256=m.sha(__file__),profile_SHA256=m.sha(a.profile),suffix_engine_SHA256=m.sha(engine.__file__),
        camera_paused_by_human=True,camera_started=False,new_CCD=0,SDK_used=False,selected_correct=best,measured=['router','E1','G1','E2'],simulated=['G2','E3','G3'],
        only_unmeasured_masks_and_postE2_electronics_trainable=True,new_downstream_CCD_required=True,original_bestlast_TRAIN1600_TEST411=True))
    m.save(a.out/'next_capture_handoff.json',dict(state='camera_paused_until_new_explicit_human_authorization',PT=str(a.out/'best.pt'),PT_SHA256=m.sha(a.out/'best.pt'),prefix_SHA256=identities,full_physical=False,new_downstream_CCD_required=True,camera_started=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequence',type=Path,required=True);p.add_argument('--profile',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.out.exists():(a.out/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False,camera_started=False),indent=2))
        raise
