"""Finite TRAIN-only original readout fit with class-contrast regularization.

Softmax ignores a shared class row, so it must not set the discriminative L2.
No model layer, TEST training, camera, physical gate or data identity changes.
"""
import argparse,json
from pathlib import Path
import lab_caltech20_TRAIN_photometric_readout_20261024 as engine
import lab_caltech20_TRAIN_native_mask_20261024 as native
base=native.base

SOURCE='LightGenV2/tasks/t18_optical_residual_ablation/runs/hardware/caltech20_residual_TUCam_20261022/layerwise_L4_on_20261023/e2_native_mask_20261024/mask_fit'
REPORT='4e97eb9440886e363152f2b78afaa120112b7129d7dcbbeeb1b6d642c09fd4a6'
PT='0cf1e41fc11b9d4c24e95ec9079aee5a118373e9af5279ce0b33116267d72da2'
RECIPE=dict(analytical_fits=1,outer_steps=20,seconds=900,LBFGS_max_iter=5,LBFGS_max_eval=8,label_smoothing=.03,L2_parent_TRAIN_NLL_fraction=.1)


def selftest():
    import torch
    from lab_caltech20_tucam_TRAIN_linear_20261023 import TRAIN_affine_objective
    torch.set_num_threads(2);torch.manual_seed(73);x=torch.randn(12,6,dtype=torch.float64);common=torch.randn(1,6,dtype=torch.float64)*1000
    w=torch.randn(3,6,dtype=torch.float64)+common;b=torch.randn(3,dtype=torch.float64)+2000;wc=w-w.mean(0,keepdim=True);bc=b-b.mean()
    before=torch.nn.functional.linear(x,w,b).softmax(1);after=torch.nn.functional.linear(x,wc,bc).softmax(1);error=float((before-after).abs().max());assert error<1e-10
    nll=.7;lam=2*.1*nll/float(wc.square().sum());assert abs(.5*lam*float(wc.square().sum())-.1*nll)<1e-12
    wc.requires_grad_();bc.requires_grad_();y=torch.arange(12)%3;loss,_,_=TRAIN_affine_objective(wc,bc,x,y,torch.ones(3,dtype=torch.float64),.03,lam);loss.backward();assert all(torch.isfinite(v.grad).all() for v in (wc,bc))
    print(json.dumps(dict(passed=True,softmax_gauge_identity_error=error,L2_class_contrast_fraction_exact=True,finite_TRAIN_gradients=True,SDK_used=False,GPU_used=False,TEST_used=False,new_inference_layers=0)))


def run(a):
    np,torch,m,loop=base.numerical_modules()
    assert m.sha(engine.__file__)=='cf1ac658285b233df7ba5da435be39a5dd1c724c3322f3e84b7b622f1f74fc3e'
    assert m.sha(native.__file__)=='a356882c6ba5ce1757c7665712f7e8ed2a7aa7db870f7a27cdf7c861033e9302'
    source=base.ROOT/SOURCE;r=m.read(source/'report.json');audit=m.read(source/'independent_audit.json');parent_pt=source/'best.pt'
    assert m.read(source.parent.with_suffix('.return.json'))['returncode']==0 and not m.LOCK.exists()
    assert m.sha(source/'report.json')==audit['report_SHA256']==REPORT and audit['passed'] and r['strict_original_state_and_CPU']
    assert m.sha(parent_pt)==r['results']['best']['PT_SHA256']==PT and r['selected_predictive_correct']==258 and r['actual_epochs']==24
    assert m.read(source/'local_SHA_backup.json')['all_local_remote_SHA_matched']
    response=source.parents[1]/'e2_response_20261024/native_feature_fit';rr=m.read(response/'report.json')
    assert rr['native_response_admitted'] and m.sha(response/'report.json')=='d883bd75f4cbfb822aeb2375b71bad4e266b3ee370935ddd18563e1b3a02afc6'
    assert m.sha(response/'TRAIN_native_response.npz')=='2e8a1aa53f0acc4df02f4ef82609462fa2fba06e14709cce1168d9abb3c47e96'
    with np.load(response/'TRAIN_native_response.npz',allow_pickle=False) as z:g=torch.from_numpy(z['gain'].copy());b=torch.from_numpy(z['background'].copy());sigma=float(z['sigma_pixels'])
    p=m.read(a.profile);native.parent.validate_policy(p);assert m.sha(base.HERE/p['base_profile'])==p['base_profile_SHA256'];c=m.read(base.HERE/p['base_profile']);base.validate_config(c)
    assert not a.out.exists();a.out.mkdir();torch.set_num_threads(2);torch.manual_seed(73)
    old,data,dm,fields,paths,_,reference,_,_,_=native.parent.admit(p,c,a.out);model,_=m.model_for(parent_pt,PT,4,reference);loop.validate_state(m,model,m.state_cpu(old),m.allowed_keys(model,3))
    cfg=dict(schema='native_response_original5140_class_contrast_v1',mask_fit=SOURCE,mask_report_SHA256=REPORT,recipe=RECIPE,SDK_used=False,GPU_used=False,TEST_gradient=False,new_inference_layers=0)
    factory=lambda model,gain,bg:native.make_predictor(model,gain,bg,sigma)
    scorer=lambda model,gain,bg,data,fields,role:native.score(model,gain,bg,sigma,data,fields,role)
    report=engine.fit(model,data,dm,fields,paths,parent_pt,reference,g,b,c,cfg,a.out/'readout_fit',model_factory=factory,score_function=scorer,gauge_regularization=True)
    m.save(a.out/'report.json',dict(state='class_contrast_readout_complete',selected_predictive_correct=report['selected_predictive_correct'],not_physical_accuracy=True,
        actual_source_correct_still273=True,new_CCD=0,G2_not_started=True,SDK_released=not m.LOCK.exists(),next_requires_original_physical_E2_gate305=True))


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--profile',type=Path,required=True);q.add_argument('--out',type=Path,required=True);run(q.parse_args())
