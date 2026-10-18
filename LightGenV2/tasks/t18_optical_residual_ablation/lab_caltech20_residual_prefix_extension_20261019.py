"""Finite continuation of an original measured-prefix optical suffix, no SDK.

Reuse the existing fit algorithm without its old CLI's depth4 ceiling. Admission
proves the depth6/prefix4 graph, frozen observed prefix and native incident data.
New PTs remain MIXED metrics and cannot be replayed on old downstream CCDs.
"""
import argparse,json,os,sys,traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import lab_caltech20_rich_layerwise_20261018 as core
import lab_caltech20_residual_sequence_20261018 as seq
from lab_caltech20_rich_model_20261018 import sha
from lab_caltech10_20261015 import phase_native,save


class PrefixComplete(Exception):pass


def selftest():
    torch.set_num_threads(2);torch.manual_seed(73);m=core.MeasuredPrefix(6).eval()
    with torch.no_grad():
        for p in m.fusion_logits:p.fill_(.7)
        x=torch.rand(2,224,224);p=m(x)[0];fields={k:v['intensity'].clone() for k,v in m.last.items()}
    q=core.MeasuredPrefix(6).eval();q.load_state_dict(m.state_dict());q.measured={k:fields[k] for k in ('router','0','1','2')};q.requires_grad_(False);allowed=core.allowed_keys(q,4);future={f'phase_sets.{i}' for lo,hi in q.stage_slices[3:] for i in range(lo,hi)}
    assert {n for n in allowed if n.startswith('phase_sets.')}==future
    assert 'raw_router' not in allowed and not any(f'electronic.{i}.local.weight' in allowed or f'fusion_logits.{i}' in allowed for i in (0,1))
    for n,v in q.named_parameters():v.requires_grad_(n in allowed)
    out=q(x)[0];assert torch.equal(out,p);(-out[:,0].log().mean()).backward();assert sum(v.numel() for n,v in q.named_parameters() if n in allowed)==662852
    assert all(v.grad is None for n,v in q.named_parameters() if n not in allowed)
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for n,v in q.named_parameters() if n in allowed)
    assert sum(float(dict(q.named_parameters())[n].grad.abs().sum()) for n in future)>0
    print(json.dumps(dict(passed=True,depth6_prefix4_exact_parity=True,TRAIN_gradient_parameters=662852,frozen_router_and_first3_optical_stages=True,SDK_opened=False,GPU_used=False)))


def run(a):
    assert a.depth==6 and a.prefix==4 and not a.out.exists();torch.set_num_threads(2);cfg=core.read(a.profile);assert cfg['TRAIN']==1200 and cfg['TEST']==411 and cfg['VALID']==0 and cfg['mask']['seconds']<=1200 and cfg['mask']['epochs']<=120
    assert not torch.cuda.is_initialized();os.environ['CUDA_VISIBLE_DEVICES']=cfg['GPU_UUID']
    parent=core.read(a.parent_report);assert parent['state']=='complete' and parent['full_CPU_replay_passed'] and parent['results']['best']['PT_SHA256']==a.checkpoint_sha and parent['selected_correct']==171
    cfg0,data,dm,model,original,canonical,fields,identities=seq.dataset_model(a);assert list(fields)==['router','0','1','2'];saved=np.load(a.parent_report.parent/'best_TEST_predictions.npz');assert np.array_equal(saved['ids'],data['test'][2]) and np.array_equal(saved['labels'],data['test'][1]);metric,p,_=seq.assess(model,data,fields,canonical,'test',4);error=float(np.max(np.abs(p-saved['probabilities'])));assert error<=3e-6 and np.array_equal(p.argmax(1),saved['probabilities'].argmax(1))
    indices=np.concatenate([np.flatnonzero(data['train'][1]==k)[:3] for k in range(20)]);model.measured={};checked=set()
    with torch.inference_mode():
        for begin in range(0,60,4):
            ix=indices[begin:begin+4];calls=[]
            def replay(stage,amplitude,phase,ids):
                calls.append(stage)
                if stage not in fields:assert stage=='3';raise PrefixComplete()
                d=a.session/'capture'/stage;r=core.read(d/'report.json');assert np.array_equal(ids,data['train'][2][ix])
                if stage not in checked:
                    assert sha(d/'phase.bmp')==r['phase_sha256'] and np.array_equal(np.asarray(Image.open(d/'phase.bmp')),phase_native(phase));checked.add(stage)
                for j,index in enumerate(ix):
                    path=d/'train'/f'{index:06d}_input.png';v=core.read(d/'train'/f'{index:06d}.json');assert sha(path)==v['canonical_SHA256'] and v['sample_id']==str(ids[j]);assert np.array_equal(np.asarray(Image.open(path)),np.rint(amplitude[j].clip(0,1)*255).astype(np.uint8))
                return np.asarray(fields[stage]['train'][ix]).copy()
            try:model(seq.encode_images(data['train'][0][ix]),callback=replay,ids=data['train'][2][ix])
            except PrefixComplete:pass
            else:raise AssertionError('Expected stop before an unmeasured future stage')
            assert calls==['router','0','1','2','3']
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.items());g=core.read(a.session/'server_gate.json');cfg['canonical_reference_SHA256']=g['TorchScript_SHA256'];effective=a.out.with_suffix('.profile.json');assert not effective.exists();save(effective,cfg)
    admission=dict(state='admitted',depth=6,prefix=['router','0','1','2'],parent_PT_SHA256=a.checkpoint_sha,parent_report_SHA256=sha(a.parent_report),baseline_mixed_TEST_correct=round(metric['accuracy']*411),prefix_CPU_probability_error=error,TRAIN60_native_phase_and_canonical_identity_passed=True,original_data_SHA256=dm['sha256'],captured_prefix_SHA256=identities,source_SHA256=sha(__file__),existing_fit_source_SHA256=sha(core.__file__),source_profile_SHA256=sha(a.profile),effective_profile_SHA256=sha(effective),SDK_opened=False,TEST_gradients=False,VALID=0,no_new_inference_layer=True,no_new_capture=True,old_downstream_CCDs_not_loaded=True);save(a.out.with_suffix('.admission.json'),admission)
    a.profile=effective;a.controller_receipt=None;a.controller_pid=None;a.device='cuda';core.fit(a)
    r=core.read(a.out/'report.json');assert r['state']=='complete' and r['frozen_parameters_unchanged'] and r['evaluation_kind']=='measured_prefix_simulated_suffix' and all(v['strict_CPU'] for v in r['results'].values())
    save(a.out/'continuation_receipt.json',dict(state='complete',admission_SHA256=sha(a.out.with_suffix('.admission.json')),fit_report_SHA256=sha(a.out/'report.json'),baseline_mixed_correct=admission['baseline_mixed_TEST_correct'],selected_mixed_correct=round(r['results']['best']['TEST_development']['accuracy']*411),selected_epoch=r['selected_epoch'],actual_epochs=r['actual_epochs'],PT_SHA256=r['results']['best']['PT_SHA256'],is_NOT_fullphysical_result=True,requires_fresh_downstream_CCD_before_fullphysical=True,original_fullphysical165_preserved=True,TEST_development=True,TEST_gradients=False,SDK_opened=False,new_CCD=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--depth',type=int,default=6);p.add_argument('--prefix',type=int,default=4);p.add_argument('--checkpoint-sha')
    for n in ('session','profile','parent-report','checkpoint','out'):p.add_argument('--'+n,type=Path)
    a=p.parse_args()
    if sys.stdout is None:sys.stdout=a.out.with_suffix('.log').open('x',encoding='utf-8',buffering=1);sys.stderr=sys.stdout
    try:selftest() if a.selftest else run(a)
    except BaseException as e:
        if a.out:save(a.out.with_suffix('.failure.json'),dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False))
        raise
