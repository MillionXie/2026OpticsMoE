"""Read-only TRAIN response audit through the unchanged native nonlinearity.

No fitting, labels, TEST, camera, checkpoint writes, or forward-model updates.
Compare both the frozen original E2 CCD and later same-input response probes.
"""
import argparse,json,os,time
from pathlib import Path
import lab_caltech20_TRAIN_photometric_mask_20261024 as mask
base=mask.base


def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    np,torch,m,loop=base.numerical_modules()
    from torch.nn import functional as F
    from PIL import Image
    p=m.read(a.profile);mask.validate_policy(p)
    assert m.sha(mask.__file__)=='dfb6ab85b3344301fd88f83fe3fbd4e774c969b5ac8115172e9bffbe69ae3607'
    assert m.sha(base.__file__)==p['suffix_engine_SHA256']
    assert m.sha(base.HERE/'lab_caltech20_TRAIN_response_operator_20261024.py')==p['operator_SHA256']
    assert m.sha(base.HERE/p['base_profile'])==p['base_profile_SHA256']
    c=m.read(base.HERE/p['base_profile']);base.validate_config(c)
    assert not a.out.exists() and not m.LOCK.exists();a.out.mkdir();torch.set_num_threads(2);start=time.monotonic()
    model,data,dm,fields,paths,parent,reference,gain,bg,_=mask.admit(p,c,a.out)
    physical=m.read(base.ROOT/p['captures']['2']/'report.json');old=np.load(base.ROOT/p['captures']['2']/'train/ccd.npy',mmap_mode='r')
    assert m.sha(base.ROOT/p['captures']['2']/'train/ccd.npy')==physical['arrays']['train']['SHA256']
    probes=base.ROOT/p['response_fit'];probes=probes.parent/'probes';pr=m.read(probes/'report.json')
    assert m.sha(probes/'report.json')=='ed5f98770ce72fc4297d92f7125db0c41508af7bac22405b5dac0012e8973639' and not pr['TEST_used']
    rows=[v for v in pr['records'] if v['condition']=='nominal_before'];assert len(rows)==40 and len({v['index'] for v in rows})==40
    nominal,_=m.model_for(parent,p['source_PT_SHA256'],4,reference);predictor=mask.make_predictor(model,gain,bg).eval();model.eval();nominal.eval()
    all_fields={**fields,'2':{'train':old}};lineage=base.identities(m,paths)
    def cc(x,y):return float(m.correlation(np.asarray(x),np.asarray(y)))
    def metric(predicted,observed):
        pp=F.adaptive_avg_pool2d(predicted[None,None],(16,16))[0,0].numpy();oo=F.adaptive_avg_pool2d(observed[None,None],(16,16))[0,0].numpy()
        return dict(coarse16_PCC=cc(pp,oo),full478_PCC=cc(predicted.numpy(),observed.numpy()),native_optical_after_nonlinearity_PCC=cc(model.nonlinear(predicted[None],'global')[0].numpy(),model.nonlinear(observed[None],'global')[0].numpy()))
    results=[]
    with torch.inference_mode():
        for row in rows:
            ix=np.array([row['index']]);assert row['sample_id']==data['train'][2][ix[0]] and row['split'] in ('fit','held_TRAIN')
            assert row['phase_SHA256']==physical['phase_SHA256'] and m.sha(row['phase'])==physical['phase_SHA256']
            recent_path=Path(row['crop']);assert m.sha(recent_path)==row['crop_SHA256'];recent=torch.from_numpy(np.asarray(Image.open(recent_path)).astype(np.float32));recorded=torch.from_numpy(np.asarray(old[ix[0]]).astype(np.float32))
            nominal(loop.put_fields(nominal,data,fields,'train',ix,'cpu'));ni=nominal.last['2']['intensity'][0];nf=nominal.readout['features'][0].numpy().copy()
            predictor(loop.put_fields(predictor,data,fields,'train',ix,'cpu'));pi=predictor.last['2']['intensity'][0];pf=predictor.readout['features'][0].numpy().copy()
            canonical=Path(row['canonical']);assert m.sha(canonical)==row['canonical_SHA256']
            encoded=np.rint(predictor.last['2']['amplitude'][0].numpy().clip(0,1)*255).astype(np.uint8)
            assert np.array_equal(encoded,np.asarray(Image.open(canonical)))
            model(loop.put_fields(model,data,all_fields,'train',ix,'cpu'));actual_feature=model.readout['features'][0].numpy().copy()
            item=dict(index=row['index'],sample_id=row['sample_id'],split=row['split'],canonical_exact=True,
                nominal_vs_old=metric(ni,recorded),photometric_vs_old=metric(pi,recorded),nominal_vs_recent=metric(ni,recent),photometric_vs_recent=metric(pi,recent),
                same_input_old_vs_recent=metric(recorded,recent),old_recent_mean_ratio=float(recent.mean()/recorded.mean().clamp_min(1e-8)),
                final256_nominal_vs_old_PCC=cc(nf,actual_feature),final256_photometric_vs_old_PCC=cc(pf,actual_feature))
            results.append(item);m.save(a.out/'progress.json',dict(state='TRAIN_native_response_feature_audit',TRAIN_anchors=len(results),expected=40,TEST_used=False,SDK_used=False,elapsed_s=time.monotonic()-start))
    summary={}
    for split in ('fit','held_TRAIN'):
        selected=[v for v in results if v['split']==split];assert len(selected)==20
        summary[split]={}
        for key in ('nominal_vs_old','photometric_vs_old','nominal_vs_recent','photometric_vs_recent','same_input_old_vs_recent'):
            summary[split][key]={metric:float(np.mean([v[key][metric] for v in selected])) for metric in selected[0][key]}
        for key in ('final256_nominal_vs_old_PCC','final256_photometric_vs_old_PCC','old_recent_mean_ratio'):summary[split][key]=float(np.mean([v[key] for v in selected]))
    assert base.identities(m,paths)==lineage and not m.LOCK.exists()
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),profile_SHA256=m.sha(a.profile),source_PT_SHA256=m.sha(parent),data_SHA256=dm['sha256'],prefix_SHA256=lineage,
        old_E2_TRAIN_SHA256=physical['arrays']['train']['SHA256'],probe_report_SHA256=m.sha(probes/'report.json'),response_report_SHA256=p['response_report_SHA256'],response_map_SHA256=p['response_map_SHA256'],
        TRAIN_unique=40,TEST_used=False,classification_labels_used=False,fit_updates=0,new_CCD=0,SDK_used=False,GPU_used=False,hardware_changed=False,new_inference_layers=0,model_checkpoint_written=False)
    m.save(a.out/'protocol.json',protocol);m.save(a.out/'per_anchor.json',results)
    report=dict(state='complete',summary=summary,TRAIN_unique=40,TEST_used=False,classification_labels_used=False,fit_updates=0,new_CCD=0,SDK_used=False,GPU_used=False,not_classification_accuracy=True,
        protocol_SHA256=m.sha(a.out/'protocol.json'),per_anchor_SHA256=m.sha(a.out/'per_anchor.json'),elapsed_s=time.monotonic()-start,unique_physical_cause_identified=False)
    m.save(a.out/'report.json',report);m.save(a.out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(a.out/'report.json'),all40_input_canonical_exact=True,source_PT_and_arrays_frozen=True));print(json.dumps(report),flush=True)


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--profile',type=Path,required=True);q.add_argument('--out',type=Path,required=True);run(q.parse_args())
