"""E2-only TRAIN response identification; no deployed graph or LUT mutation.

Reuse the current TUCam contract, not the historical DVP acquisition. Numerical
four-parameter response functions are imported from the published diagnostic.
Fit/held response anchors both belong to TRAIN; no dataset VALID or TEST use.
"""
import argparse,json,os,time,traceback
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def validate_policy(p):
    assert p['schema']=='TUCam_E2_TRAIN_response_v1'
    assert p['capture'].endswith('/e2h/s_2/capture')
    assert p['parent'].endswith('/e2h/mask_fit/best.pt')
    assert p['parent_SHA256']=='4dc922d782d6a7842b9f01c287b791d88a2485a40c602813e562d3c0fa8cd0ba'
    assert p['hardware_SHA256']=='32291e10c2aaf179bc51ab001bf8ad503b8cc759241fed37b4ba0944f91f32e7'
    assert p['phase_SHA256']=='e7e8ec2b6f5e352af0415684b391ef55f880c992e51331191121bd1293cc4a8e'
    assert p['capture_TRAIN_SHA256']=='3c42ab4f2095629bbfd6b42daa5f46775f63f3727f9101e0d993c074f481cf7e'
    assert p['original_TRAIN_per_class']==2 and p['expected_frames']==208
    assert p['conditions']==['nominal_before','flat','half_code','global_offset128','nominal_after']
    assert p['checker_levels']==[0,64,128,192]
    assert p['fit']==dict(steps=160,batch=4,lr=.04,seed=73,CPU_threads=2)
    assert p['admission']==dict(repeat_PCC=.95,repeat_mean_ratio=[.8,1.2],held_loss_ratio=.95,held_coarse_PCC_gain=.02)
    assert p['intermediate_target']==305 and p['final_target']==340
    assert p['TEST_used'] is False and p['TEST_gradient'] is False and p['VALID']==0
    assert p['new_inference_layers']==0 and p['LUT_modified'] is False and p['scheduler'] is False


def anchors(records):
    """Two original TRAIN anchors per class; rank0 fit, rank1 response holdout."""
    rows=[]
    for label in range(20):
        selected=[(i,r) for i,r in enumerate(records) if r['label']==label and r['kind']=='original'][:2]
        assert len(selected)==2
        rows.extend(dict(index=i,sample_id=r['sample_id'],split='fit' if j==0 else 'held_TRAIN') for j,(i,r) in enumerate(selected))
    assert len({r['index'] for r in rows})==40
    return rows


def modules():
    import cv2,numpy as np,torch
    from PIL import Image
    import lab_caltech20_dvp_layerwise_20261019 as m
    from lab_caltech20_tucam_capture_20261023 import validate_hardware
    return cv2,np,torch,Image,m,validate_hardware


def context(p):
    cv2,np,torch,Image,m,validate_hardware=modules()
    import lab_caltech20_tucam_layerwise_20261023 as base
    import lab_caltech20_dvp_TRAIN_response_20261020 as patterns
    assert m.sha(patterns.__file__)==p['patterns_engine_SHA256']
    assert m.sha(base.__file__)==p['suffix_engine_SHA256']
    c=m.read(HERE/p['base_profile']);assert m.sha(HERE/p['base_profile'])==p['base_profile_SHA256']
    cap=ROOT/p['capture'];r=m.read(cap/'report.json');a=m.read(cap/'independent_audit.json')
    assert r['state']=='complete' and r['physical_ccd']==2011 and r['SDK_released']
    assert a['passed'] and a['matched_samples']==2011 and a['report_SHA256']==m.sha(cap/'report.json')
    assert a['protocol_SHA256']==r['protocol_SHA256']==m.sha(cap/'protocol.json')
    assert r['PT_SHA256']==p['parent_SHA256'] and r['phase_SHA256']==p['phase_SHA256']==m.sha(cap/'phase.bmp')
    assert m.sha(cap/'train/ccd.npy')==r['arrays']['train']['SHA256']==p['capture_TRAIN_SHA256']
    hwpath=ROOT/c['hardware_by_stage']['2']['path'];assert m.sha(hwpath)==r['hardware_SHA256']==p['hardware_SHA256']
    hw=validate_hardware(hwpath)
    assert hw['exposure_us']==19998. and hw['orientation']=='identity' and hw['wait_ms']==240 and hw['discard_frames']==6
    protected={str(q):m.sha(q) for q in [hwpath]+[Path(hw[k]) for k in ('coordinates','response_lut','phase_lut')]}
    parent=ROOT/p['parent'];assert m.sha(parent)==p['parent_SHA256']
    reference=ROOT/c['reference'];assert m.sha(reference)==c['reference_SHA256']
    model,_=m.model_for(parent,p['parent_SHA256'],4,reference);model.cpu().eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    plane=model.plane('2');phase=torch.floor(torch.remainder(torch.angle(plane),2*np.pi)/(2*np.pi)*256).clamp(0,255)[20:498,20:498].detach().numpy().astype(np.uint8)
    assert np.array_equal(m.phase_native(phase),np.asarray(Image.open(cap/'phase.bmp')))
    sequence=ROOT/c['sequence'];dm=m.read(sequence/'assets/data_manifest.json');line=sequence/'assets/TRAIN_lineage.json'
    assert dm['sha256']==c['data_SHA256'] and m.sha(line)==dm['lineage_SHA256']
    selected=anchors(m.read(line)['records']);rows=[]
    for row in selected:
        stem=cap/'train'/f"{row['index']:06d}";receipt=m.read(stem.with_suffix('.json'));path=stem.with_name(stem.name+'_input.png')
        assert receipt['sample_id']==row['sample_id'] and receipt['phase_SHA256']==p['phase_SHA256']
        assert receipt['PT_SHA256']==p['parent_SHA256'] and receipt['hardware_SHA256']==p['hardware_SHA256']
        assert m.sha(path)==receipt['canonical_SHA256']
        rows.append(dict(**row,canonical=str(path),canonical_SHA256=m.sha(path)))
    return model,phase,hw,protected,rows


def capture(p,out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    cv2,np,torch,Image,m,_=modules()
    from lab_caltech20_tucam_session_20261022 import Bench,logical
    from lab_caltech20_tucam_L6_20261022 import check_connection
    from lab_caltech20_dvp_TRAIN_response_20261020 import phase_patterns,checker_phase
    torch.set_num_threads(2);model,phase,hw,protected,inputs=context(p)
    assert not out.exists() and not m.LOCK.exists();out.mkdir(parents=True)
    m.save(out/'protocol.json',dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),policy=p,protected=protected,TRAIN_unique=40,TEST_used=False,VALID=0,automatic_retry=False,
        exposure_us=hw['exposure_us'],orientation=hw['orientation'],wait_ms=hw['wait_ms'],discard_frames=hw['discard_frames'],stage='2',depth=4,phase_LUT_scope='linear voltage NOT verified linear phase'))
    warp=cv2.getPerspectiveTransform(np.float32(m.corners(Path(hw['coordinates']))),np.float32([[0,0],[477,0],[477,477],[0,477]]))
    drive=np.asarray(m.read(hw['response_lut'])['drive'],np.uint8);rows=[];start=time.monotonic();restored=False
    with m.LOCK.open('x') as f:json.dump(dict(pid=os.getpid(),camera='tucam',kind='E2_TRAIN_response',out=str(out)),f)
    try:
        for v in inputs:
            gray=np.asarray(Image.open(v['canonical']));bmp=out/f"input_{v['index']:04d}.bmp"
            Image.fromarray(m.native(drive[gray],'amplitude')).save(bmp);v['BMP']=str(bmp)
        nominal=out/'nominal_before_phase.bmp';Image.fromarray(m.phase_native(phase)).save(nominal)
        with Bench(hw,out) as bench:
            try:
                check_connection(bench,hw,warp,out)
                patterns=phase_patterns(phase)+[(f'checker_{v}',checker_phase(v)) for v in p['checker_levels']]
                for tag,pixels in patterns:
                    ph=out/(tag+'_phase.bmp');Image.fromarray(m.phase_native(pixels)).save(ph);bench.stage(ph)
                    for v in inputs if not tag.startswith('checker_') else inputs[:2]:
                        raw,meta=bench.capture(Path(v['BMP']));crop=logical(raw,warp,hw['sensor_full_scale'],hw['orientation'])
                        stem=out/f"{tag}_{v['index']:04d}";rp=stem.with_name(stem.name+'_raw.png');cp=stem.with_suffix('.png')
                        Image.fromarray(raw).save(rp,compress_level=1);Image.fromarray(crop).save(cp)
                        row=dict(**v,condition=tag,raw=str(rp),crop=str(cp),raw_SHA256=m.sha(rp),crop_SHA256=m.sha(cp),input_BMP_SHA256=m.sha(v['BMP']),phase=str(ph),phase_SHA256=m.sha(ph),
                            p99=float(np.percentile(crop,99)),mean=float(crop.mean()),saturation_fraction=float((crop==255).mean()),metadata=meta,captured_at_ns=time.time_ns())
                        m.save(stem.with_suffix('.json'),row);rows.append(row)
                        m.save(out/'progress.json',dict(state='TRAIN_phase_response',frames=len(rows),expected=208,latest=row,elapsed_s=time.monotonic()-start))
                        assert row['p99']>=15 and row['saturation_fraction']<=.15,'Signal guard: preserve, diagnose, no retry or exposure change'
            finally:
                bench.stage(nominal);restored=True
        assert len(rows)==208 and protected=={q:m.sha(q) for q in protected}
        m.save(out/'report.json',dict(state='complete',response_frames=208,TRAIN_unique=40,TEST_used=False,VALID=0,records=rows,protected=protected,
            SDK_released=True,nominal_phase_restored_before_close=restored,hardware_unchanged=True,LUT_modified=False,model_modified=False,
            hardware_SHA256=p['hardware_SHA256'],source_capture_phase_SHA256=p['phase_SHA256'],protocol_SHA256=m.sha(out/'protocol.json'),elapsed_s=time.monotonic()-start))
    except BaseException as e:
        m.save(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),frames=len(rows),nominal_phase_restored=restored,automatic_retry=False));raise
    finally:
        assert m.read(m.LOCK)['pid']==os.getpid();m.LOCK.unlink()


def fit(p,probes,out,contrast=None,spatial_policy=None,polish=None):
    """Fixed 160-step nuisance fit, with TRAIN-only spatial holdout admission."""
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    cv2,np,torch,Image,m,_=modules()
    from lab_caltech20_dvp_TRAIN_response_fit_20261020 import response_intensity,response_parameters,normalized_loss
    from lab_caltech20_dvp_TRAIN_response_20261020 import phase_patterns,checker_phase
    import lab_caltech20_dvp_TRAIN_response_fit_20261020 as kernel
    assert m.sha(kernel.__file__)==p['response_engine_SHA256']
    assert not out.exists() and not m.LOCK.exists();torch.set_num_threads(2);torch.manual_seed(73)
    model,phase,hw,protected,inputs=context(p);r=m.read(probes/'report.json')
    assert r['state']=='complete' and r['response_frames']==208 and r['SDK_released'] and not r['TEST_used']
    assert r['protocol_SHA256']==m.sha(probes/'protocol.json') and r['protected']==protected and r['nominal_phase_restored_before_close']
    assert m.read(probes/'protocol.json')['policy']==p and r['hardware_SHA256']==p['hardware_SHA256']
    out.mkdir();start=time.monotonic();bykey={(v['condition'],v['index']):v for v in r['records']};pack=[];repeat=[];effects=[]
    def image(row):
        assert m.sha(row['crop'])==row['crop_SHA256'] and m.sha(row['raw'])==row['raw_SHA256'] and m.sha(row['phase'])==row['phase_SHA256']
        raw=np.asarray(Image.open(row['raw']));assert raw.dtype==np.uint16 and raw.shape==(2048,2048)
        warp=cv2.getPerspectiveTransform(np.float32(m.corners(Path(hw['coordinates']))),np.float32([[0,0],[477,0],[477,477],[0,477]]))
        from lab_caltech20_tucam_session_20261022 import logical
        crop=np.asarray(Image.open(row['crop']));assert np.array_equal(crop,logical(raw,warp,65535,hw['orientation']))
        return crop.astype(np.float32)
    def coarse(x):return cv2.resize(x,(16,16),interpolation=cv2.INTER_AREA)
    patterns=dict(phase_patterns(phase));patterns.update({f'checker_{v}':checker_phase(v) for v in p['checker_levels']})
    for v in inputs:
        i=v['index'];before=image(bykey['nominal_before',i]);after=image(bykey['nominal_after',i]);ratio=float(after.mean()/max(before.mean(),1e-8))
        repeat.append(dict(index=i,coarse_PCC=m.correlation(coarse(before),coarse(after)),mean_ratio=ratio))
        amp=np.asarray(Image.open(v['canonical'])).astype(np.float32)/255
        tags=['nominal_before','flat','half_code','global_offset128']+([f'checker_{x}' for x in p['checker_levels']] if v in inputs[:2] else [])
        for tag in tags:
            row=bykey[tag,i];assert row['sample_id']==v['sample_id'] and row['canonical_SHA256']==v['canonical_SHA256'] and row['split']==v['split']
            assert np.array_equal(np.asarray(Image.open(row['phase'])),m.phase_native(patterns[tag]))
            pack.append(dict(split=v['split'],amplitude=amp,turns=patterns[tag].astype(np.float32)/256,observed=image(row),index=i,condition=tag))
        effects.append(dict(index=i,split=v['split'],phase_effect_coarse_PCC={tag:m.correlation(coarse(before),coarse(image(bykey[tag,i]))) for tag in ('flat','half_code','global_offset128')}))
    stable=all(v['coarse_PCC']>=.95 and .8<=v['mean_ratio']<=1.2 for v in repeat)
    m.save(out/'stability.json',dict(passed=stable,repeats=repeat,TEST_used=False))
    assert stable,'Within-connection TRAIN repeats unstable: no response fitting/admission'
    train=[v for v in pack if v['split']=='fit'];held=[v for v in pack if v['split']=='held_TRAIN'];assert len(train)==len(held)==84
    def tensors(rows):return tuple(torch.from_numpy(np.stack([v[k] for v in rows])) for k in ('amplitude','turns','observed'))
    tr=tensors(train);ht=tensors(held);nominal=torch.tensor([1.,1.,0.,0.]);initial=torch.tensor([1.,1.,2.,.1]);lo=torch.tensor([.25,.4,0.,0.]);hi=torch.tensor([2.,2.5,12.,3.])
    if spatial_policy is not None:
        return fit_spatial(p,spatial_policy,probes,out,model,tr,ht,train,held,protected,start,polish=polish)
    objective=normalized_loss;prior=None
    if contrast is not None:
        import lab_caltech20_dvp_TRAIN_spatial_response_20261020 as spatial
        assert m.sha(spatial.__file__)==contrast['spatial_engine_SHA256']
        priorpath=ROOT/contrast['prior_fit'];assert m.sha(priorpath)==contrast['prior_fit_SHA256']
        prior=m.read(priorpath);assert prior['state']=='complete' and not prior['response_fit_admitted']
        assert prior['probe_report_SHA256']==m.sha(probes/'report.json') and prior['stability_SHA256']==m.sha(out/'stability.json')
        assert prior['fitted']['held_TRAIN']['loss']<prior['baseline']['held_TRAIN']['loss']*.95 and prior['fitted']['held_TRAIN']['coarse_PCC']<=prior['baseline']['held_TRAIN']['coarse_PCC']
        initial=torch.tensor([prior['parameters'][k] for k in ('phase_scale','phase_gamma','sigma_pixels','relative_background')]);objective=spatial.spatial_loss
    raw=torch.nn.Parameter(torch.logit((initial-lo)/(hi-lo)));opt=torch.optim.Adam([raw],lr=.04);history=[]
    m.save(out/'protocol.json',dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),response_engine_SHA256=m.sha(kernel.__file__),policy=p,probe_report_SHA256=m.sha(probes/'report.json'),
        TRAIN_fit_pairs=84,held_TRAIN_pairs=84,TEST_used=False,TEST_labels_used=False,VALID=0,training_forward_only=True,new_inference_layers=0,SDK_used=False,
        contrast_policy=contrast,objective='multiscale_contrast' if contrast else 'normalized_field',parameter_count=4))
    def evaluate(values,params):
        total=0.;pcc=[]
        with torch.no_grad():
            for i in range(0,len(values[0]),4):
                aa,pp,oo=(v[i:i+4] for v in values);pred=response_intensity(aa,pp,model.propagator,params)
                total+=float(objective(pred,oo))*len(aa)
                for x,y in zip(pred.numpy(),oo.numpy()):pcc.append(m.correlation(coarse(x),coarse(y)))
        return dict(loss=total/len(values[0]),coarse_PCC=float(np.mean(pcc)))
    baseline=dict(TRAIN=evaluate(tr,nominal),held_TRAIN=evaluate(ht,nominal));start_metrics=dict(TRAIN=evaluate(tr,initial),held_TRAIN=evaluate(ht,initial))
    for step in range(1,161):
        ix=torch.randperm(84)[:4];aa,pp,oo=(v[ix] for v in tr);opt.zero_grad(set_to_none=True)
        loss=objective(response_intensity(aa,pp,model.propagator,response_parameters(raw)),oo);assert torch.isfinite(loss)
        loss.backward();assert torch.isfinite(raw.grad).all();opt.step()
        if step%20==0:
            row=dict(step=step,TRAIN_batch_loss=float(loss.detach()),parameters=response_parameters(raw).detach().tolist(),elapsed_s=time.monotonic()-start)
            history.append(row);m.save(out/'history.json',history);m.save(out/'progress.json',row)
    params=response_parameters(raw).detach();final=dict(TRAIN=evaluate(tr,params),held_TRAIN=evaluate(ht,params))
    admitted=final['TRAIN']['loss']<baseline['TRAIN']['loss'] and final['held_TRAIN']['loss']<=baseline['held_TRAIN']['loss']*.95 and final['held_TRAIN']['coarse_PCC']>=baseline['held_TRAIN']['coarse_PCC']+.02
    if contrast is not None:admitted=admitted and final['held_TRAIN']['coarse_PCC']>=start_metrics['held_TRAIN']['coarse_PCC']+.02
    assert protected=={q:m.sha(q) for q in protected}
    m.save(out/'phase_effects.json',dict(TRAIN_only=True,records=effects,phase_effect_coarse_PCC_median={tag:float(np.median([v['phase_effect_coarse_PCC'][tag] for v in effects])) for tag in ('flat','half_code','global_offset128')},unique_physical_cause_identified=False))
    m.save(out/'report.json',dict(state='complete',response_fit_admitted=admitted,baseline=baseline,fitted=final,parameters=dict(zip(('phase_scale','phase_gamma','sigma_pixels','relative_background'),params.tolist())),
        TRAIN_unique=40,TRAIN_fit_pairs=84,held_TRAIN_pairs=84,TEST_used=False,TEST_labels_used=False,VALID=0,training_forward_only=True,new_inference_layers=0,
        objective='multiscale_contrast' if contrast else 'normalized_field',parameter_count=4,initial=start_metrics,contrast_policy=contrast,phase_effects_SHA256=m.sha(out/'phase_effects.json'),
        deployed_architecture_unchanged=True,physical_LUT_calibrated=False,LUT_modified=False,hardware_unchanged=True,SDK_used=False,SDK_released=not m.LOCK.exists(),
        protocol_SHA256=m.sha(out/'protocol.json'),probe_report_SHA256=m.sha(probes/'report.json'),stability_SHA256=m.sha(out/'stability.json'),history_SHA256=m.sha(out/'history.json'),elapsed_s=time.monotonic()-start,
        interpretation='Empirical nuisance fit, not unique physical identification or classification improvement; unadmitted fits must not enter training forward.'))


def selftest():
    from lab_caltech20_dvp_TRAIN_response_fit_20261020 import selftest as numerical
    numerical()
    rows=[dict(label=k,kind='original',sample_id=str(k)+'a') for k in range(20)]+[dict(label=k,kind='original',sample_id=str(k)+'b') for k in range(20)]
    selected=anchors(rows);assert len(selected)==40 and sum(v['split']=='fit' for v in selected)==20


def validate_contrast(c):
    assert c['schema']=='TUCam_E2_TRAIN_response_contrast_v1'
    assert c['prior_fit'].endswith('/e2_response_20261024/fit/report.json')
    assert c['prior_fit_SHA256']=='a2a18b63d6064d01cc3af91fd39c908bc7dd7435f134624a49883a91c4940587'
    assert c['probes'].endswith('/e2_response_20261024/probes')
    assert c['fit_steps']==160 and c['shared_parameters']==4 and c['new_CCD']==0 and c['TEST_used'] is False


def validate_spatial(c):
    assert c['schema']=='TUCam_E2_TRAIN_response_spatial_v1'
    assert c['prior_fit'].endswith('/e2_response_20261024/contrast_fit/report.json')
    assert c['prior_fit_SHA256']=='8a83995b14d568f56b1cb8596c6ffc4c1ce36614e9d5837fe761d2dda6594d66'
    assert c['probes'].endswith('/e2_response_20261024/probes')
    assert c['fit_steps']==400 and c['shared_parameters']==27 and c['new_CCD']==0 and c['TEST_used'] is False
    assert c['held_PCC_gain']==.03 and c['held_loss_ratio']==.95 and c['nominal_held_PCC_not_lower'] is True
    assert c['image_warp'] is False and c['new_inference_layers']==0


def validate_polish(c):
    assert c['schema']=='TUCam_E2_TRAIN_response_fullbatch_polish_v1'
    assert c['prior_fit'].endswith('/e2_response_20261024/spatial_fit/report.json')
    assert c['prior_fit_SHA256']=='a9ebe0129680bde1366888b9d79879b94c0843d9c02eb705e3dd9e1d2ec0a234'
    assert c['max_iter']==12 and c['max_eval']==24 and c['new_CCD']==0 and c['TEST_used'] is False
    assert c['gate_unchanged'] is True and c['shared_parameters']==27


def fit_spatial(p,c,probes,out,model,tr,held,train_rows,held_rows,protected,start,polish=None):
    """Monotonic phase/pupil nuisance response; never a deployed network layer."""
    cv2,np,torch,Image,m,_=modules()
    import lab_caltech20_dvp_TRAIN_spatial_response_20261020 as spatial
    from lab_caltech20_dvp_TRAIN_response_fit_20261020 import response_intensity
    assert m.sha(spatial.__file__)==c['spatial_engine_SHA256']
    priorpath=ROOT/c['prior_fit'];assert m.sha(priorpath)==c['prior_fit_SHA256']
    prior=m.read(priorpath);assert prior['state']=='complete' and not prior['response_fit_admitted']
    assert prior['probe_report_SHA256']==m.sha(probes/'report.json') and not prior['TEST_used']
    assert prior['fitted']['held_TRAIN']['coarse_PCC']<prior['baseline']['held_TRAIN']['coarse_PCC']+.02
    pars=prior['parameters'];old=torch.tensor([pars[k] for k in ('phase_scale','phase_gamma','sigma_pixels','relative_background')])
    x=torch.linspace(0,1,17);steps=(x[1:].pow(pars['phase_gamma'])-x[:-1].pow(pars['phase_gamma']))*16
    knots=torch.nn.Parameter(torch.log(torch.expm1(steps)));scale=torch.nn.Parameter(torch.logit(torch.tensor((pars['phase_scale']-.25)/1.75)))
    sigma=torch.nn.Parameter(torch.logit(torch.tensor(pars['sigma_pixels']/12)));background=torch.nn.Parameter(torch.logit(torch.tensor(pars['relative_background']/3)))
    coefficients=torch.nn.Parameter(torch.zeros(8));parameters=[knots,scale,sigma,background,coefficients]
    if polish is not None:
        priorpath=ROOT/polish['prior_fit'];assert m.sha(priorpath)==polish['prior_fit_SHA256']
        prev=m.read(priorpath);assert prev['state']=='complete' and not prev['spatial_response_admitted'] and not prev['TEST_used']
        assert prev['probe_report_SHA256']==m.sha(probes/'report.json') and prev['prior_fit_SHA256']==c['prior_fit_SHA256']
        curve=torch.tensor(prev['phase_curve_radians']);ratio=(curve[-1]/(2*torch.pi)-.25)/1.75
        with torch.no_grad():
            knots.copy_(torch.log(torch.expm1(curve[1:]-curve[:-1])))
            scale.copy_(torch.logit(ratio.clamp(1e-6,1-1e-6)))
            sigma.copy_(torch.logit(torch.tensor(prev['sigma_pixels']/12).clamp(1e-6,1-1e-6)))
            background.copy_(torch.logit(torch.tensor(prev['relative_background']/3).clamp(1e-6,1-1e-6)))
            coefficients.copy_(torch.atanh((torch.tensor(prev['pupil_coefficients_radians'])/(12*torch.pi)).clamp(-1+1e-6,1-1e-6)))
    assert sum(v.numel() for v in parameters)==27
    def physical():return spatial.monotonic_curve(knots,scale),12*sigma.sigmoid(),3*background.sigmoid(),12*torch.pi*coefficients.tanh()
    def evaluate(values,rows,baseline=False):
        total=0.;pcc={}
        with torch.no_grad():
            for i in range(0,len(values[0]),4):
                aa,pp,oo=(v[i:i+4] for v in values)
                pred=response_intensity(aa,pp,model.propagator,old) if baseline else spatial.spatial_intensity(aa,pp,model.propagator,*physical())
                total+=float(spatial.spatial_loss(pred,oo))*len(aa)
                for j,(xx,yy) in enumerate(zip(pred.numpy(),oo.numpy())):
                    cc=m.correlation(cv2.resize(xx,(16,16),interpolation=cv2.INTER_AREA),cv2.resize(yy,(16,16),interpolation=cv2.INTER_AREA))
                    pcc.setdefault(rows[i+j]['condition'],[]).append(cc)
        return dict(loss=total/len(values[0]),coarse_PCC=float(np.mean([x for vs in pcc.values() for x in vs])),PCC_by_condition={k:float(np.mean(v)) for k,v in pcc.items()})
    baseline=dict(TRAIN=evaluate(tr,train_rows,True),held_TRAIN=evaluate(held,held_rows,True));initial=dict(TRAIN=evaluate(tr,train_rows),held_TRAIN=evaluate(held,held_rows))
    m.save(out/'protocol.json',dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),policy=c,response_policy=p,probe_report_SHA256=m.sha(probes/'report.json'),prior_fit_SHA256=m.sha(priorpath),
        TRAIN_fit_pairs=84,held_TRAIN_pairs=84,TRAIN_unique=40,labels_used=False,TEST_used=False,VALID=0,steps=400,batch=4,Adam_lr=.02,shared_parameters=27,phase_knots=17,pupil_modes=8,
        image_warp=False,coordinates_changed=False,LUT_written=False,model_checkpoint_written=False,training_forward_only=True,nominal_held_PCC_must_not_regress=True,polish_policy=polish,
        actual_optimizer='full84TRAIN_LBFGS12iter24eval' if polish else 'Adam400batch4'))
    history=[];closure_evaluations=0
    if polish is not None:
        opt=torch.optim.LBFGS(parameters,lr=1,max_iter=12,max_eval=24,history_size=12,line_search_fn='strong_wolfe')
        def closure():
            nonlocal closure_evaluations
            opt.zero_grad(set_to_none=True);total=0.
            for i in range(0,84,4):
                aa,pp,oo=(v[i:i+4] for v in tr);pred=spatial.spatial_intensity(aa,pp,model.propagator,*physical())
                loss=spatial.spatial_loss(pred,oo)*len(aa)/84
                assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())
            penalty=.0001*coefficients.square().sum();penalty.backward();total+=float(penalty.detach())
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in parameters)
            closure_evaluations+=1;row=dict(closure=closure_evaluations,TRAIN_response_loss=total,TRAIN_response_pairs=84,TRAIN_gradient_norm=float(torch.cat([v.grad.flatten() for v in parameters]).norm()),elapsed_s=time.monotonic()-start)
            history.append(row);m.save(out/'history.json',history);m.save(out/'progress.json',row)
            return parameters[0].new_tensor(total)
        opt.step(closure);step=int(opt.state[parameters[0]]['n_iter'])
    else:
        opt=torch.optim.Adam(parameters,lr=.02)
        for step in range(1,401):
            ix=torch.randperm(84)[:4];aa,pp,oo=(v[ix] for v in tr);opt.zero_grad(set_to_none=True)
            pred=spatial.spatial_intensity(aa,pp,model.propagator,*physical());loss=spatial.spatial_loss(pred,oo)+.0001*coefficients.square().sum()
            assert torch.isfinite(loss);loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in parameters);opt.step()
            if step%50==0:
                row=dict(step=step,TRAIN_batch_spatial_loss=float(loss.detach()),elapsed_s=time.monotonic()-start)
                history.append(row);m.save(out/'history.json',history);m.save(out/'progress.json',row)
    final=dict(TRAIN=evaluate(tr,train_rows),held_TRAIN=evaluate(held,held_rows));curve,psf,bg,coeff=physical()
    b,f=baseline['held_TRAIN'],final['held_TRAIN']
    admitted=final['TRAIN']['loss']<baseline['TRAIN']['loss'] and f['loss']<=b['loss']*.95 and f['coarse_PCC']>=b['coarse_PCC']+.03 and f['PCC_by_condition']['nominal_before']>=b['PCC_by_condition']['nominal_before']
    if polish is not None:admitted=admitted and final['TRAIN']['loss']<initial['TRAIN']['loss']
    assert protected=={q:m.sha(q) for q in protected}
    m.save(out/'report.json',dict(state='complete',spatial_response_admitted=admitted,baseline=baseline,fitted=final,phase_curve_radians=curve.detach().tolist(),sigma_pixels=float(psf.detach()),relative_background=float(bg.detach()),pupil_coefficients_radians=coeff.detach().tolist(),
        shared_parameters=27,steps=step,closure_evaluations=closure_evaluations,initial=initial,polish_policy=polish,TRAIN_unique=40,TRAIN_fit_pairs=84,held_TRAIN_pairs=84,labels_used=False,TEST_used=False,VALID=0,training_forward_only=True,image_warp=False,hardware_unchanged=True,LUT_written=False,model_checkpoint_written=False,new_inference_layers=0,
        protocol_SHA256=m.sha(out/'protocol.json'),probe_report_SHA256=m.sha(probes/'report.json'),prior_fit_SHA256=m.sha(priorpath),history_SHA256=m.sha(out/'history.json'),SDK_used=False,SDK_released=not m.LOCK.exists(),unique_physical_cause_identified=False,elapsed_s=time.monotonic()-start))


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--mode',choices=('selftest','capture_fit','contrast_fit','spatial_fit','spatial_polish'),required=True);q.add_argument('--profile',type=Path);q.add_argument('--out',type=Path);a=q.parse_args()
    if a.mode=='selftest':selftest()
    else:
        if a.mode=='spatial_polish':
            v=json.loads(a.profile.read_text());validate_polish(v);assert not a.out.exists();c=json.loads((HERE/v['spatial_profile']).read_text());validate_spatial(c)
            p=json.loads((HERE/c['response_profile']).read_text());validate_policy(p);fit(p,ROOT/c['probes'],a.out,spatial_policy=c,polish=v)
        elif a.mode in ('contrast_fit','spatial_fit'):
            c=json.loads(a.profile.read_text());(validate_contrast if a.mode=='contrast_fit' else validate_spatial)(c);assert not a.out.exists()
            p=json.loads((HERE/c['response_profile']).read_text());validate_policy(p)
            fit(p,ROOT/c['probes'],a.out,**({'contrast':c} if a.mode=='contrast_fit' else {'spatial_policy':c}))
        else:
            p=json.loads(a.profile.read_text());validate_policy(p);assert not a.out.exists();a.out.mkdir(parents=True)
            try:
                capture(p,a.out/'probes');fit(p,a.out/'probes',a.out/'fit')
                _,_,_,_,m,_=modules();r=m.read(a.out/'fit/report.json');m.save(a.out/'report.json',dict(state='complete',response_fit_admitted=r['response_fit_admitted'],probe_frames=208,fit_report_SHA256=m.sha(a.out/'fit/report.json'),TEST_used=False,new_inference_layers=0,SDK_released=not m.LOCK.exists(),G2_not_started=True))
            except BaseException as e:
                (a.out/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False),indent=2));raise
